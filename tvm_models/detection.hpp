#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <random>
#include <torch/torch.h>
#include "clipp.hpp"
#include <cstdio>

double sum_score, sum_fps;

std::vector<std::string> COCO_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

namespace args {
static std::string image = "./people.jpg";
static bool gpu = false;
static bool libtorch = false;
static bool quite = false;
static bool no_display = false;
static float viz_thresh = 0.3;
static int min_size = 512;
static int max_size = 512;
static int multiplier = 32;  // just to ensure image shapes are multipliers of feature strides, for yolo3 models
}  // namespace args

void ParseArgs(int argc, char** argv) {
    using namespace clipp;

    auto cli = (
        value("image file", args::image),
        option("--gpu").set(args::gpu).doc("to use gpu or not, by default it is false"),
        option("--libtorch").set(args::libtorch).doc("to use libtorch or not, by default it is false"),
        (option("-t", "--thresh") & number("thresh", 0.3)) % "Visualize threshold, from 0 to 1, default 0.3."
    );
    if (!parse(argc, argv, cli) || args::image.empty()) {
        std::cout << make_man_page(cli, argv[0]);
        exit(-1);
    }
}

std::string format_frame_number(std::string filename, int frame_number) {
  char buffer[13];
  std::snprintf(buffer, sizeof(buffer), "%08d.jpg", frame_number);
  std::string file_name = filename + buffer;
  return file_name;
}

cv::Mat image_for_plot(cv::Mat image){
       
    cv::Size new_size = cv::Size(640, 480);
    cv::Mat resized_image;
    // bgr to rgb
    cv::resize(image, resized_image, new_size);
    return resized_image;
}

struct TensorOutput {
    DLTensor *output_tensor_ids;
    DLTensor *output_tensor_scores;
    DLTensor *output_tensor_bboxes;
};
struct TensorInput {
    DLTensor *input;
    float *data_x;
};

// box
struct box{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

// adjBox
struct bbox_result{
    int id;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};
// boxInfo
struct yoloresults{
    bbox_result* boxes;
    int num;
};

namespace viz {
// convert color from hsv to bgr for plotting
inline cv::Scalar HSV2BGR(cv::Scalar hsv) {
    cv::Mat from(1, 1, CV_32FC3, hsv);
    cv::Mat to;
    cv::cvtColor(from, to, cv::COLOR_HSV2BGR);
    auto pixel = to.at<cv::Vec3f>(0, 0);
    unsigned char b = static_cast<unsigned char>(pixel[0] * 255);
    unsigned char g = static_cast<unsigned char>(pixel[1] * 255);
    unsigned char r = static_cast<unsigned char>(pixel[2] * 255);
    return cv::Scalar(b, g, r);
}

inline void PutLabel(cv::Mat &im, const std::string label, const cv::Point & orig, cv::Scalar color) {
    int fontface = cv::FONT_HERSHEY_DUPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;
    double alpha = 0.6;
    //std::cout << "getting size: \n";
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    // make sure roi inside image region
    //std::cout << "setting region: \n";
    cv::Rect blend_rect = cv::Rect(orig + cv::Point(0, baseline),
        orig + cv::Point(text.width, -text.height)) & cv::Rect(0, 0, im.cols, im.rows);
    //std::cout << "making roi: \n";
    cv::Mat roi = im(blend_rect);
    //std::cout << "blending: \n";
    cv::Mat blend(roi.size(), CV_8UC3, color);
    // cv::rectangle(im, orig + cv::Point(0, baseline), orig + cv::Point(text.width, -text.height), CV_RGB(0, 0, 0), CV_FILLED);
    //std::cout << "adding weight: \n";
    cv::addWeighted(blend, alpha, roi, 1.0 - alpha, 0.0, roi);
    //std::cout << "putting text: \n";
    cv::putText(im, label, orig, fontface, scale, cv::Scalar(255, 255, 255), thickness, 8);
}

// plot bounding boxes on raw image
inline cv::Mat PlotBbox(cv::Mat img, yoloresults* results,
               float thresh, std::vector<std::string> class_names,
               std::map<int, cv::Scalar> colors, bool verbose) {
    std::mt19937 eng;
    std::uniform_real_distribution<float> rng(0, 1);
    float hue = rng(eng);
    //if (verbose) {
    std::cout << "Start Ploting with visualize score threshold: \n" << thresh << std::endl;
    //}
    for (int i = 0; i < results->num; ++i) {
        std::cout << "getting info from boxes start: \n";
        float label = static_cast<float>(results->boxes[i].id);
        //std::cout << "getting info from scores/labels end: ";
        if (results->boxes[i].score < thresh) continue;
        if (results->boxes[i].id < 0) continue;

        if (colors.find(results->boxes[i].id) == colors.end()) {
            // create a new color
            int csize = static_cast<int>(class_names.size());
            if (class_names.size() > 0) {
                float hue = label / csize;
                colors[results->boxes[i].id] = HSV2BGR(cv::Scalar(hue * 255, 0.75, 0.95));//cv::Scalar(0, 255, 0); //HSV2BGR(cv::Scalar(hue * 255, 0.75, 0.95));
            } else {
                // generate color for this id
                hue += 0.618033988749895;  // golden ratio
                hue = fmod(hue, 1.0);
                colors[results->boxes[i].id] = HSV2BGR(cv::Scalar(hue * 255, 0.75, 0.95));//cv::Scalar(0, 255, 0); //HSV2BGR(cv::Scalar(hue * 255, 0.75, 0.95));
            }
        }

        // draw bounding box
        //std::cout << "drawing boxes: \n";
        auto color = colors[results->boxes[i].id];
        cv::Point pt1(results->boxes[i].xmin, results->boxes[i].ymin);
        cv::Point pt2(results->boxes[i].xmax, results->boxes[i].ymax);
        cv::rectangle(img, pt1, pt2, color, 2);

        if (verbose) {
            if (results->boxes[i].id >= class_names.size()) {
                std::cout << "id: " << results->boxes[i].id << ", scores: " << results->boxes[i].score << "/n" << std::endl;
            } else {
                std::cout << "id: " << class_names[results->boxes[i].id] << ", scores: " << results->boxes[i].score << "/n" <<  std::endl;
            }

        }

        // put text
        //std::cout << "putting text: \n";
        std::string txt;
        if (class_names.size() > results->boxes[i].id) {
            txt += class_names[results->boxes[i].id];
        }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << results->boxes[i].score;
        txt += " " + ss.str();
        //std::cout << "using PutLabel with label: \n";
        PutLabel(img, txt, pt1, color);
        //std::cout << "done with label: \n";
    }
    return img;
}
}// namespace viz

class YoloTVM{
    private:
        //working: void * handle;
        std::unique_ptr<tvm::runtime::Module> handle;

    public:
        int deploy_device_id;
        int deploy_device_type;
        std::string deploy_lib_path;
        std::string deploy_graph_path;
        std::string deploy_param_path;
        int max_boxes = 100;
        int width = 512;
        int height = 512;
        static constexpr int deploy_dtype_code = kDLFloat;
        static constexpr int deploy_dtype_bits = 32;
        static constexpr int deploy_dtype_lanes = 1;
        static constexpr int deploy_in_ndim = 4;
        static constexpr int deploy_out_ndim = 3;
        static constexpr int max_yolo_boxes = 100;
        static constexpr int64_t deploy_in_shape[deploy_in_ndim] = {1, 3, 512, 512};
        static constexpr int64_t deploy_tvm_id_and_score_size[deploy_out_ndim] = {1, 100, 1};
        static constexpr int64_t deploy_tvm_box_size[deploy_out_ndim] = {1, 100, 4};    

        YoloTVM(std::string model_folder) {
            // tvm module for compiled functions
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_folder + "/deploy_lib_gpu.so");
            // json graph
            std::ifstream json_in(model_folder + "/deploy_graph_gpu.json", std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();

            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int device_type = kDLGPU;//kDLCPU;//kDLGPU
            int device_id = 1;
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                  device_type, device_id);
            this->handle.reset(new tvm::runtime::Module(mod));
            // parameters in binary
            std::ifstream params_in(model_folder + "/deploy_param_gpu.params", std::ios::binary);
            std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
            params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray params_arr;
            params_arr.data = params_data.c_str();
            params_arr.size = params_data.length();
            tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
            load_params(params_arr);
        }

        cv::Mat preprocess_image(cv::Mat frame){
            cv::Size new_size = cv::Size(512, 512);
            cv::Mat resized_image;
            cv::Mat rgb;
            // bgr to rgb
            cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
            // resize to 512x512
            cv::resize(rgb, resized_image, new_size);
            cv::Mat resized_image_floats(new_size, CV_32FC3);
            // convert resized image to floats and normalize
            resized_image.convertTo(resized_image_floats, CV_32FC3, 1.0f/255.0f);
            //mimic mxnets 'to_tensor' function 
            cv::Mat normalized_image(new_size, CV_32FC3);
            // mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

            // these values can also be set from somewhere if need be
            // but are static for now
            cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
            cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
            cv::Mat temp;
            temp = resized_image_floats - mean;
            normalized_image = temp / theta;
            return normalized_image; 
        }

        yoloresults* forward_full(cv::Mat frame, float thresh)
        {
            
            //Set constants and variables
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLGPU;
            constexpr int device_id = 1;
            int in_ndim = 4;
            int out_ndim = 3;
            int in_c = 3, in_h = 512, in_w = 512;
            int ratio_x = 1, ratio_y = 1;
            int64_t in_shape[4] = {1, in_c, in_h, in_w};
            int64_t tvm_id_and_score_size[3] = {1, 100, 1};
            int64_t tvm_box_size[3] = {1, 100, 4};
            int total_input = 3 * in_w * in_h;
            DLTensor *output_tensor_ids;
            DLTensor *output_tensor_scores;
            DLTensor *output_tensor_bboxes;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
            
            // allocate memory for results
            yoloresults* results = (yoloresults*)calloc(1, sizeof(yoloresults));
            results->num = 100;
            results->boxes = (bbox_result*)calloc(100, sizeof(bbox_result));

            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_ids);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_scores);
            TVMArrayAlloc(tvm_box_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_bboxes);

            //copy processed image to DLTensor
            cv::Mat processed_image = preprocess_image(frame);
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));

            // standard tvm module run 
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");

            // https://github.com/apache/incubator-tvm/issues/979?from=timeline
            //"This may give you some ideas to start with. 
            //In general you want to use pinned memory and you want 
            //to interleave computation with copying; so you want to
            // be upload the next thing while you are computing the 
            //current thing while you are downloading the last thing."
            TVMSynchronize(device_type, device_id, nullptr);
            get_output(0, output_tensor_ids);
            get_output(1, output_tensor_scores);
            get_output(2, output_tensor_bboxes);

            torch::Tensor ndarray_ids = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_scores = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_bboxes = torch::zeros({1, 100, 4}, at::kFloat);

            TVMArrayCopyToBytes(output_tensor_ids, ndarray_ids.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_scores, ndarray_scores.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_bboxes, ndarray_bboxes.data_ptr(),  1 * 100 * 4 * sizeof(float));

            auto ndarray_scores_a = ndarray_scores.accessor<float,3>();
            auto ndarray_ids_a = ndarray_ids.accessor<float,3>();
            auto ndarray_bboxes_a = ndarray_bboxes.accessor<float,3>();

            int new_num = 0;
            //int num = 100;
            for (int i = 0; i < max_yolo_boxes; ++i) {
                float xmin;
                float ymin;
                float xmax;
                float ymax;

                float score = ndarray_scores_a[0][i][0]; //TODO change 00i
                float label = ndarray_ids_a[0][i][0];
                if (score < thresh) continue;
                if (label < 0) continue;

                int cls_id = static_cast<int>(label);
                xmin = ndarray_bboxes_a[0][i][0];
                ymin = ndarray_bboxes_a[0][i][1];
                xmax = ndarray_bboxes_a[0][i][2];
                ymax = ndarray_bboxes_a[0][i][3];

                results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space
                results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                results->boxes[i].xmax = xmax * (640.0/512.0);
                results->boxes[i].ymax = ymax / (512.0/480.0);
                results->boxes[i].id = cls_id;
                results->boxes[i].score = score;
                new_num+=1;
            };
            results->num = new_num;

            TVMArrayFree(input);
            TVMArrayFree(output_tensor_ids);
            TVMArrayFree(output_tensor_scores);
            TVMArrayFree(output_tensor_bboxes);
            input = nullptr;
            output_tensor_ids = nullptr;
            output_tensor_scores = nullptr;
            output_tensor_bboxes = nullptr;
            free(data_x);
            data_x = nullptr;

            return results;
    }
};    


int main(int argc, char** argv)
{
    ParseArgs(argc, argv);
    double fps, current;
    char string[10];
    char buff[10];
    using Clock = std::chrono::high_resolution_clock;
    using Timepoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;
    std::string image_name = args::image;
    cv::Mat raw_image;
    cv::Mat processed_image;
    cv::Mat plot_image;
    yoloresults* result;
    yoloresults* timer_results;
    
    raw_image = cv::imread(args::image);
    std::cout << "imread: " << std::endl;
    //processed_image = preprocess_input(raw_image);
    //plot_image = image_for_plot(raw_image);

    YoloTVM model("./model_folder");
    std::cout << "model time: " << std::endl;

    // WARM UP
    //auto tic = Clock::now();
    result = model.forward_full(raw_image, .5);
    //auto toc = Clock::now();
    //auto elapsed = Duration(toc - tic).count();


    cv::VideoCapture capture("/home/mkrzus/Videos/burroughs.mp4.mp4");
    yoloresults* frame_result;
    cv::Mat frame, frame_r;
    cv::Mat processed_frame;
    cv::Mat plot_frame;
    int frame_count = 1;
    capture >> frame;
    if(!frame.data) {
        std::cout<< "read first frame failed!";
        exit(1);
    }
    cv::namedWindow("Frame");

    std::cout << "frame resolution: " << frame.cols << "*" << frame.rows << "\n";

    char keyboard = 0;
    bool stop=false;
    while( keyboard != 'q' && keyboard != 27 ){
        keyboard = (char)cv::waitKey( 30 );
        if(stop){
            imshow("Frame", frame_r);
            if(keyboard==32) stop = false;
            continue;
        } else if(keyboard==32){
            stop = true;
            continue;
        }
        double t = (double) cv::getTickCount();
        capture >> frame;
        if(!frame.data)   break;
        frame_count++;
        // processed_frame = preprocess_input(frame);
        cv::Mat frame_copy = frame.clone();
        plot_frame = image_for_plot(frame_copy);
        //frame_result = model.forward(processed_frame, .5);
        auto tic = Clock::now();
        frame_result = model.forward_full(frame, .5);
        auto toc = Clock::now();
        auto elapsed = Duration(toc - tic).count();
        std::cout << "full forward time: " << elapsed << std::endl;
        auto output_image = viz::PlotBbox(plot_frame, frame_result, 0.3, COCO_CLASS_NAMES, std::map<int, cv::Scalar>(), true);
        t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        sum_fps += fps;
        sprintf(string, "%.2f", fps);
        std::string fpsString("FPS: ");
        fpsString += string;
        cv::putText(output_image, fpsString, cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        std::string framecount("Frame: ");
        framecount += std::to_string(frame_count);
        cv::putText(output_image, framecount, cv::Point(5, 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        cv::imshow("Frame", output_image);
        cv::waitKey(1);

        std::string frame_name = format_frame_number("/home/mkrzus/github/open_ptrack_lite/tests/frames/", frame_count);
        std::cout<< frame_name << "\n" << std::endl;
        cv::imwrite(frame_name, output_image);
    }


    auto plt = viz::PlotBbox(plot_image, result, 0.3, COCO_CLASS_NAMES, std::map<int, cv::Scalar>(), true);
    cv::imwrite("OUTPUTIMAGE.JPG", plt);
    return 0;
}