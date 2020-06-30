#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/dnn/dnn.hpp>
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
#include <cstdio>
#include <nlohmann/json.hpp>
#include <open_ptrack/nms/nms.h>
using json = nlohmann::json;

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

// adjBox
struct pose_result{
    int id;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    // actually point2d, with z being the confidence
    std::vector<cv::Point3f> points;
};

// boxInfo
struct pose_results{
    pose_result* boxes;
    int num;
};


class YoloTVMGPU256{
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
        // we have have some kind of way to make this configureable
        int width = 256;
        int height = 256;
        static constexpr int deploy_dtype_code = kDLFloat;
        static constexpr int deploy_dtype_bits = 32;
        static constexpr int deploy_dtype_lanes = 1;
        static constexpr int deploy_in_ndim = 4;
        static constexpr int deploy_out_ndim = 3;
        static constexpr int max_yolo_boxes = 100;
        static constexpr int64_t deploy_in_shape[deploy_in_ndim] = {1, 3, 256, 256};
        static constexpr int64_t deploy_tvm_id_and_score_size[deploy_out_ndim] = {1, 100, 1};
        static constexpr int64_t deploy_tvm_box_size[deploy_out_ndim] = {1, 100, 4};    

        YoloTVMGPU256(std::string model_folder) {
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
            int device_id = 0;
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
            cv::Size new_size = cv::Size(256, 256);
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
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            using Clock = std::chrono::high_resolution_clock;
            using Timepoint = Clock::time_point;
            using Duration = std::chrono::duration<double>;
            auto start = Clock::now();

            //Set constants and variables
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLGPU;
            constexpr int device_id = 0;
            int in_ndim = 4;
            int out_ndim = 3;
            int in_c = 3, in_h = 256, in_w = 256;
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

            auto allocstart = Clock::now();
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_ids);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_scores);
            TVMArrayAlloc(tvm_box_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_bboxes);
            auto allocend = Clock::now();
            auto tvmalloc_elapsed = Duration(allocend - allocstart).count();
            std::cout << "TVMArrayAlloc time elapsed: " << tvmalloc_elapsed << std::endl;

            auto copyfrom_start = Clock::now();
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
            auto copyfrom_end = Clock::now();
            auto tvmcopy_elapsed = Duration(copyfrom_end - copyfrom_start).count();
            std::cout << "TVMArrayCopyFromBytes + processing image time elapsed: " << tvmcopy_elapsed << std::endl;

            auto tvmruntime_start = Clock::now();
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

            auto tvmruntime_end = Clock::now();
            auto tvmruntime_elapsed = Duration(tvmruntime_end - tvmruntime_start).count();
            std::cout << "tvm_runtime_functs: mod, set_input, run, getouput time elapsed: " << tvmruntime_elapsed << std::endl;


            auto toc2 = Clock::now();
            auto elapsed2 = Duration(toc2 - start).count();
            std::cout << "tvm setup/model-runtime/getoutput time elapsed: " << elapsed2 << std::endl;

            auto torchboxsstart = Clock::now();
            torch::Tensor ndarray_ids = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_scores = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_bboxes = torch::zeros({1, 100, 4}, at::kFloat);

            auto tvmarraycopystart = Clock::now();
            TVMArrayCopyToBytes(output_tensor_ids, ndarray_ids.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_scores, ndarray_scores.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_bboxes, ndarray_bboxes.data_ptr(),  1 * 100 * 4 * sizeof(float));
            auto tvmarraycopyend = Clock::now();
            auto tvmarraycopyend_elapsed = Duration(tvmarraycopyend - tvmarraycopystart).count();
            std::cout << "TVMArrayCopyToBytes time elapsed: " << tvmarraycopyend_elapsed << std::endl;


            auto torchboxsend = Clock::now();
            auto torchboxsend_elapsed = Duration(torchboxsend - torchboxsstart).count();
            std::cout << "torch inital tensor creation and copytobytes time elapsed: " << torchboxsend_elapsed << std::endl;

            auto accessor_start = Clock::now();
            auto ndarray_scores_a = ndarray_scores.accessor<float,3>();
            auto ndarray_ids_a = ndarray_ids.accessor<float,3>();
            auto ndarray_bboxes_a = ndarray_bboxes.accessor<float,3>();
            auto accessor_end = Clock::now();
            auto accessor_time_elapsed = Duration(accessor_end - accessor_start).count();
            std::cout << "accessor_time elapsed: " <<accessor_time_elapsed << std::endl;



            auto for_loop_start = Clock::now();
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


                results->boxes[i].xmin = xmin * (img_width/256.0); // move down to 480 space  ()
                results->boxes[i].ymin = ymin / (256.0/img_height); // move up to 640
                results->boxes[i].xmax = xmax * (img_width/256.0);
                results->boxes[i].ymax = ymax / (256.0/img_height);  

                //results->boxes[i].xmin = xmin * (640.0/256.0); // move down to 480 space
                //results->boxes[i].ymin = ymin / (256.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/256.0);
                //results->boxes[i].ymax = ymax / (256.0/480.0);

                results->boxes[i].id = cls_id;
                results->boxes[i].score = score;
                new_num+=1;
            };
            results->num = new_num;
            auto for_loop_end = Clock::now();
            auto loop_time_elapsed = Duration(for_loop_end - for_loop_start).count();
            std::cout << "loop time elapsed: " << loop_time_elapsed << std::endl;

            auto free_start = Clock::now();
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
            auto end = Clock::now();
            auto free_time_elapsed = Duration(end - free_start).count();
            std::cout << "free vars time elapsed: " << free_time_elapsed << std::endl;
            auto total_time_elapsed = Duration(end - start).count();
            std::cout << "total time elapsed: " <<total_time_elapsed << std::endl;
            return results;
    }
};


class YoloTVMGPU320{
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
        // we have have some kind of way to make this configureable
        int width = 320;
        int height = 320;
        static constexpr int deploy_dtype_code = kDLFloat;
        static constexpr int deploy_dtype_bits = 32;
        static constexpr int deploy_dtype_lanes = 1;
        static constexpr int deploy_in_ndim = 4;
        static constexpr int deploy_out_ndim = 3;
        static constexpr int max_yolo_boxes = 100;
        static constexpr int64_t deploy_in_shape[deploy_in_ndim] = {1, 3, 320, 320};
        static constexpr int64_t deploy_tvm_id_and_score_size[deploy_out_ndim] = {1, 100, 1};
        static constexpr int64_t deploy_tvm_box_size[deploy_out_ndim] = {1, 100, 4};    

        YoloTVMGPU320(std::string model_folder) {
            // tvm module for compiled functions
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_folder + "/mnet.1.aarch64.cuda.hands.so");
            // json graph
            std::ifstream json_in(model_folder + "/mnet.1.aarch64.cuda.hands.json", std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();

            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int device_type = kDLGPU;//kDLCPU;//kDLGPU
            int device_id = 0;
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                  device_type, device_id);
            this->handle.reset(new tvm::runtime::Module(mod));
            // parameters in binary
            std::ifstream params_in(model_folder + "/mnet.1.aarch64.cuda.hands.params", std::ios::binary);
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
            cv::Size new_size = cv::Size(320, 320);
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
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            using Clock = std::chrono::high_resolution_clock;
            using Timepoint = Clock::time_point;
            using Duration = std::chrono::duration<double>;
            auto start = Clock::now();

            //Set constants and variables
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLGPU;
            constexpr int device_id = 0;
            int in_ndim = 4;
            int out_ndim = 3;
            int in_c = 3, in_h = 320, in_w = 320;
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

            auto allocstart = Clock::now();
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_ids);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_scores);
            TVMArrayAlloc(tvm_box_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_bboxes);
            auto allocend = Clock::now();
            auto tvmalloc_elapsed = Duration(allocend - allocstart).count();
            std::cout << "TVMArrayAlloc time elapsed: " << tvmalloc_elapsed << std::endl;

            auto copyfrom_start = Clock::now();
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
            auto copyfrom_end = Clock::now();
            auto tvmcopy_elapsed = Duration(copyfrom_end - copyfrom_start).count();
            std::cout << "TVMArrayCopyFromBytes + processing image time elapsed: " << tvmcopy_elapsed << std::endl;

            auto tvmruntime_start = Clock::now();
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

            auto tvmruntime_end = Clock::now();
            auto tvmruntime_elapsed = Duration(tvmruntime_end - tvmruntime_start).count();
            std::cout << "tvm_runtime_functs: mod, set_input, run, getouput time elapsed: " << tvmruntime_elapsed << std::endl;


            auto toc2 = Clock::now();
            auto elapsed2 = Duration(toc2 - start).count();
            std::cout << "tvm setup/model-runtime/getoutput time elapsed: " << elapsed2 << std::endl;

            auto torchboxsstart = Clock::now();
            torch::Tensor ndarray_ids = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_scores = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_bboxes = torch::zeros({1, 100, 4}, at::kFloat);

            auto tvmarraycopystart = Clock::now();
            TVMArrayCopyToBytes(output_tensor_ids, ndarray_ids.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_scores, ndarray_scores.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_bboxes, ndarray_bboxes.data_ptr(),  1 * 100 * 4 * sizeof(float));
            auto tvmarraycopyend = Clock::now();
            auto tvmarraycopyend_elapsed = Duration(tvmarraycopyend - tvmarraycopystart).count();
            std::cout << "TVMArrayCopyToBytes time elapsed: " << tvmarraycopyend_elapsed << std::endl;


            auto torchboxsend = Clock::now();
            auto torchboxsend_elapsed = Duration(torchboxsend - torchboxsstart).count();
            std::cout << "torch inital tensor creation and copytobytes time elapsed: " << torchboxsend_elapsed << std::endl;

            auto accessor_start = Clock::now();
            auto ndarray_scores_a = ndarray_scores.accessor<float,3>();
            auto ndarray_ids_a = ndarray_ids.accessor<float,3>();
            auto ndarray_bboxes_a = ndarray_bboxes.accessor<float,3>();
            auto accessor_end = Clock::now();
            auto accessor_time_elapsed = Duration(accessor_end - accessor_start).count();
            std::cout << "accessor_time elapsed: " <<accessor_time_elapsed << std::endl;



            auto for_loop_start = Clock::now();
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


                results->boxes[i].xmin = xmin * (img_width/320.0); // move down to 480 space  ()
                results->boxes[i].ymin = ymin / (320.0/img_height); // move up to 640
                results->boxes[i].xmax = xmax * (img_width/320.0);
                results->boxes[i].ymax = ymax / (320.0/img_height);  

                //results->boxes[i].xmin = xmin * (640.0/320.0); // move down to 480 space
                //results->boxes[i].ymin = ymin / (320.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/320.0);
                //results->boxes[i].ymax = ymax / (320.0/480.0);
                results->boxes[i].id = cls_id;
                results->boxes[i].score = score;
                new_num+=1;
            };
            results->num = new_num;
            auto for_loop_end = Clock::now();
            auto loop_time_elapsed = Duration(for_loop_end - for_loop_start).count();
            std::cout << "loop time elapsed: " << loop_time_elapsed << std::endl;

            auto free_start = Clock::now();
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
            auto end = Clock::now();
            auto free_time_elapsed = Duration(end - free_start).count();
            std::cout << "free vars time elapsed: " << free_time_elapsed << std::endl;
            auto total_time_elapsed = Duration(end - start).count();
            std::cout << "total time elapsed: " <<total_time_elapsed << std::endl;
            return results;
    }
};



class YoloTVMGPU{
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
        // we have have some kind of way to make this configureable
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

        YoloTVMGPU(std::string model_folder) {
            // tvm module for compiled functions
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_folder + "/mnet1.0_yolo_aarch64_cuda_lib.so");
            // json graph
            std::ifstream json_in(model_folder + "/mnet1.0_yolo_aarch64_cuda_graph.json", std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();

            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int device_type = kDLGPU;//kDLCPU;//kDLGPU
            int device_id = 0;
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                  device_type, device_id);
            this->handle.reset(new tvm::runtime::Module(mod));
            // parameters in binary
            std::ifstream params_in(model_folder + "/mnet1.0_yolo_aarch64_cuda_param.params", std::ios::binary);
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
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            //Set constants and variables
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLGPU;
            constexpr int device_id = 0;
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

                results->boxes[i].xmin = xmin * (img_width/512.0); // move down to 480 space  ()
                results->boxes[i].ymin = ymin / (512.0/img_height); // move up to 640
                results->boxes[i].xmax = xmax * (img_width/512.0);
                results->boxes[i].ymax = ymax / (512.0/img_height);                

                //results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space  ()
                //results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/512.0);
                //results->boxes[i].ymax = ymax / (512.0/480.0);
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


class YoloTVMCPU{
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

        YoloTVMCPU(std::string model_folder) {
            // tvm module for compiled functions
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_folder + "/deploy_lib_cpu.so");
            // json graph
            std::ifstream json_in(model_folder + "/deploy_graph_cpu.json", std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();

            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int device_type = kDLCPU;
            int device_id = 0;
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                  device_type, device_id);
            this->handle.reset(new tvm::runtime::Module(mod));
            // parameters in binary
            std::ifstream params_in(model_folder + "/deploy_param_cpu.params", std::ios::binary);
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
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            //Set constants and variables
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLCPU;
            constexpr int device_id = 0;
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


                results->boxes[i].xmin = xmin * (img_width/512.0); // move down to 480 space  ()
                results->boxes[i].ymin = ymin / (512.0/img_height); // move up to 640
                results->boxes[i].xmax = xmax * (img_width/512.0);
                results->boxes[i].ymax = ymax / (512.0/img_height);  

                // int in_c = 3, in_h = 480, in_w = 640;
                //results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space
                //results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/512.0);
                //results->boxes[i].ymax = ymax / (512.0/480.0);
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



class YoloTVMCPUNoTorch{
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

        YoloTVMCPUNoTorch(std::string model_folder) {
            // tvm module for compiled functions
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_folder + "/deploy_lib_cpu.so");
            // json graph
            std::ifstream json_in(model_folder + "/deploy_graph_cpu.json", std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();

            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int device_type = kDLCPU;
            int device_id = 0;
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                  device_type, device_id);
            this->handle.reset(new tvm::runtime::Module(mod));
            // parameters in binary
            std::ifstream params_in(model_folder + "/deploy_param_cpu.params", std::ios::binary);
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
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            //Set constants and variables
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLCPU;
            constexpr int device_id = 0;
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


            // figure out how to do this not in torch...?
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


                results->boxes[i].xmin = xmin * (img_width/512.0); // move down to 480 space  ()
                results->boxes[i].ymin = ymin / (512.0/img_height); // move up to 640
                results->boxes[i].xmax = xmax * (img_width/512.0);
                results->boxes[i].ymax = ymax / (512.0/img_height);  

                // int in_c = 3, in_h = 480, in_w = 640;
                //results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space
                //results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/512.0);
                //results->boxes[i].ymax = ymax / (512.0/480.0);
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


class YoloTVMFromConfig{
    private:
        //working: void * handle;
        std::unique_ptr<tvm::runtime::Module> handle;

    public:
        // confident in these
        // set these as the default
        std::string deploy_lib_path;
        std::string deploy_graph_path;
        std::string deploy_param_path;
        bool gpu = true;
        int device_id;// = 0;
        int dtype_code = kDLFloat;
        int dtype_bits = 32;
        int dtype_lanes = 1;
        int device_type = kDLGPU;
        // set default here???
        int width;// = 512;
        int height;// = 512;
        //int64_t in_shape;// = {1, 3, height, width};
        //int64_t tvm_id_and_score_size[3] = {1, 100, 1};
        //int64_t tvm_box_size[3];// = {1, 100, 4};
        int total_input;// = 3 * width * height;
        int in_ndim = 4;
        int out_ndim = 3;
        int max_yolo_boxes = 100;
        int64_t tvm_id_and_score_size[3] = {1, 100, 1};
        int64_t tvm_box_size[3] = {1, 100, 4};

        YoloTVMFromConfig(std::string config_path, std::string ros_package_string) {
            // read config with nlohmann-json
            std::cout << "start model_config reading" << std::endl;
            json model_config;
            std::string package_path = ros::package::getPath(ros_package_string);
            std::string full_path = package_path + config_path;
            std::ifstream json_read(full_path);
            json_read >> model_config;
            std::cout << "model_config read into memory" << std::endl;
            // read variables
            std::string lib_path = model_config["deploy_lib_path"];
            std::string graph_path = model_config["deploy_graph_path"];
            std::string param_path = model_config["deploy_param_path"];

            deploy_lib_path = package_path + lib_path;
            deploy_graph_path = package_path + graph_path;
            deploy_param_path = package_path + param_path;

            device_id = model_config["device_id"];
            width = model_config["width"];
            height = model_config["height"];
            gpu = model_config["gpu"];
            total_input = 3 * width * height;

            //int64_t in_shape[4] = {1, 3, height, width};
            //int64_t in_shape[4] = {1, 3, height, width};
            // set device type -- I think this has to be set here...
            if (gpu){
                device_type = kDLGPU;
            } else {
                device_type = kDLCPU;
            }

            // read deploy lib
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(deploy_lib_path);
            // read deplpy json
            std::ifstream json_in(deploy_graph_path, std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                  device_type, device_id);
            this->handle.reset(new tvm::runtime::Module(mod));
            // parameters in binary
            std::ifstream params_in(deploy_param_path, std::ios::binary);
            std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
            params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray params_arr;
            params_arr.data = params_data.c_str();
            params_arr.size = params_data.length();
            tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
            load_params(params_arr);
        }
       
        /**
         * \brief function to normalize an image before it's processed by the network
         * \param[in] the raw cv::mat image
         * \return the normalized version of the iamge.
         */  
        cv::Mat preprocess_image(cv::Mat frame){
            cv::Size new_size = cv::Size(width, height); // or is it height width????
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
            cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
            cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
            cv::Mat temp;
            temp = resized_image_floats - mean;
            normalized_image = temp / theta;
            return normalized_image;
        }

        yoloresults* forward_full(cv::Mat frame, float thresh)
        {
            std::cout << "starting function" << std::endl;
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            //Set constants and variables -- this is a prob,
            // how to do this at runtime?
            //constexpr int const_dtype_code = dtype_code;
            //constexpr int const_dtype_bits = dtype_bits;
            //constexpr int const_dtype_lanes = dtype_lanes;
            //constexpr int const_device_type = device_type;
            //constexpr int const_device_id = device_id;
            //int64_t in_shape[4] = {1, in_c, in_h, in_w};
            int64_t in_shape[4] = {1, 3, height, width};
            total_input = 3 * width * height;
            std::cout << "width: " << width << std::endl;
            std::cout << "height: " << height << std::endl;
            std::cout << "total_input: " << total_input << std::endl;
            std::cout << "device_id: " << device_id << std::endl;
            std::cout << "dtype_code: " << dtype_code << std::endl;
            std::cout << "dtype_bits: " << dtype_bits << std::endl;
            std::cout << "dtype_lanes: " << dtype_lanes << std::endl;
            std::cout << "device_type: " << device_type << std::endl;

            DLTensor *output_tensor_ids;
            DLTensor *output_tensor_scores;
            DLTensor *output_tensor_bboxes;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            // allocate memory for results
            yoloresults* results = (yoloresults*)calloc(1, sizeof(yoloresults));
            results->num = 100;
            results->boxes = (bbox_result*)calloc(100, sizeof(bbox_result));

            std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_ids);
            TVMArrayAlloc(tvm_id_and_score_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_scores);
            TVMArrayAlloc(tvm_box_size, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_bboxes);
            std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(frame);
            std::cout << "preprocess finished" << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            std::cout << "run/getoutput/setinput finished" << std::endl;
  
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
            std::cout << "TVMSynchronize finished" << std::endl;  

            // dynamically set?
            torch::Tensor ndarray_ids = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_scores = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_bboxes = torch::zeros({1, 100, 4}, at::kFloat);

            TVMArrayCopyToBytes(output_tensor_ids, ndarray_ids.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_scores, ndarray_scores.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_bboxes, ndarray_bboxes.data_ptr(),  1 * 100 * 4 * sizeof(float));

            auto ndarray_scores_a = ndarray_scores.accessor<float,3>();
            auto ndarray_ids_a = ndarray_ids.accessor<float,3>();
            auto ndarray_bboxes_a = ndarray_bboxes.accessor<float,3>();
            std::cout << "torch part finished" << std::endl; 

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

                results->boxes[i].xmin = xmin * (img_width/height); // move down to 480 space  ()
                results->boxes[i].ymin = ymin / (width/img_height); // move up to 640
                results->boxes[i].xmax = xmax * (img_width/height);
                results->boxes[i].ymax = ymax / (width/img_height);                

                //results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space  ()
                //results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/512.0);
                //results->boxes[i].ymax = ymax / (512.0/480.0);
                results->boxes[i].id = cls_id;
                results->boxes[i].score = score;
                new_num+=1;
            };
            results->num = new_num;
            std::cout << "torch array iter finished" << std::endl;            

            // free outputs
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
            std::cout << "freeing finished" << std::endl;
            return results;
        }  
};


class PoseFromConfig{
    private:
        //working: void * handle;
        std::unique_ptr<tvm::runtime::Module> detector_handle;
        std::unique_ptr<tvm::runtime::Module> pose_handle;

    public:
        // confident in these
        // set these as the default
        std::string deploy_lib_path;
        std::string deploy_graph_path;
        std::string deploy_param_path;
        bool gpu = true;
        int device_id;// = 0;
        int dtype_code = kDLFloat;
        int dtype_bits = 32;
        int dtype_lanes = 1;
        int device_type = kDLGPU;
        // set default here???
        int detector_width;// = 512;
        int detector_height;// = 512;
        int pose_width;// = 256;
        int pose_height;// = 192;
        int detector_total_input;// = 3 * width * height;
        int pose_total_input;// = 3 * width * height;
        int in_ndim = 4;
        int pose_out_ndim = 4;
        int detector_out_ndim = 3;
        int max_yolo_boxes = 100;
        // maybe we can dynamically set all of these
        int64_t tvm_id_and_score_size[3] = {1, 100, 1};
        int64_t tvm_box_size[3] = {1, 100, 4};
        int64_t tvm_heatmap_size[4] = {1, 17, 64, 48};

        /**
         * function that reads both the yolo detector and the pose detector
         * 
        */
        PoseFromConfig(std::string config_path, std::string ros_package_string) {
            // read config with nlohmann-json
            std::cout << "start model_config reading" << std::endl;
            json model_config;
            std::string package_path = "/opt/catkin_ws/src/open_ptrack/recognition";
            std::string full_path = package_path + config_path;
            std::ifstream json_read(full_path);
            json_read >> model_config;
            std::cout << "model_config read into memory" << std::endl;
            // read detector variables
            std::string detector_lib_path = model_config["detector_deploy_lib_path"];
            std::string detector_graph_path = model_config["detector_deploy_graph_path"];
            std::string detector_param_path = model_config["detector_deploy_param_path"];
            // read pose variables
            std::string pose_lib_path = model_config["pose_deploy_lib_path"];
            std::string pose_graph_path = model_config["pose_deploy_graph_path"];
            std::string pose_param_path = model_config["pose_deploy_param_path"];

            device_id = model_config["device_id"];
            detector_width = model_config["detector_width"]; //(512,512)
            detector_height = model_config["detector_height"]; //(512,512)
            pose_width = model_config["pose_width"]; //(256, 192)
            pose_height = model_config["pose_height"]; //(256, 192)
            gpu = model_config["gpu"];
            detector_total_input = 1 * 3 * detector_width * detector_height;
            pose_total_input = 1 * 3 * pose_width * pose_height;

            std::string detector_deploy_lib_path = package_path + detector_lib_path;
            std::string detector_deploy_graph_path = package_path + detector_graph_path;
            std::string detector_deploy_param_path = package_path + detector_param_path;

            std::string pose_deploy_lib_path = package_path + pose_lib_path;
            std::string pose_deploy_graph_path = package_path + pose_graph_path;
            std::string pose_deploy_param_path = package_path + pose_param_path;

            // set device type -- I think this has to be set here...
            if (gpu){
                device_type = kDLGPU;
            } else {
                device_type = kDLCPU;
            }
            // DETECTOR READ
            // read deploy lib
            tvm::runtime::Module detector_mod_syslib = tvm::runtime::Module::LoadFromFile(detector_deploy_lib_path);
            // read deplpy json
            std::ifstream detector_json_in(detector_deploy_graph_path, std::ios::in);
            std::string detector_json_data((std::istreambuf_iterator<char>(detector_json_in)), std::istreambuf_iterator<char>());
            detector_json_in.close();
            // get global function module for graph runtime
            tvm::runtime::Module detector_mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(detector_json_data, detector_mod_syslib,
                                                                                                  device_type, device_id);
            this->detector_handle.reset(new tvm::runtime::Module(detector_mod));
            // parameters in binary
            std::ifstream detector_params_in(detector_deploy_param_path, std::ios::binary);
            std::string detector_params_data((std::istreambuf_iterator<char>(detector_params_in)), std::istreambuf_iterator<char>());
            detector_params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray detector_params_arr;
            detector_params_arr.data = detector_params_data.c_str();
            detector_params_arr.size = detector_params_data.length();
            tvm::runtime::PackedFunc detector_load_params = detector_mod.GetFunction("load_params");
            detector_load_params(detector_params_arr);

            //POSE READ
            // read deploy lib
            tvm::runtime::Module pose_mod_syslib = tvm::runtime::Module::LoadFromFile(pose_deploy_lib_path);
            // read deplpy json
            std::ifstream pose_json_in(pose_deploy_graph_path, std::ios::in);
            std::string pose_json_data((std::istreambuf_iterator<char>(pose_json_in)), std::istreambuf_iterator<char>());
            pose_json_in.close();
            // get global function module for graph runtime
            tvm::runtime::Module pose_mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(pose_json_data, pose_mod_syslib,
                                                                                                  device_type, device_id);
            this->pose_handle.reset(new tvm::runtime::Module(pose_mod));
            // parameters in binary
            std::ifstream pose_params_in(pose_deploy_param_path, std::ios::binary);
            std::string pose_params_data((std::istreambuf_iterator<char>(pose_params_in)), std::istreambuf_iterator<char>());
            pose_params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray pose_params_arr;
            pose_params_arr.data = pose_params_data.c_str();
            pose_params_arr.size = pose_params_data.length();
            tvm::runtime::PackedFunc pose_load_params = pose_mod.GetFunction("load_params");
            pose_load_params(pose_params_arr);
        }
       
        /**
         * \brief function to normalize an image before it's processed by the network
         * \param[in] the raw cv::mat image
         * \return the normalized version of the iamge.
         */  
        cv::Mat preprocess_image(cv::Mat frame, int width, int height, bool convert){
            cv::Size new_size = cv::Size(width, height); // or is it height width????
            cv::Mat resized_image;
            if (convert){
              cv::Mat rgb;
              // bgr to rgb
              cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
              cv::resize(rgb, resized_image, new_size);
            } else {
              cv::resize(frame, resized_image, new_size);
            }
            // resize to 512x512
            cv::Mat resized_image_floats(new_size, CV_32FC3);
            // convert resized image to floats and normalize
            resized_image.convertTo(resized_image_floats, CV_32FC3, 1.0f/255.0f);
            //mimic mxnets 'to_tensor' function
            cv::Mat normalized_image(new_size, CV_32FC3);
            // mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
            cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
            cv::Mat temp;
            temp = resized_image_floats - mean;
            normalized_image = temp / theta;
            return normalized_image;
        }

        pose_results* forward_full(cv::Mat frame, float thresh)
        {
            std::cout << "starting function" << std::endl;
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            //Set constants and variables -- this is a prob,
            // how to do this at runtime?
            //constexpr int const_dtype_code = dtype_code;
            //constexpr int const_dtype_bits = dtype_bits;
            //constexpr int const_dtype_lanes = dtype_lanes;
            //constexpr int const_device_type = device_type;
            //constexpr int const_device_id = device_id;
            //int64_t in_shape[4] = {1, in_c, in_h, in_w};
            int64_t in_shape[4] = {1, 3, detector_height, detector_width};
            int total_input = 3 * detector_width * detector_height;
            std::cout << "width: " << detector_width << std::endl;
            std::cout << "height: " << detector_height << std::endl;
            std::cout << "total_input: " << total_input << std::endl;
            std::cout << "device_id: " << device_id << std::endl;
            std::cout << "dtype_code: " << dtype_code << std::endl;
            std::cout << "dtype_bits: " << dtype_bits << std::endl;
            std::cout << "dtype_lanes: " << dtype_lanes << std::endl;
            std::cout << "device_type: " << device_type << std::endl;

            DLTensor *output_tensor_ids;
            DLTensor *output_tensor_scores;
            DLTensor *output_tensor_bboxes;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            // allocate memory for results
            pose_results* results = (pose_results*)calloc(1, sizeof(pose_results));
            results->num = 100;
            results->boxes = (pose_result*)calloc(100, sizeof(pose_result));

            std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_id_and_score_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_ids);
            TVMArrayAlloc(tvm_id_and_score_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_scores);
            TVMArrayAlloc(tvm_box_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_bboxes);
            std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(frame, detector_width, detector_height, true);
            std::cout << "preprocess finished" << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) detector_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            std::cout << "run/getoutput/setinput finished" << std::endl;
  
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
            std::cout << "TVMSynchronize finished" << std::endl;  

            // dynamically set?
            torch::Tensor ndarray_ids = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_scores = torch::zeros({1, 100, 1}, at::kFloat);
            torch::Tensor ndarray_bboxes = torch::zeros({1, 100, 4}, at::kFloat);

            TVMArrayCopyToBytes(output_tensor_ids, ndarray_ids.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_scores, ndarray_scores.data_ptr(),  1 * 100 * 1 * sizeof(float));
            TVMArrayCopyToBytes(output_tensor_bboxes, ndarray_bboxes.data_ptr(),  1 * 100 * 4 * sizeof(float));

            auto ndarray_scores_a = ndarray_scores.accessor<float,3>();
            auto ndarray_ids_a = ndarray_ids.accessor<float,3>();
            auto ndarray_bboxes_a = ndarray_bboxes.accessor<float,3>();
            std::cout << "torch part finished" << std::endl; 

            // we can probably free outputs right here...
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

            // ******* NOTE **********
            // Instead of running:
            // yoloresult = yolo(image);
            // poseresult = pose(image);
            // aligned_poses_boxes = hungarian_munkres(yoloresult, poseresult);
            // we're just going to run the pose detector
            // in the forloop of the item
            // ***************************
            float fheight = static_cast<float>(img_height);
            float fwidth = static_cast<float>(img_width);
            int new_num = 0;
            for (int i = 0; i < max_yolo_boxes; ++i) {
                float xmin;
                float ymin;
                float xmax;
                float ymax;

                float score = ndarray_scores_a[0][i][0];
                float label = ndarray_ids_a[0][i][0];
                if (score < thresh) continue;
                if (label < 0) continue;
                // people only
                if (label > 0) continue;

                int cls_id = static_cast<int>(label);
                xmin = ndarray_bboxes_a[0][i][0];
                ymin = ndarray_bboxes_a[0][i][1];
                xmax = ndarray_bboxes_a[0][i][2];
                ymax = ndarray_bboxes_a[0][i][3];
                //SCALE to frame height
                xmin = xmin * (img_width/detector_height); // move down to 480 space  ()
                ymin = ymin / (detector_width/img_height); // move up to 640
                xmax = xmax * (img_width/detector_height);
                ymax = ymax / (detector_width/img_height);
                
                std::cout << "xmin: " << xmin << std::endl;
                std::cout << "ymin: " << ymin << std::endl;
                std::cout << "xmax: " << xmax << std::endl;
                std::cout << "ymax: " << ymax << std::endl;
                // upscale bbox function from simple pose
                // pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
                // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L218
                //def upscale_bbox_fn(bbox, img, scale=1.25):
                //    new_bbox = []
                //    x0 = bbox[0]
                //    y0 = bbox[1]
                //    x1 = bbox[2]
                //    y1 = bbox[3]
                //    w = (x1 - x0) / 2
                //    h = (y1 - y0) / 2
                //    center = [x0 + w, y0 + h]
                //    new_x0 = max(center[0] - w * scale, 0)
                //    new_y0 = max(center[1] - h * scale, 0)
                //    new_x1 = min(center[0] + w * scale, img.shape[1])
                //    new_y1 = min(center[1] + h * scale, img.shape[0])
                //    new_bbox = [new_x0, new_y0, new_x1, new_y1]
                //    return new_bbox

                float scale = 1.26;
                float w = (xmax - xmin) / 2.0f;
                float h = (ymax - ymin) / 2.0f;
                float center_x = xmin + w; 
                float center_y = ymin + h;
                float upminx = center_x - w * scale;
                float upminy = center_y - h * scale;
                float upmaxx = center_x + w * scale;
                float upmaxy = center_y + h * scale;

                std::cout << "yolo_forward w: " << w << std::endl;
                std::cout << "yolo_forward h: " << h << std::endl;
                std::cout << "yolo_forward center_x " << center_x << std::endl;
                std::cout << "yolo_forward center_y " << center_y << std::endl;

                float upscaled_xmin = std::max(upminx, 0.0f);
                float upscaled_ymin = std::max(upminy, 0.0f);
                float upscaled_xmax = std::min(upmaxx, fwidth);
                float upscaled_ymax = std::min(upmaxy, fheight);
                std::cout << "upscaled_xmin: " << upscaled_xmin << std::endl;
                std::cout << "upscaled_ymin: " << upscaled_ymin << std::endl;
                std::cout << "upscaled_xmax: " << upscaled_xmax << std::endl;
                std::cout << "upscaled_ymax: " << upscaled_ymax << std::endl;

                //float upscaled_xmin = std::max(center_x - w * scale, 0.0f);
                //float upscaled_ymin = std::max(center_y - h * scale, 0.0f);
                //float upscaled_xmax = std::min(center_x + w * scale, static_cast<float>(img_height));
                //float upscaled_ymax = std::min(center_y + h * scale, static_cast<float>(img_width));
                // convert to int for roi-transform
                int int_upscaled_xmin = static_cast<int>(upscaled_xmin);
                int int_upscaled_ymin = static_cast<int>(upscaled_ymin);
                int int_upscaled_xmax = static_cast<int>(upscaled_xmax);
                int int_upscaled_ymax = static_cast<int>(upscaled_ymax);
                std::cout << "int_upscaled_xmin: " << int_upscaled_xmin << std::endl;
                std::cout << "int_upscaled_ymin: " << int_upscaled_ymin << std::endl;
                std::cout << "int_upscaled_xmax: " << int_upscaled_xmax << std::endl;
                std::cout << "int_upscaled_ymax: " << int_upscaled_ymax << std::endl;
                
                //0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows
                if (0 >= int_upscaled_xmin){
                  int_upscaled_xmin = 1;
                  upscaled_xmin = 1.0;
                }
                if (int_upscaled_xmax > img_width){
                  int_upscaled_xmax = img_width;
                  upscaled_xmax = fwidth;
                }
                if (0 >= int_upscaled_ymin){
                  int_upscaled_ymin = 0;
                  upscaled_ymin = 0.0;
                }
                if (int_upscaled_ymax > img_height){
                  int_upscaled_ymax = img_height;
                  upscaled_ymax = fheight;
                }
                std::cout << "post_upscaled_xmin: " << upscaled_xmin << std::endl;
                std::cout << "post_upscaled_ymin: " << upscaled_ymin << std::endl;
                std::cout << "post_upscaled_xmax: " << upscaled_xmax << std::endl;
                std::cout << "post_upscaled_ymax: " << upscaled_ymax << std::endl;
                std::cout << "post_int_upscaled_xmin: " << int_upscaled_xmin << std::endl;
                std::cout << "post_int_upscaled_ymin: " << int_upscaled_ymin << std::endl;
                std::cout << "post_int_upscaled_xmax: " << int_upscaled_xmax << std::endl;
                std::cout << "post_int_upscaled_ymax: " << int_upscaled_ymax << std::endl;
                
                // get upscaled bounding box and extract image-patch/mask
                cv::Rect roi(int_upscaled_xmin, int_upscaled_ymin, int_upscaled_xmax-int_upscaled_xmin, int_upscaled_ymax-int_upscaled_ymin);
                std::cout << "created rect created" << std::endl;
                cv::Mat image_roi = frame(roi);
                cv::Size image_roi_image_size = image_roi.size();
                std::cout << "image_roi_image_size created: " << image_roi_image_size.height << std::endl;
                std::cout << "image_roi_image_size created: " << image_roi_image_size.width << std::endl;
                //debug only cv::imwrite("/home/nvidia/pose_image_roi.jpg", image_roi);
                //preprocessing happens inside forward function
                // why point3f and not 2f? 
                // we're using z as the confidence
                std::vector<cv::Point3f> pose_coords = pose_forward(image_roi, upscaled_xmin, upscaled_ymin, upscaled_xmax, upscaled_ymax);
                results->boxes[i].xmin = xmin;
                results->boxes[i].ymin = ymin;
                results->boxes[i].xmax = xmax;
                results->boxes[i].ymax = ymax;                 
                results->boxes[i].points = pose_coords;
                //results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space  ()
                //results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/512.0);
                //results->boxes[i].ymax = ymax / (512.0/480.0);
                results->boxes[i].id = cls_id;
                results->boxes[i].score = score;
                new_num+=1;
            };
            results->num = new_num;
            std::cout << "torch array iter finished" << std::endl;            

            // free outputs
            //TVMArrayFree(input);
            //TVMArrayFree(output_tensor_ids);
            //TVMArrayFree(output_tensor_scores);
            //TVMArrayFree(output_tensor_bboxes);
            //input = nullptr;
            //output_tensor_ids = nullptr;
            //output_tensor_scores = nullptr;
            //output_tensor_bboxes = nullptr;
            //free(data_x);
            //data_x = nullptr;
            //std::cout << "freeing finished" << std::endl;
            return results;
        }

        std::vector<cv::Point3f> pose_forward(cv::Mat bbox_mask, float xmin, float ymin, float xmax, float ymax)
        {
            std::cout << "running pose forward" << std::endl;
            // get height/width dynamically
            cv::Size image_size = bbox_mask.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);
            int64_t in_shape[4] = {1, 3, pose_height, pose_width};
            int total_input = 3 * pose_width * pose_height;
            
            DLTensor *output_tensor_heatmap;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_heatmap_size, pose_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_heatmap);
            std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(bbox_mask, pose_width, pose_height, true);
            cv::Size processed_image_size = processed_image.size();
            std::cout << "preprocess finished: " << std::endl;
            std::cout << "preprocess height: " << processed_image_size.height << std::endl;
            std::cout << "preprocess width: " << processed_image_size.width << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) pose_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            std::cout << "run/getoutput/setinput finished" << std::endl;
  
            // https://github.com/apache/incubator-tvm/issues/979?from=timeline
            //"This may give you some ideas to start with.
            //In general you want to use pinned memory and you want
            //to interleave computation with copying; so you want to
            // be upload the next thing while you are computing the
            //current thing while you are downloading the last thing."
            TVMSynchronize(device_type, device_id, nullptr);
            get_output(0, output_tensor_heatmap);
            std::cout << "TVMSynchronize finished" << std::endl;  

            torch::Tensor ndarray_heat_map_full = torch::zeros({1, 17, 64, 48}, at::kFloat);

            TVMArrayCopyToBytes(output_tensor_heatmap, ndarray_heat_map_full.data_ptr(), 1*17*64*48 * sizeof(float));
            std::cout << "saving array output " << std::endl;
            auto bytes = torch::pickle_save(ndarray_heat_map_full);
            std::ofstream fout("/home/nvidia/pose.zip", std::ios::out | std::ios::binary);
            fout.write(bytes.data(), bytes.size());
            fout.close();

            //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L172
            //heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
            //  
            // pytorch view vs. reshape; use of auto?
            auto ndarray_heat_map = ndarray_heat_map_full.view({17, 3072});
            //std::vector<int64_t> heatsize = ndarray_heat_map.sizes();
            std::cout << "ndarray_heat_map reshape finished: " << ndarray_heat_map.sizes().size() << std::endl;
            
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L173
            // idx = nd.argmax(heatmaps_reshaped, 2)
            torch::Tensor idx = torch::argmax(ndarray_heat_map, 1);
            //std::vector<int64_t> idxsize = idx.sizes().size();
            std::cout << "argmax finished: " << idx.sizes().size() << std::endl;
            
            // creat empty pred container
            torch::Tensor preds = torch::zeros({17, 2}, at::kFloat);
            // create accessors


            auto idx_accessor = idx.accessor<long,1>(); // 1, 17 -> batch_size, 17
            auto heat_map_accessor = ndarray_heat_map.accessor<float,2>(); // 1, 17, 1
            
            // vars to preset
            // pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L273
            // use z as container for probability
            float heatmap_width = 48.0f;
            float heatmap_height = 64.0f;
            std::vector<cv::Point3f> points;
            float w = (xmax - xmin) / 2.0f;
            float h = (ymax - ymin) / 2.0f;
            float center_x = xmin + w; 
            float center_y = ymin + h;
            std::cout << "pose_forward w: " << w << std::endl;
            std::cout << "pose_forward h: " << h << std::endl;
            std::cout << "pose_forward center_x: " << center_x << std::endl;
            std::cout << "pose_forward center_y: " << center_y << std::endl;
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L168
            // might have to use a diff var name
            for (size_t i = 0; i < 17; i++){
              float index = idx_accessor[i];
              std::cout << "index: " << index << std::endl;
              float probability = heat_map_accessor[i][static_cast<int>(index)];
              std::cout << "probability: " << probability << std::endl;
              
              //// python modulo vs c++ is dfff
              ////https://stackoverflow.com/questions/1907565/c-and-python-different-behaviour-of-the-modulo-operation
              //preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)
              //preds[:, :, 0] = (preds[:, :, 0]) % width
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L181
              // float modulo_pred = ((index % heatmap_width) + heatmap_width) % heatmap_width;
              // float floor_pred = std::floor(index / heatmap_width);
              int modulo_int = static_cast<int>(index) % static_cast<int>(heatmap_width);
              std::cout << "modulo_int: " << modulo_int << std::endl;
              float modulo_pred = static_cast<float>(modulo_int);
              std::cout << "modulo_pred: " << modulo_pred << std::endl;
              float floor = index / heatmap_width;
              std::cout << "floor: " << floor << std::endl;
              float floor_pred = std::floor(floor);
              std::cout << "floor_pred: " << floor_pred << std::endl;
              if (probability <= 0.0) {
                // zero out the pred if the prob is bad...
                //pred_mask = nd.tile(nd.greater(maxvals, 0.0), (1, 1, 2))
                //pred_mask = pred_mask.astype(np.float32)
                //preds *= pred_mask
                modulo_pred = 0.0f;
                floor_pred = 0.0f;
              }
              std::cout << "modulo_pred_end: " << modulo_pred << std::endl;
              std::cout << "floor_pred_end: " << floor_pred << std::endl;
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L289-L290
              float w_ratio = modulo_pred / heatmap_width;
              float h_ratio = floor_pred / heatmap_height;
              std::cout << "w_ratio: " << w_ratio << std::endl;
              std::cout << "h_ratio: " << h_ratio << std::endl;              
              cv::Point3f point;
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L291-L292
              //scale = np.array([w, h])
              //preds[i][:, 0] = scale[0] * 2 * w_ratio + center[0] - scale[0]
              //center = np.array([x0 + w, y0 + h])
              point.x = w * 2.0f * w_ratio + center_x - w;
              point.y = h * 2.0f * h_ratio + center_y - h;
              point.z = probability;
              std::cout << "point.x: " << point.x << std::endl;
              std::cout << "point.y: " << point.y << std::endl;
              points.push_back(point);
            }
            // free outputs
            TVMArrayFree(input);
            TVMArrayFree(output_tensor_heatmap);
            input = nullptr;
            output_tensor_heatmap = nullptr;
            free(data_x);
            data_x = nullptr;
            std::cout << "freeing finished" << std::endl;
            return points;
        }
};

class NoNMSPoseFromConfig{
    private:
        //working: void * handle;
        std::unique_ptr<tvm::runtime::Module> detector_handle;
        std::unique_ptr<tvm::runtime::Module> pose_handle;

    public:
        // confident in these
        // set these as the default
        std::string deploy_lib_path;
        std::string deploy_graph_path;
        std::string deploy_param_path;
        bool gpu = true;
        int device_id;// = 0;
        int dtype_code = kDLFloat;
        int dtype_bits = 32;
        int dtype_lanes = 1;
        int device_type = kDLGPU;
        // set default here???
        int detector_width;// = 512;
        int detector_height;// = 512;
        int pose_width;// = 256;
        int pose_height;// = 192;
        int detector_total_input;// = 3 * width * height;
        int pose_total_input;// = 3 * width * height;
        int in_ndim = 4;
        int pose_out_ndim = 4;
        int detector_out_ndim = 3;
        int max_yolo_boxes = 100;
        int n_dets = 322560;
        // maybe we can dynamically set all of these
        int64_t tvm_id_and_score_size[3] = {1, 100, 1};
        int64_t tvm_box_size[3] = {1, 100, 4};
        int64_t tvm_heatmap_size[4] = {1, 17, 64, 48};
        int64_t no_nms_output_size[3] = {1, 322560, 6};
        float thresh = 0.3f;

        /**
         * function that reads both the yolo detector and the pose detector
         * 
        */
        NoNMSPoseFromConfig(std::string config_path, std::string ros_package_string) {
            // read config with nlohmann-json
            std::cout << "start model_config reading" << std::endl;
            json model_config;
            std::string package_path = "/opt/catkin_ws/src/open_ptrack/recognition";
            std::string full_path = package_path + config_path;
            std::ifstream json_read(full_path);
            json_read >> model_config;
            std::cout << "model_config read into memory" << std::endl;
            // read detector variables
            std::string detector_lib_path = model_config["detector_deploy_lib_path"];
            std::string detector_graph_path = model_config["detector_deploy_graph_path"];
            std::string detector_param_path = model_config["detector_deploy_param_path"];
            // read pose variables
            std::string pose_lib_path = model_config["pose_deploy_lib_path"];
            std::string pose_graph_path = model_config["pose_deploy_graph_path"];
            std::string pose_param_path = model_config["pose_deploy_param_path"];

            device_id = model_config["device_id"];
            detector_width = model_config["detector_width"]; //(512,512)
            detector_height = model_config["detector_height"]; //(512,512)
            pose_width = model_config["pose_width"]; //(256, 192)
            pose_height = model_config["pose_height"]; //(256, 192)
            gpu = model_config["gpu"];
            n_dets = model_config["n_dets"];
            no_nms_output_size[1] = n_dets;
            detector_total_input = 1 * 3 * detector_width * detector_height;
            pose_total_input = 1 * 3 * pose_width * pose_height;
            thresh = model_config["threshold"];

            std::string detector_deploy_lib_path = package_path + detector_lib_path;
            std::string detector_deploy_graph_path = package_path + detector_graph_path;
            std::string detector_deploy_param_path = package_path + detector_param_path;

            std::string pose_deploy_lib_path = package_path + pose_lib_path;
            std::string pose_deploy_graph_path = package_path + pose_graph_path;
            std::string pose_deploy_param_path = package_path + pose_param_path;

            // set device type -- I think this has to be set here...
            if (gpu){
                device_type = kDLGPU;
            } else {
                device_type = kDLCPU;
            }
            // DETECTOR READ
            // read deploy lib
            tvm::runtime::Module detector_mod_syslib = tvm::runtime::Module::LoadFromFile(detector_deploy_lib_path);
            // read deplpy json
            std::ifstream detector_json_in(detector_deploy_graph_path, std::ios::in);
            std::string detector_json_data((std::istreambuf_iterator<char>(detector_json_in)), std::istreambuf_iterator<char>());
            detector_json_in.close();
            // get global function module for graph runtime
            tvm::runtime::Module detector_mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(detector_json_data, detector_mod_syslib,
                                                                                                  device_type, device_id);
            this->detector_handle.reset(new tvm::runtime::Module(detector_mod));
            // parameters in binary
            std::ifstream detector_params_in(detector_deploy_param_path, std::ios::binary);
            std::string detector_params_data((std::istreambuf_iterator<char>(detector_params_in)), std::istreambuf_iterator<char>());
            detector_params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray detector_params_arr;
            detector_params_arr.data = detector_params_data.c_str();
            detector_params_arr.size = detector_params_data.length();
            tvm::runtime::PackedFunc detector_load_params = detector_mod.GetFunction("load_params");
            detector_load_params(detector_params_arr);

            //POSE READ
            // read deploy lib
            tvm::runtime::Module pose_mod_syslib = tvm::runtime::Module::LoadFromFile(pose_deploy_lib_path);
            // read deplpy json
            std::ifstream pose_json_in(pose_deploy_graph_path, std::ios::in);
            std::string pose_json_data((std::istreambuf_iterator<char>(pose_json_in)), std::istreambuf_iterator<char>());
            pose_json_in.close();
            // get global function module for graph runtime
            tvm::runtime::Module pose_mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(pose_json_data, pose_mod_syslib,
                                                                                                  device_type, device_id);
            this->pose_handle.reset(new tvm::runtime::Module(pose_mod));
            // parameters in binary
            std::ifstream pose_params_in(pose_deploy_param_path, std::ios::binary);
            std::string pose_params_data((std::istreambuf_iterator<char>(pose_params_in)), std::istreambuf_iterator<char>());
            pose_params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray pose_params_arr;
            pose_params_arr.data = pose_params_data.c_str();
            pose_params_arr.size = pose_params_data.length();
            tvm::runtime::PackedFunc pose_load_params = pose_mod.GetFunction("load_params");
            pose_load_params(pose_params_arr);
        }
       
        /**
         * \brief function to normalize an image before it's processed by the network
         * \param[in] the raw cv::mat image
         * \return the normalized version of the iamge.
         */  
        cv::Mat preprocess_image(cv::Mat frame, int width, int height, bool convert){
            cv::Size new_size = cv::Size(width, height); // or is it height width????
            cv::Mat resized_image;
            if (convert){
              cv::Mat rgb;
              // bgr to rgb
              cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
              cv::resize(rgb, resized_image, new_size);
            } else {
              cv::resize(frame, resized_image, new_size);
            }
            // resize to 512x512
            cv::Mat resized_image_floats(new_size, CV_32FC3);
            // convert resized image to floats and normalize
            resized_image.convertTo(resized_image_floats, CV_32FC3, 1.0f/255.0f);
            //mimic mxnets 'to_tensor' function
            cv::Mat normalized_image(new_size, CV_32FC3);
            // mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
            cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
            cv::Mat temp;
            temp = resized_image_floats - mean;
            normalized_image = temp / theta;
            return normalized_image;
        }

        pose_results* forward_full(cv::Mat frame)
        {
            std::cout << "starting function" << std::endl;
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            //Set constants and variables -- this is a prob,
            // how to do this at runtime?
            //constexpr int const_dtype_code = dtype_code;
            //constexpr int const_dtype_bits = dtype_bits;
            //constexpr int const_dtype_lanes = dtype_lanes;
            //constexpr int const_device_type = device_type;
            //constexpr int const_device_id = device_id;
            //int64_t in_shape[4] = {1, in_c, in_h, in_w};
            int64_t in_shape[4] = {1, 3, detector_height, detector_width};
            int total_input = 3 * detector_width * detector_height;
            std::cout << "width: " << detector_width << std::endl;
            std::cout << "height: " << detector_height << std::endl;
            std::cout << "total_input: " << total_input << std::endl;
            std::cout << "device_id: " << device_id << std::endl;
            std::cout << "dtype_code: " << dtype_code << std::endl;
            std::cout << "dtype_bits: " << dtype_bits << std::endl;
            std::cout << "dtype_lanes: " << dtype_lanes << std::endl;
            std::cout << "device_type: " << device_type << std::endl;

            DLTensor *output_for_nms;
            //DLTensor *output_tensor_ids;
            //DLTensor *output_tensor_scores;
            //DLTensor *output_tensor_bboxes;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            // allocate memory for results
            pose_results* results = (pose_results*)calloc(1, sizeof(pose_results));
            results->num = 100;
            results->boxes = (pose_result*)calloc(100, sizeof(pose_result));

            std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            //TVMArrayAlloc(tvm_id_and_score_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_ids);
            //TVMArrayAlloc(tvm_id_and_score_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_scores);
            //TVMArrayAlloc(tvm_box_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_bboxes);
            TVMArrayAlloc(no_nms_output_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_for_nms);
            std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(frame, detector_width, detector_height, true);
            std::cout << "preprocess finished" << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) detector_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            std::cout << "run/getoutput/setinput finished" << std::endl;
  
            // https://github.com/apache/incubator-tvm/issues/979?from=timeline
            //"This may give you some ideas to start with.
            //In general you want to use pinned memory and you want
            //to interleave computation with copying; so you want to
            // be upload the next thing while you are computing the
            //current thing while you are downloading the last thing."
            TVMSynchronize(device_type, device_id, nullptr);
            get_output(0, output_for_nms);
            //get_output(0, output_tensor_ids);
            //get_output(1, output_tensor_scores);
            //get_output(2, output_tensor_bboxes);
            std::cout << "TVMSynchronize finished" << std::endl;  


            // copy to output
            //ulsMatF(int cols, int rows, int channels)
            //at(int channel, int row, int c
            MatF yolo_output(6, n_dets, 1); //ulsMatF yolo_output(1, n_dets, 6);
            TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
            std::cout << "TVMSynchronize finished" << std::endl;  
            
            std::cout << "starting nms" << std::endl;
            //auto tick = Clock::now();
            std::vector<sortable_result> tvm_results;
            std::vector<sortable_result> proposals;
            proposals.clear();
            tvm_nms_cpu(proposals, yolo_output, thresh, thresh, tvm_results);
            std::cout << "ending nms" << std::endl;

            // dynamically set?
            //torch::Tensor ndarray_ids = torch::zeros({1, 100, 1}, at::kFloat);
            //torch::Tensor ndarray_scores = torch::zeros({1, 100, 1}, at::kFloat);
            //torch::Tensor ndarray_bboxes = torch::zeros({1, 100, 4}, at::kFloat);

            //TVMArrayCopyToBytes(output_tensor_ids, ndarray_ids.data_ptr(),  1 * 100 * 1 * sizeof(float));
            //TVMArrayCopyToBytes(output_tensor_scores, ndarray_scores.data_ptr(),  1 * 100 * 1 * sizeof(float));
            //TVMArrayCopyToBytes(output_tensor_bboxes, ndarray_bboxes.data_ptr(),  1 * 100 * 4 * sizeof(float));

            //auto ndarray_scores_a = ndarray_scores.accessor<float,3>();
            //auto ndarray_ids_a = ndarray_ids.accessor<float,3>();
            //auto ndarray_bboxes_a = ndarray_bboxes.accessor<float,3>();
            //std::cout << "torch part finished" << std::endl; 

            // we can probably free outputs right here...
            TVMArrayFree(input);
            //TVMArrayFree(output_tensor_ids);
            //TVMArrayFree(output_tensor_scores);
            //TVMArrayFree(output_tensor_bboxes);
            TVMArrayFree(output_for_nms); // may have to move this to the bottom.
            input = nullptr;
            //output_tensor_ids = nullptr;
            //output_tensor_scores = nullptr;
            //output_tensor_bboxes = nullptr;
            //output_for_nms = nullptr;
            free(data_x);
            data_x = nullptr;

            // ******* NOTE **********
            // Instead of running:
            // yoloresult = yolo(image);
            // poseresult = pose(image);
            // aligned_poses_boxes = hungarian_munkres(yoloresult, poseresult);
            // we're just going to run the pose detector
            // in the forloop of the item
            // ***************************
            float fheight = static_cast<float>(img_height);
            float fwidth = static_cast<float>(img_width);
            int new_num = 0;
            for (int i = 0; i < tvm_results.size(); ++i) {

                float xmin;
                float ymin;
                float xmax;
                float ymax;

                float score = tvm_results[i].probs;
                float label = tvm_results[i].cls;
                if (score < thresh) continue;
                if (label < 0) continue;
                // people only
                if (label > 0) continue;

                int cls_id = static_cast<int>(label);
                xmin = tvm_results[i].xmin;
                ymin = tvm_results[i].ymin;
                xmax = tvm_results[i].xmax;
                ymax = tvm_results[i].ymax;
                //SCALE to frame height
                xmin = xmin * (img_width/detector_height); // move down to 480 space  ()
                ymin = ymin / (detector_width/img_height); // move up to 640
                xmax = xmax * (img_width/detector_height);
                ymax = ymax / (detector_width/img_height);
                
                //std::cout << "xmin: " << xmin << std::endl;
                //std::cout << "ymin: " << ymin << std::endl;
                //std::cout << "xmax: " << xmax << std::endl;
                //std::cout << "ymax: " << ymax << std::endl;
                // upscale bbox function from simple pose
                // pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
                // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L218
                //def upscale_bbox_fn(bbox, img, scale=1.25):
                //    new_bbox = []
                //    x0 = bbox[0]
                //    y0 = bbox[1]
                //    x1 = bbox[2]
                //    y1 = bbox[3]
                //    w = (x1 - x0) / 2
                //    h = (y1 - y0) / 2
                //    center = [x0 + w, y0 + h]
                //    new_x0 = max(center[0] - w * scale, 0)
                //    new_y0 = max(center[1] - h * scale, 0)
                //    new_x1 = min(center[0] + w * scale, img.shape[1])
                //    new_y1 = min(center[1] + h * scale, img.shape[0])
                //    new_bbox = [new_x0, new_y0, new_x1, new_y1]
                //    return new_bbox

                float scale = 1.26;
                float w = (xmax - xmin) / 2.0f;
                float h = (ymax - ymin) / 2.0f;
                float center_x = xmin + w; 
                float center_y = ymin + h;
                float upminx = center_x - w * scale;
                float upminy = center_y - h * scale;
                float upmaxx = center_x + w * scale;
                float upmaxy = center_y + h * scale;

                //std::cout << "yolo_forward w: " << w << std::endl;
                //std::cout << "yolo_forward h: " << h << std::endl;
                //std::cout << "yolo_forward center_x " << center_x << std::endl;
                //std::cout << "yolo_forward center_y " << center_y << std::endl;

                float upscaled_xmin = std::max(upminx, 0.0f);
                float upscaled_ymin = std::max(upminy, 0.0f);
                float upscaled_xmax = std::min(upmaxx, fwidth);
                float upscaled_ymax = std::min(upmaxy, fheight);
                //std::cout << "upscaled_xmin: " << upscaled_xmin << std::endl;
                //std::cout << "upscaled_ymin: " << upscaled_ymin << std::endl;
                //std::cout << "upscaled_xmax: " << upscaled_xmax << std::endl;
                //std::cout << "upscaled_ymax: " << upscaled_ymax << std::endl;

                //float upscaled_xmin = std::max(center_x - w * scale, 0.0f);
                //float upscaled_ymin = std::max(center_y - h * scale, 0.0f);
                //float upscaled_xmax = std::min(center_x + w * scale, static_cast<float>(img_height));
                //float upscaled_ymax = std::min(center_y + h * scale, static_cast<float>(img_width));
                // convert to int for roi-transform
                int int_upscaled_xmin = static_cast<int>(upscaled_xmin);
                int int_upscaled_ymin = static_cast<int>(upscaled_ymin);
                int int_upscaled_xmax = static_cast<int>(upscaled_xmax);
                int int_upscaled_ymax = static_cast<int>(upscaled_ymax);
                //std::cout << "int_upscaled_xmin: " << int_upscaled_xmin << std::endl;
                //std::cout << "int_upscaled_ymin: " << int_upscaled_ymin << std::endl;
                //std::cout << "int_upscaled_xmax: " << int_upscaled_xmax << std::endl;
                //std::cout << "int_upscaled_ymax: " << int_upscaled_ymax << std::endl;
                
                //0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows
                if (0 >= int_upscaled_xmin){
                  int_upscaled_xmin = 1;
                  upscaled_xmin = 1.0;
                }
                if (int_upscaled_xmax > img_width){
                  int_upscaled_xmax = img_width;
                  upscaled_xmax = fwidth;
                }
                if (0 >= int_upscaled_ymin){
                  int_upscaled_ymin = 0;
                  upscaled_ymin = 0.0;
                }
                if (int_upscaled_ymax > img_height){
                  int_upscaled_ymax = img_height;
                  upscaled_ymax = fheight;
                }
                //std::cout << "post_upscaled_xmin: " << upscaled_xmin << std::endl;
                //std::cout << "post_upscaled_ymin: " << upscaled_ymin << std::endl;
                //std::cout << "post_upscaled_xmax: " << upscaled_xmax << std::endl;
                //std::cout << "post_upscaled_ymax: " << upscaled_ymax << std::endl;
                //std::cout << "post_int_upscaled_xmin: " << int_upscaled_xmin << std::endl;
                //std::cout << "post_int_upscaled_ymin: " << int_upscaled_ymin << std::endl;
                //std::cout << "post_int_upscaled_xmax: " << int_upscaled_xmax << std::endl;
                //std::cout << "post_int_upscaled_ymax: " << int_upscaled_ymax << std::endl;
                
                // get upscaled bounding box and extract image-patch/mask
                cv::Rect roi(int_upscaled_xmin, int_upscaled_ymin, int_upscaled_xmax-int_upscaled_xmin, int_upscaled_ymax-int_upscaled_ymin);
                //std::cout << "created rect created" << std::endl;
                cv::Mat image_roi = frame(roi);
                cv::Size image_roi_image_size = image_roi.size();
                //std::cout << "image_roi_image_size created: " << image_roi_image_size.height << std::endl;
                //std::cout << "image_roi_image_size created: " << image_roi_image_size.width << std::endl;
                //debug only cv::imwrite("/home/nvidia/pose_image_roi.jpg", image_roi);
                //preprocessing happens inside forward function
                // why point3f and not 2f? 
                // we're using z as the confidence
                std::vector<cv::Point3f> pose_coords = pose_forward(image_roi, upscaled_xmin, upscaled_ymin, upscaled_xmax, upscaled_ymax);
                results->boxes[i].xmin = xmin;
                results->boxes[i].ymin = ymin;
                results->boxes[i].xmax = xmax;
                results->boxes[i].ymax = ymax;                 
                results->boxes[i].points = pose_coords;
                //results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space  ()
                //results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                //results->boxes[i].xmax = xmax * (640.0/512.0);
                //results->boxes[i].ymax = ymax / (512.0/480.0);
                results->boxes[i].id = cls_id;
                results->boxes[i].score = score;
                new_num+=1;
            };
            results->num = new_num;
            std::cout << "torch array iter finished" << std::endl;            

            // free outputs
            //TVMArrayFree(input);
            //TVMArrayFree(output_tensor_ids);
            //TVMArrayFree(output_tensor_scores);
            //TVMArrayFree(output_tensor_bboxes);
            //input = nullptr;
            //output_tensor_ids = nullptr;
            //output_tensor_scores = nullptr;
            //output_tensor_bboxes = nullptr;
            //free(data_x);
            //data_x = nullptr;
            //std::cout << "freeing finished" << std::endl;
            return results;
        }

        std::vector<cv::Point3f> pose_forward(cv::Mat bbox_mask, float xmin, float ymin, float xmax, float ymax)
        {
            std::cout << "running pose forward" << std::endl;
            // get height/width dynamically
            cv::Size image_size = bbox_mask.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);
            int64_t in_shape[4] = {1, 3, pose_height, pose_width};
            int total_input = 3 * pose_width * pose_height;
            
            DLTensor *output_tensor_heatmap;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_heatmap_size, pose_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_heatmap);
            std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(bbox_mask, pose_width, pose_height, true);
            cv::Size processed_image_size = processed_image.size();
            std::cout << "preprocess finished: " << std::endl;
            std::cout << "preprocess height: " << processed_image_size.height << std::endl;
            std::cout << "preprocess width: " << processed_image_size.width << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) pose_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            std::cout << "run/getoutput/setinput finished" << std::endl;
  
            // https://github.com/apache/incubator-tvm/issues/979?from=timeline
            //"This may give you some ideas to start with.
            //In general you want to use pinned memory and you want
            //to interleave computation with copying; so you want to
            // be upload the next thing while you are computing the
            //current thing while you are downloading the last thing."
            TVMSynchronize(device_type, device_id, nullptr);
            get_output(0, output_tensor_heatmap);
            std::cout << "TVMSynchronize finished" << std::endl;  

            torch::Tensor ndarray_heat_map_full = torch::zeros({1, 17, 64, 48}, at::kFloat);

            TVMArrayCopyToBytes(output_tensor_heatmap, ndarray_heat_map_full.data_ptr(), 1*17*64*48 * sizeof(float));
            //std::cout << "saving array output " << std::endl;
            //auto bytes = torch::pickle_save(ndarray_heat_map_full);
            //std::ofstream fout("/home/nvidia/pose.zip", std::ios::out | std::ios::binary);
            //fout.write(bytes.data(), bytes.size());
            //fout.close();

            //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L172
            //heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
            //  
            // pytorch view vs. reshape; use of auto?
            auto ndarray_heat_map = ndarray_heat_map_full.view({17, 3072});
            //std::vector<int64_t> heatsize = ndarray_heat_map.sizes();
            //std::cout << "ndarray_heat_map reshape finished: " << ndarray_heat_map.sizes().size() << std::endl;
            
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L173
            // idx = nd.argmax(heatmaps_reshaped, 2)
            torch::Tensor idx = torch::argmax(ndarray_heat_map, 1);
            //std::vector<int64_t> idxsize = idx.sizes().size();
            //std::cout << "argmax finished: " << idx.sizes().size() << std::endl;
            
            // creat empty pred container
            torch::Tensor preds = torch::zeros({17, 2}, at::kFloat);
            // create accessors


            auto idx_accessor = idx.accessor<long,1>(); // 1, 17 -> batch_size, 17
            auto heat_map_accessor = ndarray_heat_map.accessor<float,2>(); // 1, 17, 1
            
            // vars to preset
            // pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L273
            // use z as container for probability
            float heatmap_width = 48.0f;
            float heatmap_height = 64.0f;
            std::vector<cv::Point3f> points;
            float w = (xmax - xmin) / 2.0f;
            float h = (ymax - ymin) / 2.0f;
            float center_x = xmin + w; 
            float center_y = ymin + h;
            //std::cout << "pose_forward w: " << w << std::endl;
            //std::cout << "pose_forward h: " << h << std::endl;
            //std::cout << "pose_forward center_x: " << center_x << std::endl;
            //std::cout << "pose_forward center_y: " << center_y << std::endl;
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L168
            // might have to use a diff var name
            for (size_t i = 0; i < 17; i++){
              float index = idx_accessor[i];
              //std::cout << "index: " << index << std::endl;
              float probability = heat_map_accessor[i][static_cast<int>(index)];
              //std::cout << "probability: " << probability << std::endl;
              
              //// python modulo vs c++ is dfff
              ////https://stackoverflow.com/questions/1907565/c-and-python-different-behaviour-of-the-modulo-operation
              //preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)
              //preds[:, :, 0] = (preds[:, :, 0]) % width
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L181
              // float modulo_pred = ((index % heatmap_width) + heatmap_width) % heatmap_width;
              // float floor_pred = std::floor(index / heatmap_width);
              int modulo_int = static_cast<int>(index) % static_cast<int>(heatmap_width);
              //std::cout << "modulo_int: " << modulo_int << std::endl;
              float modulo_pred = static_cast<float>(modulo_int);
              //std::cout << "modulo_pred: " << modulo_pred << std::endl;
              float floor = index / heatmap_width;
              //std::cout << "floor: " << floor << std::endl;
              float floor_pred = std::floor(floor);
              //std::cout << "floor_pred: " << floor_pred << std::endl;
              if (probability <= 0.0) {
                // zero out the pred if the prob is bad...
                //pred_mask = nd.tile(nd.greater(maxvals, 0.0), (1, 1, 2))
                //pred_mask = pred_mask.astype(np.float32)
                //preds *= pred_mask
                modulo_pred = 0.0f;
                floor_pred = 0.0f;
              }
              //std::cout << "modulo_pred_end: " << modulo_pred << std::endl;
              //std::cout << "floor_pred_end: " << floor_pred << std::endl;
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L289-L290
              float w_ratio = modulo_pred / heatmap_width;
              float h_ratio = floor_pred / heatmap_height;
              //std::cout << "w_ratio: " << w_ratio << std::endl;
              //std::cout << "h_ratio: " << h_ratio << std::endl;              
              cv::Point3f point;
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L291-L292
              //scale = np.array([w, h])
              //preds[i][:, 0] = scale[0] * 2 * w_ratio + center[0] - scale[0]
              //center = np.array([x0 + w, y0 + h])
              point.x = w * 2.0f * w_ratio + center_x - w;
              point.y = h * 2.0f * h_ratio + center_y - h;
              point.z = probability;
              //std::cout << "point.x: " << point.x << std::endl;
              //std::cout << "point.y: " << point.y << std::endl;
              points.push_back(point);
            }
            // free outputs
            TVMArrayFree(input);
            TVMArrayFree(output_tensor_heatmap);
            input = nullptr;
            output_tensor_heatmap = nullptr;
            free(data_x);
            data_x = nullptr;
            //std::cout << "freeing finished" << std::endl;
            return points;
        }
};



class NoNMSYoloFromConfig{
    private:
        //working: void * handle;
        std::unique_ptr<tvm::runtime::Module> detector_handle;
        std::unique_ptr<tvm::runtime::Module> pose_handle;

    public:
        // confident in these
        // set these as the default
        std::string deploy_lib_path;
        std::string deploy_graph_path;
        std::string deploy_param_path;
        bool gpu = true;
        int device_id = 0;// = 0;
        int dtype_code = kDLFloat;
        int dtype_bits = 32;
        int dtype_lanes = 1;
        int device_type = kDLGPU;
        // set default here???
        int detector_width;// = 512;
        int detector_height;// = 512;
        int detector_total_input;// = 3 * width * height;
        int in_ndim = 4;
        int detector_out_ndim = 3;
        int max_yolo_boxes = 100;
        // maybe we can dynamically set all of these
        int64_t no_nms_output_size[3] = {1, 322560, 6};
        float thresh = 0.3f;
        int n_dets;

        /**
         * function that reads both the yolo detector and the pose detector
         * 
        */
        NoNMSYoloFromConfig(std::string config_path, std::string ros_package_string) {
            // read config with nlohmann-json
            std::cout << "start model_config reading" << std::endl;
            json model_config;
            std::string package_path = "/opt/catkin_ws/src/open_ptrack/recognition";
            std::string full_path = package_path + config_path;
            std::ifstream json_read(full_path);
            json_read >> model_config;
            std::cout << "model_config read into memory" << std::endl;
            // read detector variables
            std::string detector_lib_path = model_config["detector_deploy_lib_path"];
            std::string detector_graph_path = model_config["detector_deploy_graph_path"];
            std::string detector_param_path = model_config["detector_deploy_param_path"];

            device_id = model_config["device_id"];
            detector_width = model_config["detector_width"]; //(512,512)
            detector_height = model_config["detector_height"]; //(512,512)
            gpu = model_config["gpu"];
            n_dets = model_config["n_dets"];
            no_nms_output_size[1] = n_dets;
            thresh = model_config["threshold"];
            detector_total_input = 1 * 3 * detector_width * detector_height;

            std::string detector_deploy_lib_path = package_path + detector_lib_path;
            std::string detector_deploy_graph_path = package_path + detector_graph_path;
            std::string detector_deploy_param_path = package_path + detector_param_path;

            // set device type -- I think this has to be set here...
            if (gpu){
                device_type = kDLGPU;
            } else {
                device_type = kDLCPU;
            }
            // DETECTOR READ
            // read deploy lib
            tvm::runtime::Module detector_mod_syslib = tvm::runtime::Module::LoadFromFile(detector_deploy_lib_path);
            // read deplpy json
            std::ifstream detector_json_in(detector_deploy_graph_path, std::ios::in);
            std::string detector_json_data((std::istreambuf_iterator<char>(detector_json_in)), std::istreambuf_iterator<char>());
            detector_json_in.close();
            // get global function module for graph runtime
            tvm::runtime::Module detector_mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(detector_json_data, detector_mod_syslib,
                                                                                                  device_type, device_id);
            this->detector_handle.reset(new tvm::runtime::Module(detector_mod));
            // parameters in binary
            std::ifstream detector_params_in(detector_deploy_param_path, std::ios::binary);
            std::string detector_params_data((std::istreambuf_iterator<char>(detector_params_in)), std::istreambuf_iterator<char>());
            detector_params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray detector_params_arr;
            detector_params_arr.data = detector_params_data.c_str();
            detector_params_arr.size = detector_params_data.length();
            tvm::runtime::PackedFunc detector_load_params = detector_mod.GetFunction("load_params");
            detector_load_params(detector_params_arr);
        }
       
        /**
         * \brief function to normalize an image before it's processed by the network
         * \param[in] the raw cv::mat image
         * \return the normalized version of the iamge.
         */  
        cv::Mat preprocess_image(cv::Mat frame, int width, int height, bool convert){
            cv::Size new_size = cv::Size(width, height); // or is it height width????
            cv::Mat resized_image;
            if (convert){
              cv::Mat rgb;
              // bgr to rgb
              cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
              cv::resize(rgb, resized_image, new_size);
            } else {
              cv::resize(frame, resized_image, new_size);
            }
            // resize to 512x512
            cv::Mat resized_image_floats(new_size, CV_32FC3);
            // convert resized image to floats and normalize
            resized_image.convertTo(resized_image_floats, CV_32FC3, 1.0f/255.0f);
            //mimic mxnets 'to_tensor' function
            cv::Mat normalized_image(new_size, CV_32FC3);
            // mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
            cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
            cv::Mat temp;
            temp = resized_image_floats - mean;
            normalized_image = temp / theta;
            return normalized_image;
        }

        yoloresults* forward_full(cv::Mat frame)
        {
            std::cout << "starting function" << std::endl;
            // get height/width dynamically
            cv::Size image_size = frame.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);

            //Set constants and variables -- this is a prob,
            // how to do this at runtime?
            //constexpr int const_dtype_code = dtype_code;
            //constexpr int const_dtype_bits = dtype_bits;
            //constexpr int const_dtype_lanes = dtype_lanes;
            //constexpr int const_device_type = device_type;
            //constexpr int const_device_id = device_id;
            //int64_t in_shape[4] = {1, in_c, in_h, in_w};
            int64_t in_shape[4] = {1, 3, detector_height, detector_width};
            int total_input = 3 * detector_width * detector_height;
            std::cout << "width: " << detector_width << std::endl;
            std::cout << "height: " << detector_height << std::endl;
            std::cout << "total_input: " << total_input << std::endl;
            std::cout << "device_id: " << device_id << std::endl;
            std::cout << "dtype_code: " << dtype_code << std::endl;
            std::cout << "dtype_bits: " << dtype_bits << std::endl;
            std::cout << "dtype_lanes: " << dtype_lanes << std::endl;
            std::cout << "device_type: " << device_type << std::endl;

            DLTensor *output_for_nms;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            // allocate memory for results
            yoloresults* results = (yoloresults*)calloc(1, sizeof(yoloresults));
            results->num = 100;
            results->boxes = (bbox_result*)calloc(100, sizeof(bbox_result));

            std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(no_nms_output_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_for_nms);
            std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(frame, detector_width, detector_height, true);
            std::cout << "preprocess finished" << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) detector_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            std::cout << "run/getoutput/setinput finished" << std::endl;
  
            // https://github.com/apache/incubator-tvm/issues/979?from=timeline
            //"This may give you some ideas to start with.
            //In general you want to use pinned memory and you want
            //to interleave computation with copying; so you want to
            // be upload the next thing while you are computing the
            //current thing while you are downloading the last thing."
            TVMSynchronize(device_type, device_id, nullptr);
            get_output(0, output_for_nms);
            std::cout << "TVMSynchronize finished" << std::endl;  

            // copy to output
            //ulsMatF(int cols, int rows, int channels)
            //at(int channel, int row, int c
            MatF yolo_output(6, n_dets, 1); //ulsMatF yolo_output(6, n_dets, 1);
            TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
            //TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
            std::cout << "TVMSynchronize finished" << std::endl;  
            
            std::cout << "starting nms" << std::endl;
            //auto tick = Clock::now();
            std::vector<sortable_result> tvm_results;
            std::vector<sortable_result> proposals;
            proposals.clear();
            tvm_nms_cpu(proposals, yolo_output, thresh, thresh, tvm_results);
            std::cout << "ending nms" << std::endl;

            TVMArrayFree(input);
            TVMArrayFree(output_for_nms); // may have to move this to the bottom.
            input = nullptr;
            free(data_x);
            data_x = nullptr;

            float fheight = static_cast<float>(img_height);
            float fwidth = static_cast<float>(img_width);
            int new_num = 0;
            for (int i = 0; i < tvm_results.size(); ++i) {

                float xmin;
                float ymin;
                float xmax;
                float ymax;

                float score = tvm_results[i].probs;
                float label = tvm_results[i].cls;
                if (score < thresh) continue;
                if (label < 0) continue;
                // people only
                if (label > 0) continue;

                int cls_id = static_cast<int>(label);
                xmin = tvm_results[i].xmin;
                ymin = tvm_results[i].ymin;
                xmax = tvm_results[i].xmax;
                ymax = tvm_results[i].ymax;
                //SCALE to frame height
                xmin = xmin * (img_width/detector_height); // move down to 480 space  ()
                ymin = ymin / (detector_width/img_height); // move up to 640
                xmax = xmax * (img_width/detector_height);
                ymax = ymax / (detector_width/img_height);
                
                results->boxes[i].xmin = xmin;
                results->boxes[i].ymin = ymin;
                results->boxes[i].xmax = xmax;
                results->boxes[i].ymax = ymax;
                results->boxes[i].id = cls_id;
                results->boxes[i].score = score;
                new_num+=1;
            };
            results->num = new_num;
            std::cout << "torch array iter finished" << std::endl;            
            return results;
        }
};

