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

                results->boxes[i].xmin = xmin * (640.0/256.0); // move down to 480 space
                results->boxes[i].ymin = ymin / (256.0/480.0); // move up to 640
                results->boxes[i].xmax = xmax * (640.0/256.0);
                results->boxes[i].ymax = ymax / (256.0/480.0);
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

class TVMObjectDetectorGPU{
    private:
        //working: void * handle;
        std::unique_ptr<tvm::runtime::Module> handle;

    public:
        float img_width;
        float img_height;
        int deploy_device_id;
        int deploy_device_type;
        std::string deploy_lib_path;
        std::string deploy_graph_path;
        std::string deploy_param_path;
        int max_boxes = 100;
        // we have have some kind of way to make this configureable
        int yolo_width = 512;
        int yolo_height = 512;
        static constexpr int deploy_dtype_code = kDLFloat;
        static constexpr int deploy_dtype_bits = 32;
        static constexpr int deploy_dtype_lanes = 1;
        static constexpr int deploy_in_ndim = 4;
        static constexpr int deploy_out_ndim = 3;
        static constexpr int max_yolo_boxes = 100;
        // this has to match the
        static constexpr int64_t deploy_in_shape[deploy_in_ndim] = {1, 3, 512, 512};
        static constexpr int64_t deploy_tvm_id_and_score_size[deploy_out_ndim] = {1, 100, 1};
        static constexpr int64_t deploy_tvm_box_size[deploy_out_ndim] = {1, 100, 4};    

        TVMObjectDetectorGPU(std::string model_folder, int standard_img_height, int standard_img_width) {
            // 848 width
            // 480 height
            img_width = static_cast<float>(standard_img_width);
            img_height = static_cast<float>(standard_img_height);
            // tvm module for compiled functions
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_folder + "/deploy_lib_gpu.so");
            // json graph
            std::ifstream json_in(model_folder + "/deploy_graph_gpu.json", std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();

            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int device_type = kDLGPU;
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
            cv::Size new_size = cv::Size(yolo_width, yolo_height);
            cv::Mat resized_image;
            cv::Mat rgb;
            // bgr to rgb
            cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RG
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

                results->boxes[i].xmin = xmin * (img_width/512.0); // move down to 480 space
                results->boxes[i].ymin = ymin / (512.0/img_height); // move up to 640
                results->boxes[i].xmax = xmax * (img_width/512.0);
                results->boxes[i].ymax = ymax / (512.0/img_height);
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

        yoloresults* forward_full(cv::Mat frame, float img_height, float img_width, float thresh)
        {
            
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

                // int in_c = 3, in_h = 480, in_w = 640;
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

        yoloresults* forward_full(cv::Mat frame, float img_height, float img_width, float thresh)
        {
            
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

                // int in_c = 3, in_h = 480, in_w = 640;
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