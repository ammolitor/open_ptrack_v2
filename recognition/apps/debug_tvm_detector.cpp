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
#include <mutex>
#include <memory>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <boost/format.hpp>
//#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_sequencer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <opt_msgs/DetectionArray.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

///#include <open_ptrack/yolo_tvm.hpp>
#include <dynamic_reconfigure/server.h>
// TODO change to proper config
//#include <recognition/GenDetectionConfig.h>
#include <recognition/FaceDetectionConfig.h>

// not sure if this is the correct json reading code
// but will be easier than continually recompiling t
// import header files
#include <nlohmann/json.hpp>
using json = nlohmann::json;
typedef sensor_msgs::Image Image;
typedef sensor_msgs::CameraInfo CameraInfo;
using namespace message_filters::sync_policies;


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

// ROI struct
struct ROI {
  int left;
  int top;
  int left_width; //left + width
  int top_height; //top + height
};

// rois pointer
struct regions_of_interest{
  ROI* rois;
  int num;
};

// positions pointer
struct positionxy {
  int x;
  int y;
  int z;
  int w;
};

// matrix for face posiitons
struct ObjectsMat{
  cv::Mat positions;
  int num;
};

/**
 * @brief calculate the median depth of a given object
 * @param Input the object mask within the image
 * @returns float value that represents the depth of the rectangle
 */
float calc_median_of_object(const cv::Mat& Input){
  std::vector<float> array;
  if (Input.isContinuous()){
    array.assign(Input.datastart, Input.dataend);
  } else {
    for (int i = 0; i < Input.rows; ++i) {
      array.insert(array.end(), Input.ptr<float>(i), Input.ptr<float>(i)+Input. cols);
    }
  }
  std::nth_element(array.begin() , array.begin() + array.size() * 0.5, array.end());
  return array[array.size() * 0.5];
}


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
            int device_type = kDLCPU;//kDLGPU
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

            using Clock = std::chrono::high_resolution_clock;
            using Timepoint = Clock::time_point;
            using Duration = std::chrono::duration<double>;
            auto start = Clock::now();

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

                results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space
                results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                results->boxes[i].xmax = xmax * (640.0/512.0);
                results->boxes[i].ymax = ymax / (512.0/480.0);
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

                results->boxes[i].xmin = xmin * (640.0/512.0); // move down to 480 space
                results->boxes[i].ymin = ymin / (512.0/480.0); // move up to 640
                results->boxes[i].xmax = xmax * (640.0/512.0);
                results->boxes[i].ymax = ymax / (512.0/480.0);
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
        int dtype_code;// = kDLFloat;
        int dtype_bits;// = 32;
        int dtype_lanes;// = 1;
        int device_type;// = kDLGPU;
        // set default here???
        int width;// = 512;
        int height;// = 512;
        int64_t in_shape[4];// = {1, 3, height, width};
        //int64_t tvm_id_and_score_size[3] = {1, 100, 1};
        //int64_t tvm_box_size[3];// = {1, 100, 4};
        int total_input = 3 * width * height;
        int in_ndim = 4;
        int out_ndim = 3;
        int64_t tvm_id_and_score_size[3] = {1, 100, 1};
        int64_t tvm_box_size[3] = {1, 100, 4};

        YoloTVMFromConfig(std::string config_path, std::string ros_package_string) {
            // read config with nlohmann-json
            json model_config;
            std::string package_path = ros::package::getPath(ros_package_string);
            std::string full_path = package_path + config_path;
            std::ifstream json_read(full_path);
            json_read >> model_config;
            // read variables
            deploy_lib_path = config["deploy_lib_path"];
            deploy_graph_path = config["deploy_graph_path"];
            deploy_param_path = config["deploy_param_path"];
            device_id = config["device_id"];
            width = config["width"];
            height = config["height"];
            gpu = config["gpu"];
            //int64_t in_shape[4] = {1, 3, height, width};
            in_shape = {1, 3, height, width};
            // set device type
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
            return results;
        }  
};    

/**
 * @brief The TVMDetectionNode
 */
class TVMDetectionNode {
private:
  ros::NodeHandle node_;
  // detector
  //std::unique_ptr<YoloTVMGPU256> tvm_object_detector;
  std::unique_ptr<YoloTVMFromConfig> tvm_object_detector;
  // TF listener
  tf::TransformListener tf_listener;
  // only need this if I need to debug
  //image_transport::ImageTransport it;

  // ROS
  ros::ServiceServer tvm_object_detector_service;
  dynamic_reconfigure::Server<recognition::FaceDetectionConfig> cfg_server;
  ros::ServiceServer camera_info_matrix_server;

  // Servers
  ros::ServiceServer set_predefined_faces_service;
  ros::ServiceServer save_registered_faces_service;
  ros::ServiceServer load_registered_faces_service;

  // Publishers
  ros::Publisher detections_pub;
  //ros::Publisher pub_local;
  // timer if we want..
  //ros::Timer pub_timer;

  // Subscribers
  ros::Subscriber rgb_sub;
  ros::Subscriber camera_info_matrix;
  ros::Subscriber detector_sub;

  // Message Filters
  message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub; //*;
  message_filters::Subscriber<sensor_msgs::Image> depth_image_sub; //*;
  // message_filters::Subscriber<opt_msgs::DetectionArray> detections_sub; //*;
  // message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info; //*;

  // Message Synchronizers
  typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximatePolicy;
  typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
  boost::shared_ptr<ApproximateSync> approximate_sync_;

  //using ApproximatePolicy = ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;// ApproximatePolicy;
  //ApproximatePolicy policy(10);
  //policy.setMaxIntervalDuration(ros::Duration(1.e-3));
  //typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
  //boost::shared_ptr<ApproximateSync> approximate_sync_;

  // The transforms for the TF listener
  Eigen::Affine3d world2rgb;
  Eigen::Affine3d ir2rgb;
  tf::StampedTransform world2rgb_transform;
  tf::StampedTransform ir2rgb_transform;

  // Set camera matrix transforms
  //Eigen::Matrix3f intrinsics_matrix;
  //bool camera_info_available_flag = false;
  //double _cx;
  //double _cy;
  //double _constant_x;
  //double _constant_y;
  std::string encoding;
  float mm_factor = 1000.0f;
  float median_factor = 0.1;
  // d435: name of the given sensor
  std::string sensor_name; // = "realsense_head"

  // Config parameters
  //the path to the face detector model file
  std::string model_folder_path;
  // the threshold for confidence of face detection
  double confidence_thresh;
  //the width of a face detection ROI in the world space [m]
  double roi_width;
  // if true, ROIs are calculated from the top positions of detected clusters
  bool calc_roi_from_top;
  // the distance between the top position of a human cluster and the center of the face [m]
  double head_offset_z_top;
  // the distance between the centroid of a human cluster and the center of the face [m]
  double head_offset_z_centroid;
  // the face detection ROI is upscaled so that its width get larger than #upscale_minsize
  int upscale_minsize;
  // viz or not to viz
  bool visualization;
  // Set camera matrix transforms
  //Eigen::Matrix3f intrinsics_matrix;
  //bool camera_info_available_flag = false;
  //double _cx;
  //double _cy;
  //double _constant_x;
  //double _constant_y;

public:
  // Set camera matrix transforms
  Eigen::Matrix3f intrinsics_matrix;
  bool camera_info_available_flag = false;
  double _cx;
  double _cy;
  double _constant_x;
  double _constant_y;

  /**
   * @brief constructor
   * @param nh node handler
   */
  TVMDetectionNode(ros::NodeHandle& nh, std::string sensor_string):
    node_(nh)
    {
      std::cout << "init stage " << std::endl;
      std::cout << "sensor_string "<< sensor_string << std::endl;
      // Publish Messages
      detections_pub = node_.advertise<opt_msgs::DetectionArray>("/objects_detector/detections", 3);
      //pub_local = node_.advertise<opt_msgs::DetectionArray>("/objects_detector/detections", 10);
      // Subscribe to Messages
      rgb_image_sub.subscribe(node_, sensor_string +"/color/image_rect_color", 1);
      depth_image_sub.subscribe(node_, sensor_string+"/depth/image_rect_raw", 1);
      // only used if there's a detection message...
      // detections_sub.subscribe(node_, "/objects_detector/detections", 10);
      //camera_info.subscribe(node_, sensor_string + "/color/camera_info", 1);
      //camera_info_matrix.subscribe(node_, sensor_string + "/color/camera_info", 10, &TVMDetectionNode::camera_info_callback, this);
      camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &TVMDetectionNode::camera_info_callback, this);

      // Camera-info callback for intrinsic matrix
      //camera_info_sub = node_.subscribe(camera_info, 1, camera_info_callback);
      //camera_info_matrix_server.setCallback(boost::bind(&TVMDetectionNode::camera_info_callback, this, _1);
      // Sync all subscriptions
      //ApproximatePolicy policy(10);
      //policy.setMaxIntervalDuration(ros::Duration(1.e-3));
      approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(10), rgb_image_sub, depth_image_sub));
      approximate_sync_->registerCallback(boost::bind(&TVMDetectionNode::callback, this, _1, _2));

      // create callback config
      cfg_server.setCallback(boost::bind(&TVMDetectionNode::cfg_callback, this, _1, _2));

      // create object-detector pointer
      tvm_object_detector.reset(new YoloTVMFromConfig("cfg/model.json", "recognition"))
      //tvm_object_detector.reset(new YoloTVMGPU256(model_folder_path));
      sensor_name = sensor_string;
   }

  void camera_info_callback(const CameraInfo::ConstPtr & msg){
    intrinsics_matrix << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
    _cx = msg->K[2];
    _cy = msg->K[5];
    _constant_x =  1.0f / msg->K[0];
    _constant_y = 1.0f /  msg->K[4];
    camera_info_available_flag = true;
  }

private:
  ///**
  // * @brief callback for camera information
  // * @param msg  the CameraInfo message
  // */
  //void camera_info_callback(const CameraInfo::ConstPtr & msg){
  //  std::cout << "camera callback" << std::endl;
  //  intrinsics_matrix << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
  //  _cx = msg->K[2];
  //  _cy = msg->K[5];
  //  _constant_x =  1.0f / msg->K[0];
  //  _constant_y = 1.0f /  msg->K[4];
  //  camera_info_available_flag = true;
  //}

  /**
   * @brief callback for camera information that does detection on images
   *  and publishes the detections to specific topics
   * @param rgb_image  the rgb image message
   * @param depth_image  the depth/stereo image message
   * @param rgb_info_msg  the CameraInfo message for the rgb cam
   */
  void callback(const sensor_msgs::Image::ConstPtr& rgb_image,
                const sensor_msgs::Image::ConstPtr& depth_image) {

    std::cout << "running callback" << std::endl;

    cv_bridge::CvImagePtr cv_ptr_rgb;
    cv_bridge::CvImage::Ptr  cv_ptr_depth;
    cv::Mat cv_image;
    cv::Mat cv_depth_image;
    yoloresults* output;

    // convert message to usable formats
    // available encoding types:
    // ---- sensor_msgs::image_encodings::BGR8
    // ---- sensor_msgs::image_encodings::TYPE_16UC1;
    // ---- sensor_msgs::image_encodings::TYPE_32UC1;
    cv_ptr_rgb = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
    cv_image = cv_ptr_rgb->image;
    cv_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
    cv_depth_image = cv_ptr_depth->image;

    cv::Size image_size = cv_image.size();
    float height =  static_cast<float>(image_size.height);
    float width =  static_cast<float>(image_size.width);

    ros::Time begin = ros::Time::now();
    // forward inference on object detection
    output = tvm_object_detector->forward_full(cv_image, .3);
    double duration = ros::Time::now().toSec() - begin.toSec();
    std::cout << "yolo detection time: " << duration<< std::endl;
    printf("yolo detections: %ld\n", output->num);

    //cv_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
    //cv_depth_image = cv_ptr_depth->image;

    //cv::Size image_size = cv_image.size();
    //float height =  static_cast<float>(image_size.height);
    //float width =  static_cast<float>(image_size.width);


    opt_msgs::DetectionArray::Ptr detection_array_msg(new opt_msgs::DetectionArray);
    detection_array_msg->header = rgb_image->header;

    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 3; j++){
        detection_array_msg->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
      }
    }
    detection_array_msg->confidence_type = std::string("yolo");
    detection_array_msg->image_type = std::string("rgb");

    // only publish detections if anything is detected
    if (output->num >= 1) {
      //std::cout << "passed output size test" << std::endl;
      // set detections array
      //opt_msgs::DetectionArray detection_array_msg;
      //detection_array_msg = *detections_msg;
      //detection_array_msg.header = rgb_image->header;
      // iter through the output and allocate to the detections
      // array
      for (int i = 0; i < output->num; i++) {
        // set detection message
        //const auto& detection = &detection_array_msg->detections[i];
        //std::cout << "passed for loop test: "<< i << std::endl;
        // get the float label
        float label = static_cast<float>(output->boxes[i].id);
        std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];
    // we don't need this because we only allocate boxes IF
        // the score is set already
        // -- if (output->boxes[i].score < thresh) continue;
        // -- if (output->boxes[i].id < 0) continue;

        // get the coordinate information
        float xmin = output->boxes[i].xmin;
        float ymin = output->boxes[i].ymin;
        float xmax = output->boxes[i].xmax;
        float ymax = output->boxes[i].ymax;
        //printf("xmin: %f\n", xmin);
  //printf("xmax: %f\n", xmax);
  //printf("ymin: %f\n", ymin);
  //printf("ymax: %f\n", ymax);
        // set the median of the bounding box
        float median_x = xmin + ((xmax - xmin) / 2.0);
        float median_y = ymin + ((ymax - ymin) / 2.0);
        //printf("median_x: %f\n", median_x);
  //printf("median_y: %f\n", median_y);
        //printf("height: %f\n", height);
  //printf("width: %f\n", width);

        // If the detect box coordinat is near edge of image, it will return a error 'Out of im.size().'
        if ( median_x < width*0.02 || median_x > width*0.98) continue;
        if ( median_y < height*0.02 || median_y > height*0.98) continue;

        // set the new coordinates of the image so that the boxes are set
        int new_x = static_cast<int>(median_x - (median_factor * (median_x - xmin)));
        int new_y = static_cast<int>(median_y - (median_factor * (median_y - ymin)));
        int new_width = static_cast<int>(2 * (median_factor * (median_x - xmin)));
        int new_height = static_cast<int>(2 * (median_factor * (median_y - ymin)));

        // set rectangle object to do depth info of image
        //cv::Rect rect(new_x, new_y, new_width, new_height);

        // get depth info from image
        //float median_depth = calc_median_of_object(cv_depth_image(rect)) / mm_factor;
        float median_depth = cv_depth_image.at<float>(median_y, median_x) / 1000.0f;

        //printf("new_x: %d\n", new_x);
        //printf("new_y: %d\n", new_y);
        //printf("new_width: %d\n", new_width);
        //printf("new_height: %d\n", new_height);
        //printf("median_depth: %f\n", median_depth);

  if (median_depth <= 0 || median_depth > 6.25) {
          std::cout << "median_depth " << median_depth << " rejecting" << std::endl;
          continue;
        }

        // set the mx/my wtr the intrinsic camera matrix
        float mx = (median_x - _cx) * median_depth * _constant_x;
        float my = (median_y - _cy) * median_depth * _constant_y;

  // publish the messages
        if(std::isfinite(median_depth) && std::isfinite(mx) && std::isfinite(my)){

          opt_msgs::Detection detection_msg;
          detection_msg.box_3D.p1.x = mx;
          detection_msg.box_3D.p1.y = my;
          detection_msg.box_3D.p1.z = median_depth;

          detection_msg.box_3D.p2.x = mx;
          detection_msg.box_3D.p2.y = my;
          detection_msg.box_3D.p2.z = median_depth;

          detection_msg.box_2D.x = median_x;
          detection_msg.box_2D.y = median_y;
          detection_msg.box_2D.width = 0;
          detection_msg.box_2D.height = 0;
          detection_msg.height = 0;
          detection_msg.confidence = 10;
          detection_msg.distance = median_depth;

          detection_msg.centroid.x = mx;
          detection_msg.centroid.y = my;
          detection_msg.centroid.z = median_depth;

          detection_msg.top.x = 0;
          detection_msg.top.y = 0;
          detection_msg.top.z = 0;

          detection_msg.bottom.x = 0;
          detection_msg.bottom.y = 0;
          detection_msg.bottom.z = 0;

          detection_msg.object_name=object_name;
          detection_array_msg->detections.push_back(detection_msg);
          }
        }
      //pub.publish(detection_array_msg);
      //pub_local.publish(detection_array_msg);
      }
    detections_pub.publish(detection_array_msg);
    //pub_local.publish(detection_array_msg);
    free(output->boxes);
    free(output);
  }

  // THIS IS INSIDE THE DETECTOR
  /**
   * @brief callback for dynamic reconfigure
   * @param config  configure parameters
   * @param level   configure level
   */
  void cfg_callback(recognition::FaceDetectionConfig& config, uint32_t level) {
    std::cout << "--- cfg_callback ---" << std::endl;
    std::string package_path = ros::package::getPath("recognition");
    std::cout << package_path << std::endl;
    model_folder_path = package_path + config.model_folder_path; //the path to the face detector model file
    std::cout << model_folder_path << std::endl;
    std::cout << "overwriting default model_folder_path" << std::endl;
    model_folder_path = "/home/nvidia/catkin_ws/src/open_ptrak/recognition/data/object_detector_folder";

    confidence_thresh = config.confidence_thresh; // the threshold for confidence of face detection
    roi_width = config.roi_width_; //the width of a face detection ROI in the world space [m]
    calc_roi_from_top = config.calc_roi_from_top; // if true, ROIs are calculated from the top positions of detected clusters
    head_offset_z_top = config.head_offset_z_top; // the distance between the top position of a human cluster and the center of the face [m]
    head_offset_z_centroid = config.head_offset_z_centroid; // the distance between the centroid of a human cluster and the center of the face [m]
    upscale_minsize = config.upscale_minsize; // the face detection ROI is upscaled so that its width get larger than #upscale_minsize
    visualization = config.visualization; // viz or not to viz
  }

  // THIS IS INSIDE THE DETECTOR
  /**
   * @brief callback for dynamic reconfigure
   * @param config  configure parameters
   * @param level   configure level
   */
  void json_cfg_callback(uint32_t level) {
    json model_config;
    std::string hard_coded_path = "/cfg/master.json";
    std::cout << "--- cfg_callback ---" << std::endl;
    std::string package_path = ros::package::getPath("recognition");
    std::string full_path = package_path + hard_coded_path;
    std::ifstream json_read(full_path);
    json_read >> model_config;

    model_folder_path = model_config["model_folder"]; //the path to the face detector model file
    confidence_thresh = model_config["confidence_thresh"]; // the threshold for confidence of face detection
    roi_width = model_config["roi_width_"]; //the width of a face detection ROI in the world space [m]
    calc_roi_from_top = model_config["calc_roi_from_top"]; // if true, ROIs are calculated from the top positions of detected clusters
    head_offset_z_top = model_config["head_offset_z_top"]; // the distance between the top position of a human cluster and the center of the face [m]
    head_offset_z_centroid = model_config["head_offset_z_centroid"]; // the distance between the centroid of a human cluster and the center of the face [m]
    upscale_minsize = model_config["upscale_minsize"]; // the face detection ROI is upscaled so that its width get larger than #upscale_minsize
    visualization = model_config["visualization"]; // viz or not to viz
  }
};

int main(int argc, char** argv) {
  std::string sensor_name;
  std::cout << "--- tvm_detection_node ---" << std::endl;
  ros::init(argc, argv, "tvm_detection_node");
  // something is off here... with the private namespace
  ros::NodeHandle nh;
  //nh.getParam("sensor_name", sensor_name);
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl;
  TVMDetectionNode node(nh, "d415");
  std::cout << "detection node init " << std::endl;
  ros::spin();
  return 0;
}