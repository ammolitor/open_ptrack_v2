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
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <boost/format.hpp>
//#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>

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
#include <opt_msgs/Joint2DMsg.h>
#include <opt_msgs/Joint3DMsg.h>
#include <opt_msgs/SkeletonMsg.h>
#include <opt_msgs/SkeletonArrayMsg.h>

///#include <open_ptrack/yolo_tvm.hpp>
#include <dynamic_reconfigure/server.h>
// TODO change to proper config
//#include <recognition/GenDetectionConfig.h>
#include <recognition/FaceDetectionConfig.h>
#include <recognition/GenDetectionConfig.h>

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

enum SkeletonJoints
{
  HEAD = 0,
  NECK,
  RSHOULDER,
  RELBOW,
  RWRIST,
  LSHOULDER,
  LELBOW,
  LWRIST,
  RHIP,
  RKNEE,
  RANKLE,
  LHIP,
  LKNEE,
  LANKLE,
  CHEST,
  SIZE
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
        cv::Mat preprocess_image(cv::Mat frame, int width, int height){
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
            cv::Mat processed_image = preprocess_image(frame, detector_width, detector_height);
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
                std::cout << "yolo_forward: center_x" << center_x << std::endl;
                std::cout << "yolo_forward: center_y" << center_y << std::endl;

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
                  int_upscaled_xmax = img_width - 1;
                  upscaled_xmax = img_width - 1.0;
                }
                if (0 >= int_upscaled_ymin){
                  int_upscaled_ymin = 1;
                  upscaled_ymin = 1.0;
                }
                if (int_upscaled_ymax > img_height){
                  int_upscaled_ymax = img_height - 1;
                  upscaled_ymax = img_height - 1.0;
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
                std::cout << "image_roi created" << std::endl;

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
            cv::Mat processed_image = preprocess_image(bbox_mask, pose_width, pose_height);
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

            //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L172
            //heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
            // 
            // pytorch view vs. reshape; use of auto?
            auto ndarray_heat_map = ndarray_heat_map_full.view({1, 17, 3072});
            //std::vector<int64_t> heatsize = ndarray_heat_map.sizes();
            std::cout << "ndarray_heat_map reshape finished: " << ndarray_heat_map.sizes().size() << std::endl;
            
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L173
            // idx = nd.argmax(heatmaps_reshaped, 2)
            torch::Tensor idx = torch::argmax(ndarray_heat_map, 2);
            //std::vector<int64_t> idxsize = idx.sizes().size();
            std::cout << "argmax finished: " << idx.sizes().size() << std::endl;
            
            // creat empty pred container
            torch::Tensor preds = torch::zeros({17, 2}, at::kFloat);
            // create accessors


            //terminate called after throwing an instance of 'c10::Error'
            //what():  expected scalar type Float but found Long (data_ptr<float> at /opt/src/pytorch/torch/include/ATen/core/TensorMethods.h:6321)
            //frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x78 (0x7f6b1a6258 in /opt/src/pytorch/torch/lib/libc10.so)
            //frame #1: float* at::Tensor::data_ptr<float>() const + 0x1bc (0x558d0fd914 in /opt/catkin_ws/devel/lib/recognition/debug_pose_model)
            //frame #2: at::TensorAccessor<float, 2ul, at::DefaultPtrTraits, long> at::Tensor::accessor<float, 2ul>() const & + 0x58 (0x558d107240 in /opt/catkin_ws/devel/lib/recognition/debug_pose_model)
            // changed to long 
            auto idx_accessor = idx.accessor<long,2>(); // 1, 17 -> batch_size, 17
            auto heat_map_accessor = ndarray_heat_map.accessor<float,3>(); // 1, 17, 1
            
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
              float index = idx_accessor[0][i];
              std::cout << "index: " << index << std::endl;
              float probability = heat_map_accessor[0][i][static_cast<int>(index)];
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


int main()
{

    std::string image_name = "/home/nvidia/soccer.png";
    cv::Mat cv_image;
    cv::Mat cv_image_clone;
    pose_results* output;
    bool ignore = true;
    cv_image = cv::imread(image_name);
    cv_image_clone = cv_image.clone();
    PoseFromConfig tvm_pose_detector("/cfg/pose_model.json", "recognition");
    output = tvm_pose_detector.forward_full(cv_image, .3);
    int gluon_to_rtpose[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
    if (output->num >= 1) {
      for (int i = 0; i < output->num; i++) {
        // get the label and the object name
        float label = static_cast<float>(output->boxes[i].id);
        // TODO do this with something callable from the json file
        std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];

        // get the coordinate information
        float xmin = output->boxes[i].xmin;
        float ymin = output->boxes[i].ymin;
        float xmax = output->boxes[i].xmax;
        float ymax = output->boxes[i].ymax;
        std::vector<cv::Point3f> points = output->boxes[i].points;
        int num_parts = points.size();

        for (size_t i = 0; i < num_parts; i++){
          int rtpose_part_index = gluon_to_rtpose[i];
          /* code */
          // IGNORE eyes/ears
          if (rtpose_part_index == -1){
            continue;
          } else {
            cv::Point3f point = points[i];
            float confidence = point.z;
            int cast_x = static_cast<int>(point.x);
            int cast_y = static_cast<int>(point.y);
            // debug this 
            cv::circle(cv_image_clone, cv::Point(cast_x, cast_y), 3, (0,255,0));
          }
        }
        float confidence = 0.9f;
        cv::Point3f point_left_shoulder = points[5];
        cv::Point3f point_right_shoulder = points[6];
        cv::Point3f point_left_hip = points[11];
        cv::Point3f point_right_hip = points[12];

        // ******* NECK == joint location 1
        // center of each shoulder == chest
        float x = (point_left_shoulder.x + point_right_shoulder.x) / 2;
        float y = (point_left_shoulder.y + point_right_shoulder.y) / 2;
        int cast_point_x = static_cast<int>(x);
        int cast_point_y = static_cast<int>(y);
        cv::circle(cv_image_clone, cv::Point(cast_point_x, cast_point_y), 3, (0,255,0));
        
        // ******** CHEST
        // weighted mean from rtpose
        // TODO if this looks ugly, we'll just use the neck
        float cx = (point_left_hip.x + point_right_hip.x) * 0.4 + (point_left_shoulder.x + point_right_shoulder.x) * 0.1;
        float cy = (point_left_hip.y + point_right_hip.y) * 0.4 + (point_left_shoulder.y + point_right_shoulder.y) * 0.1;
        int cast_cx = static_cast<int>(cx);
        int cast_cy = static_cast<int>(cy);
        cv::circle(cv_image_clone, cv::Point(cast_cx, cast_cy), 3, (0,0,255));
        
        cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
        cv::putText(cv_image_clone, object_name, cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
        // cv::imwrite("/home/nvidia/OUTPUTIMAGE.JPG", cv_image);
        }
      }

    // display drawn image
    // output image
    cv::imwrite("/home/nvidia/OUTPUTIMAGE.JPG", cv_image_clone);

    return 0;
}
