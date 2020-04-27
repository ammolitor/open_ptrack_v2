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

float mask_generator(cv::Mat original_image, bbox_result &bbox, std::vector processed_images){
  //cv::Mat processed_image = alpha_preprocess_image(frame);
  cv::Rect roi(bbox->xmin, bbox->ymin, bbox->xmax-bbox->xmin, bbox->ymax-bbox->ymin);
  cv::Mat image_roi = original_image(roi);
  //cv::Mat resized_roi;
  //TODO get correct size
  //cv::resize(image_roi, resized_roi, cv::Size(224,224), cv::INTER_AREA);
  cv::Mat processed_roi = preprocess_image(image_roi, 224, 224);
  // we add this to a processed_roi container????  
  processed_images.push_back(processed_roi);
}

// must run network per image in tvm...
 



class PoseTVMFromConfig{
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


        // can only have 1 output, so we run this algo for everyone in the image
        // rather than a single batch
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

            DLTensor *output_tensor_heatmap;
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
            get_output(0, output_tensor_heatmap);
            std::cout << "TVMSynchronize finished" << std::endl;  

            torch::Tensor ndarray_heat_map = torch::zeros({1, 17, 64, 48}, at::kFloat);

            TVMArrayCopyToBytes(output_tensor_heatmap, ndarray_heat_map.data_ptr(), 1*17*64*48 * sizeof(float));

            // from the get_max_pred gluoncv function???
            //heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
            ndarray_heat_map.reshape({1, 17, 3072});
            

            //((n % M) + M) % M
            torch::Tensor idx = torch::argmax(ndarray_heat_map, 2);
            //torch::Tensor idx = torch::max(ndarray_heat_map, 2);
            torch::Tensor preds = torch::zeros({1, 17, 2}, at::kFloat);

            auto idx_accessor = idx.accessor<float,2>(); // 1, 17, 1
            auto heat_map_accessor = ndarray_heat_map.accessor<float,3>(); // 1, 17, 1

            for (size_t i = 0; i < 17; i++)
            {
              float index = idx_accessor[0][i];
              float probability = heat_map_accessor[0][i][static_cast<int>(index)];

              ////https://stackoverflow.com/questions/1907565/c-and-python-different-behaviour-of-the-modulo-operation
              //preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)
              //preds[:, :, 0] = (preds[:, :, 0]) % width
              int modulo_pred = ((index % width) + width) % width
              int floor_pred = std::floor(index / width);

              

            }
            





            // dunno how to get the maxvals yet...


    idx = nd.argmax(heatmaps_reshaped, 2)
    maxvals = nd.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = nd.floor((preds[:, :, 1]) / width)

    pred_mask = nd.tile(nd.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals






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

/**
 * @brief The TVMDetectionNode
 */
class TVMDetectionNode {
  private:
    ros::NodeHandle node_;
    //std::unique_ptr<YoloTVMGPU256> tvm_object_detector;
   // std::unique_ptr<YoloTVMGPU> tvm_object_detector;
    std::unique_ptr<YoloTVMFromConfig> tvm_object_detector;
    // TF listener
    tf::TransformListener tf_listener;
    // only need this if I need to debug
    //image_transport::ImageTransport it;
    
    // ROS
    dynamic_reconfigure::Server<recognition::GenDetectionConfig> cfg_server;
    ros::ServiceServer camera_info_matrix_server;

    // Publishers
    ros::Publisher detections_pub;
    //ros::Publisher image_pub;
    image_transport::Publisher image_pub;

    // Subscribers
    ros::Subscriber rgb_sub;
    ros::Subscriber camera_info_matrix;
    ros::Subscriber detector_sub;

    // Message Filters
    message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub;
    message_filters::Subscriber<sensor_msgs::Image> depth_image_sub;

    // Message Synchronizers 
    typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximatePolicy;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    boost::shared_ptr<ApproximateSync> approximate_sync_;

    // vars
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

  public:
    // Set camera matrix transforms
    Eigen::Matrix3f intrinsics_matrix;
    bool camera_info_available_flag = false;
    double _cx;
    double _cy;
    double _constant_x;
    double _constant_y;
    bool people_only = true;
    image_transport::ImageTransport it;
    json zone_json;
    json master_json;
    int n_zones;
    // use this for tests
    bool json_found = false;

    /**
     * @brief constructor
     * @param nh node handler
     */
    TVMDetectionNode(ros::NodeHandle& nh, std::string sensor_string, json zone_json):
      node_(nh), it(node_)
      {
        
        try
        {
          //json zone_json;
          //std::string area_package_path = ros::package::getPath("recognition");
          //std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
          //std::ifstream area_json_read(area_hard_coded_path);
          //area_json_read >> zone_json;
          //double test;
          //std::cout << "zone_json test: " << zone_json["0"]["d415"]["min"]["d415"]["x"] << std::endl;
          
          // get the number of zones to scan.
          json master_config;
          std::string master_package_path = ros::package::getPath("recognition");
          std::string master_hard_coded_path = master_package_path + "/cfg/master.json";
          std::ifstream master_json_read(master_hard_coded_path);
          master_json_read >> master_config;
          n_zones = master_config["n_zones"]; //the path to the detector model file
          std::cout << "n_zones: " << n_zones << std::endl;
          json_found = true;
        }
        catch(const std::exception& e)
        {
          std::cerr << "json master/area not found: "<< e.what() << '\n';
        }
        
        // Publish Messages
        detections_pub = node_.advertise<opt_msgs::DetectionArray>("/objects_detector/detections", 3);

        // Subscribe to Messages
        rgb_image_sub.subscribe(node_, sensor_string +"/color/image_rect_color", 1);
        depth_image_sub.subscribe(node_, sensor_string+"/depth/image_rect_raw", 1);
        
        image_pub = it.advertise(sensor_string + "/objects_detector/image", 1);

        // Camera callback for intrinsics matrix update
        camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &TVMDetectionNode::camera_info_callback, this);

        //Time sync policies for the subscribers
        approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(10), rgb_image_sub, depth_image_sub));
        // _1 = rgb_image_sub
        // _2 = depth_image_sub
        // _3 = zone_json or zone_json
        approximate_sync_->registerCallback(boost::bind(&TVMDetectionNode::callback, this, _1, _2, zone_json));

        // create callback config 
        cfg_server.setCallback(boost::bind(&TVMDetectionNode::cfg_callback, this, _1, _2));      

        // create object-detector pointer
        //tvm_object_detector.reset(new YoloTVMGPU256(model_folder_path));
        //tvm_object_detector.reset(new YoloTVMGPU(model_folder_path));
        // maybe have this in
        // arg one HAS to have / in front of path
        // TODO add that to debugger
        tvm_object_detector.reset(new YoloTVMFromConfig("/cfg/model.json", "recognition"));
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
    /**
     * @brief callback for camera information that does detection on images
     *  and publishes the detections to specific topics
     * @param rgb_image  the rgb image message
     * @param depth_image  the depth/stereo image message
     * @param zone_json the json that contains the zone information
     */
    void callback(const sensor_msgs::Image::ConstPtr& rgb_image,
                  const sensor_msgs::Image::ConstPtr& depth_image,
                  json zone_json) {
        
      std::cout << "running algorithm callback" << std::endl;
    
      //tf_listener.waitForTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
      //tf_listener.lookupTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ir2rgb_transform);
      //tf_listener.waitForTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
      //tf_listener.lookupTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), world2rgb_transform);

      // transform to eigen
      //tf::transformTFToEigen(world2rgb_transform, world2rgb);
      //tf::transformTFToEigen(ir2rgb_transform, ir2rgb);


      // find a better way to do this...
      // json call back...
      //std::string area_package_path = ros::package::getPath("recognition");
      //std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
      //std::ifstream area_json_read(area_hard_coded_path);
      //area_json_read >> zone_json;

      // how to persist zone_json;

      // set message vars here
      cv_bridge::CvImagePtr cv_ptr_rgb;
      cv_bridge::CvImage::Ptr  cv_ptr_depth;
      cv::Mat cv_image;
      cv::Mat cv_depth_image;
      cv::Mat cv_image_clone;
      
      // set detection variables here
      yoloresults* output;
      cv::Size image_size;
      float height;
      float width;
      ros::Time begin;
      double duration;

      // set publication messages vars here
      // generate new detection array message with the header from the rbg image
      opt_msgs::DetectionArray::Ptr detection_array_msg(new opt_msgs::DetectionArray);
      detection_array_msg->header = rgb_image->header;
      detection_array_msg->confidence_type = std::string("yolo");
      detection_array_msg->image_type = std::string("rgb");
      // set detection intrinsic matrix from camera variables
      for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
          detection_array_msg->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
        }
      }

      // NOTE
      // convert message to usable formats
      // available encoding types:
      // ---- sensor_msgs::image_encodings::BGR8
      // ---- sensor_msgs::image_encodings::TYPE_16UC1;
      // ---- sensor_msgs::image_encodings::TYPE_32UC1;
    
      cv_ptr_rgb = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
      cv_image = cv_ptr_rgb->image;
      cv_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
      cv_depth_image = cv_ptr_depth->image;
      image_size = cv_image.size();
      height =  static_cast<float>(image_size.height);
      width =  static_cast<float>(image_size.width);
      cv_image_clone = cv_image.clone();

      std::cout << "running yolo" << std::endl;
      // forward inference of object detector
      begin = ros::Time::now();
      output = tvm_object_detector->forward_full(cv_image, .3);
      duration = ros::Time::now().toSec() - begin.toSec();
      printf("yolo detection time: %f\n", duration);
      printf("yolo detections: %ld\n", output->num);

      if (output->num >= 1) {
        for (int i = 0; i < output->num; i++) {
          // get the label and the object name
          float label = static_cast<float>(output->boxes[i].id);
          std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];

          // get the coordinate information
          float xmin = output->boxes[i].xmin;
          float ymin = output->boxes[i].ymin;
          float xmax = output->boxes[i].xmax;
          float ymax = output->boxes[i].ymax;

          // set the median of the bounding box
          float median_x = xmin + ((xmax - xmin) / 2.0);
          float median_y = ymin + ((ymax - ymin) / 2.0);

          // If the detect box coordinat is near edge of image, it will return a error 'Out of im.size().'
          if ( median_x < width*0.02 || median_x > width*0.98) continue;
          if ( median_y < height*0.02 || median_y > height*0.98) continue;
        
          // set the new coordinates of the image so that the boxes are set
          int new_x = static_cast<int>(median_x - (median_factor * (median_x - xmin)));
          int new_y = static_cast<int>(median_y - (median_factor * (median_y - ymin)));
          int new_width = static_cast<int>(2 * (median_factor * (median_x - xmin)));
          int new_height = static_cast<int>(2 * (median_factor * (median_y - ymin)));
        
          // TODO we'll have to vet this strategy but for now, we're trying it
          // this will get the median depth of the object at the center of the object,
          // however, we're moving towards the median depth as where the box hits the ground/
          // where the feet are..
          //float median_depth = cv_depth_image.at<float>(median_y, median_x) / 1000.0f;
          float median_depth = cv_depth_image.at<float>(new_y, median_x) / mm_factor;

          if (median_depth <= 0 || median_depth > 6.25) {
            std::cout << "median_depth " << median_depth << " rejecting" << std::endl;
            continue;
            }			
            
          // set the mx/my wtr the intrinsic camera matrix
          float mx = (median_x - _cx) * median_depth * _constant_x;
          float my = (median_y - _cy) * median_depth * _constant_y;

          // publish the messages
          if(std::isfinite(median_depth) && std::isfinite(mx) && std::isfinite(my)){
        
            // defaults to sending the message
            bool send_message = true;
            if (people_only) {
              if (label > 0) { // above 0 == not-hawtdog.
                send_message = false;
              }
            }

            if (send_message) {
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
              
              // adding this so scan which zone the given detection is in 
              if (json_found){
                bool inside_area_cube = false;
                int zone_id;
                std::string zone_string;                  
                double x_min;
                double y_min;
                double z_min;
                double x_max;
                double y_max;
                double z_max;
                for (zone_id = 0; zone_id < n_zones; zone_id++)
                {
                  // need a world view here bc each detection was transformed
                  // this will work for a singular cam, but would mean each cam would have to tune
                  // to the specific area; which I think would be fine. // but will need
                  // to test to be sure
                  // a given detection can be in only one place at one time, thus it can't be in
                  // multiple zones
                  zone_string = std::to_string(zone_id);
                  // type must be number but is null...
                  //https://github.com/nlohmann/json/issues/1593

                  x_min = zone_json[zone_string][sensor_name]["min"][sensor_name]["x"];
                  y_min = zone_json[zone_string][sensor_name]["min"][sensor_name]["y"];
                  z_min = zone_json[zone_string][sensor_name]["min"][sensor_name]["z"];
                  x_max = zone_json[zone_string][sensor_name]["max"][sensor_name]["x"];
                  y_max = zone_json[zone_string][sensor_name]["max"][sensor_name]["y"];
                  z_max = zone_json[zone_string][sensor_name]["max"][sensor_name]["z"];
                  
                  std::cout << "x_min: " << x_min << std::endl;
                  std::cout << "y_min: " << y_min << std::endl;
                  std::cout << "z_min: " << z_min << std::endl;
                  std::cout << "x_max: " << x_max << std::endl;
                  std::cout << "y_max: " << y_max << std::endl;
                  std::cout << "z_max: " << z_max << std::endl;
                  std::cout << "mx: " << mx << std::endl;
                  std::cout << "my: " << my << std::endl;
                  std::cout << "median_depth: " << median_depth << std::endl;

                  // pythonic representation of above
                  //double x_min = zone_json[zone_string][sensor_name]["min"][sensor_name]["x"];
                  //double y_min = zone_json[zone_string][sensor_name]["min"][sensor_name]["y"];
                  //double z_min = zone_json[zone_string][sensor_name]["min"][sensor_name]["z"];
                  //double x_max = zone_json[zone_string][sensor_name]["max"][sensor_name]["x"];
                  //double y_max = zone_json[zone_string][sensor_name]["max"][sensor_name]["y"];
                  //double z_max = zone_json[zone_string][sensor_name]["max"][sensor_name]["z"];
                  inside_area_cube = (mx <= x_max && mx >= x_min) && (my <= y_max && my >= y_min) && (median_depth <= z_max && median_depth >= z_min);
                  std::cout << "inside_cube: " << inside_area_cube << std::endl;
                  // I think this works. 
                  if (inside_area_cube) {
                    break;
                  }
                }

                if (inside_area_cube) {
                  detection_msg.zone_id = zone_id;
                  std::cout << "DEBUG -- INSIDE ZONE: " << zone_id << std::endl;
                } else {
                  // meaning they're in transit
                  detection_msg.zone_id = 1000;
                } 
              }
              
              detection_msg.object_name=object_name;            
              detection_array_msg->detections.push_back(detection_msg);
            
              cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
              cv::putText(cv_image_clone, object_name, cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
              // cv::imwrite("/home/nvidia/OUTPUTIMAGE.JPG", cv_image);
            }
          }
        }
      }
    // this will publish empty detections if nothing is found
    sensor_msgs::ImagePtr imagemsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_clone).toImageMsg();
    detections_pub.publish(detection_array_msg);
    image_pub.publish(imagemsg);
    free(output->boxes);
    free(output);
    }

    // THIS IS INSIDE THE DETECTOR
    /**
     * @brief callback for dynamic reconfigure
     * @param config  configure parameters
     * @param level   configure level
     */
    void cfg_callback(recognition::GenDetectionConfig& config, uint32_t level) {
      std::cout << "--- cfg_callback ---" << std::endl;
      std::string package_path = ros::package::getPath("recognition");
      std::cout << package_path << std::endl;
      model_folder_path = package_path + config.detector_path; //the path to the face detector model file
      std::cout << model_folder_path << std::endl;
      std::cout << "overwriting default model_folder_path" << std::endl;
      confidence_thresh = config.confidence_thresh; // the threshold for confidence of detection
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
      std::cout << "--- detector cfg_callback ---" << std::endl;
      std::string package_path = ros::package::getPath("recognition");
      std::string full_path = package_path + hard_coded_path;
      std::ifstream json_read(full_path);
      json_read >> model_config;

      model_folder_path = model_config["detector_path"]; //the path to the detector model file
      confidence_thresh = model_config["confidence_thresh"]; // the threshold for confidence of detection
    }
};

int main(int argc, char** argv) {
  // read from master config
  // perhaps even simply read from the config in the begining instead of 
  // constantly polling the dynamic reconfigure? or do both?
  // I dunno
  // NOTE: using json in main() is the way to persist across callbacks...

  std::string sensor_name;
  json master_config;
  std::string package_path = ros::package::getPath("recognition");
  std::string master_hard_coded_path = package_path + "/cfg/master.json";
  std::ifstream json_read(master_hard_coded_path);
  json_read >> master_config;
  sensor_name = master_config["sensor_name"]; //the path to the detector model file

  json zone_json;
  std::string area_package_path = ros::package::getPath("recognition");
  std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
  std::ifstream area_json_read(area_hard_coded_path);
  area_json_read >> zone_json;

  std::cout << "--- tvm_detection_node ---" << std::endl;
  ros::init(argc, argv, "tvm_detection_node");
  // something is off here... with the private namespace
  ros::NodeHandle nh;
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  TVMDetectionNode node(nh, sensor_name, zone_json);
  std::cout << "detection node init " << std::endl;
  ros::spin();
  return 0;
}
