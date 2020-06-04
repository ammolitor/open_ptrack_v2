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


#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>
#include <pcl/filters/passthrough.h>


#include <open_ptrack/detection/ground_segmentation.h>
#include <open_ptrack/detection/ground_based_people_detection_app.h>
#include <open_ptrack/opt_utils/conversions.h>




///////////////////////////////////// all detection headers here
#include <iostream>
#include <string>


#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>

// PCL includes:
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

// Open PTrack includes:
#include <open_ptrack/detection/ground_segmentation.h>
#include <open_ptrack/detection/ground_based_people_detection_app.h>
#include <open_ptrack/opt_utils/conversions.h>

//Publish Messages
#include <opt_msgs/RoiRect.h>
#include <opt_msgs/Rois.h>
#include <std_msgs/String.h>
#include <sensor_msgs/CameraInfo.h>
#include <opt_msgs/Detection.h>
#include <opt_msgs/DetectionArray.h>


#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/people/person_cluster.h>
#include <pcl/people/head_based_subcluster.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Eigen>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <tf/transform_listener.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include<iostream>
#include<fstream>
#include <Eigen/Eigen>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opt_msgs/Detection.h>
#include <open_ptrack/detection/detection_source.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <open_ptrack/opt_utils/conversions.h>
#include <tf/tf.h>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/people/person_cluster.h>
#include <pcl/people/head_based_subcluster.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <open_ptrack/detection/person_classifier.h>
#include <Eigen/Eigen>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <tf/transform_listener.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include<iostream>
#include<fstream>

///////////////////////////////////// all detection headers here end





// not sure if this is the correct json reading code
// but will be easier than continually recompiling t
// import header files
#include <nlohmann/json.hpp>
using json = nlohmann::json;
typedef sensor_msgs::Image Image;
typedef sensor_msgs::CameraInfo CameraInfo;
// adding this 
using namespace message_filters::sync_policies;
using namespace std;
using namespace cv;
using namespace std;
using namespace cv;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef boost::shared_ptr<PointCloud> PointCloudPtr;
typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

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



void draw_skelaton(cv::Mat cv_image_clone, std::vector<cv::Point3f> points){
  int num_parts = points.size();
  int gluon_to_rtpose_map[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
  cv::Point3f nose_head = points[0];
  cv::Point3f left_shoulder = points[5];
  cv::Point3f right_shoulder = points[6];
  cv::Point3f left_elbow = points[7];
  cv::Point3f right_elbow = points[8];
  cv::Point3f left_wrist = points[9];
  cv::Point3f right_wrist = points[10];
  cv::Point3f left_hip = points[11];
  cv::Point3f right_hip = points[12];
  cv::Point3f left_knee = points[13];
  cv::Point3f right_knee = points[14];
  cv::Point3f left_ankle = points[15];
  cv::Point3f right_ankle = points[16];
  for (size_t i = 0; i < num_parts; i++){
    int rtpose_part_index = gluon_to_rtpose_map[i];
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
      cv::circle(cv_image_clone, cv::Point(cast_x, cast_y), 3, (0,0,0));
    }
  }
  // ******* NECK == joint location 1\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  // center of each shoulder == chest
  float x = (left_shoulder.x + right_shoulder.x) / 2;
  float y = (left_shoulder.y + right_shoulder.y) / 2;
  int cast_point_x = static_cast<int>(x);
  int cast_point_y = static_cast<int>(y);
  cv::circle(cv_image_clone, cv::Point(cast_point_x, cast_point_y), 3, (0,0,0));
  
  // ******** CHEST
  // weighted mean from rtpose
  // TODO if this looks ugly, we'll just use the neck
  float cx = (left_hip.x + right_hip.x) * 0.4 + (left_shoulder.x + right_shoulder.x) * 0.1;
  float cy = (left_hip.y + right_hip.y) * 0.4 + (left_shoulder.y + right_shoulder.y) * 0.1;
  int cast_cx = static_cast<int>(cx);
  int cast_cy = static_cast<int>(cy);
  cv::circle(cv_image_clone, cv::Point(cast_cx, cast_cy), 3, (0,0,0));

  //joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
  //                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  //                [5, 11], [6, 12], [11, 12],
  //                [11, 13], [12, 14], [13, 15], [14, 16]]
  //[0.        , 0.06666667, 0.13333333, 0.2       , 0.26666667,
  // 0.33333333, 0.4       , 0.46666667, 0.53333333, 0.6       ,
  // 0.66666667, 0.73333333, 0.8       , 0.86666667, 0.93333333,
  // 1.        ])
  //cv::line(legend_image, cv::Point(0,y_coord), cv::Point(100,y_coord),
  //       cv::Scalar(255*color(2), 255*color(1), 255*color(0)), 8)
  float color_map[16] = {0., 0.06666667, 0.13333333, 0.2, 0.26666667, 0.33333333, 0.4, 0.46666667, 0.53333333, 0.6, 0.66666667, 0.73333333, 0.8, 0.86666667, 0.93333333, 1.};
  cv::line(cv_image_clone, cv::Point(static_cast<int>(nose_head.x), static_cast<int>(nose_head.y)), cv::Point(static_cast<int>(x), static_cast<int>(y)), cv::Scalar(static_cast<int>(255.0*color_map[12]), static_cast<int>(255.0*color_map[12]), static_cast<int>(255.0*color_map[12])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_shoulder.x), static_cast<int>(left_shoulder.y)), cv::Point(static_cast<int>(right_shoulder.x), static_cast<int>(right_shoulder.y)), cv::Scalar(static_cast<int>(255.0*color_map[0]), static_cast<int>(255.0*color_map[0]), static_cast<int>(255.0*color_map[0])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_shoulder.x), static_cast<int>(left_shoulder.y)), cv::Point(static_cast<int>(left_elbow.x), static_cast<int>(left_elbow.y)), cv::Scalar(static_cast<int>(255.0*color_map[1]), static_cast<int>(255.0*color_map[1]), static_cast<int>(255.0*color_map[1])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_elbow.x), static_cast<int>(left_elbow.y)), cv::Point(static_cast<int>(left_wrist.x), static_cast<int>(left_wrist.y)), cv::Scalar(static_cast<int>(255.0*color_map[2]), static_cast<int>(255.0*color_map[2]), static_cast<int>(255.0*color_map[2])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_shoulder.x), static_cast<int>(right_shoulder.y)), cv::Point(static_cast<int>(right_elbow.x), static_cast<int>(right_elbow.y)), cv::Scalar(static_cast<int>(255.0*color_map[3]), static_cast<int>(255.0*color_map[3]), static_cast<int>(255.0*color_map[3])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_elbow.x), static_cast<int>(right_elbow.y)), cv::Point(static_cast<int>(right_wrist.x), static_cast<int>(right_wrist.y)), cv::Scalar(static_cast<int>(255.0*color_map[4]), static_cast<int>(255.0*color_map[4]), static_cast<int>(255.0*color_map[4])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_shoulder.x), static_cast<int>(left_shoulder.y)), cv::Point(static_cast<int>(left_hip.x), static_cast<int>(left_hip.y)), cv::Scalar(static_cast<int>(255.0*color_map[5]), static_cast<int>(255.0*color_map[5]), static_cast<int>(255.0*color_map[5])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_shoulder.x), static_cast<int>(right_shoulder.y)), cv::Point(static_cast<int>(right_hip.x), static_cast<int>(right_hip.y)), cv::Scalar(static_cast<int>(255.0*color_map[6]), static_cast<int>(255.0*color_map[6]), static_cast<int>(255.0*color_map[6])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_hip.x), static_cast<int>(left_hip.y)), cv::Point(static_cast<int>(right_hip.x), static_cast<int>(right_hip.y)), cv::Scalar(static_cast<int>(255.0*color_map[7]), static_cast<int>(255.0*color_map[7]), static_cast<int>(255.0*color_map[7])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_hip.x), static_cast<int>(left_hip.y)), cv::Point(static_cast<int>(left_knee.x), static_cast<int>(left_knee.y)), cv::Scalar(static_cast<int>(255.0*color_map[8]), static_cast<int>(255.0*color_map[8]), static_cast<int>(255.0*color_map[8])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_hip.x), static_cast<int>(right_hip.y)), cv::Point(static_cast<int>(right_knee.x), static_cast<int>(right_knee.y)), cv::Scalar(static_cast<int>(255.0*color_map[9]), static_cast<int>(255.0*color_map[9]), static_cast<int>(255.0*color_map[9])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_knee.x), static_cast<int>(left_knee.y)), cv::Point(static_cast<int>(left_ankle.x), static_cast<int>(left_ankle.y)), cv::Scalar(static_cast<int>(255.0*color_map[10]), static_cast<int>(255.0*color_map[10]), static_cast<int>(255.0*color_map[10])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_knee.x), static_cast<int>(right_knee.y)), cv::Point(static_cast<int>(right_ankle.x), static_cast<int>(right_ankle.y)), cv::Scalar(static_cast<int>(255.0*color_map[11]), static_cast<int>(255.0*color_map[11]), static_cast<int>(255.0*color_map[11])), 8);
  
}

//void return_pose_segment(){
//
//    // Method: 1) Create a mask
//    Mat1b mask(image.size(), uchar(0));
//    line(mask, A, B, Scalar(255));
//
//   vector<Point> points1;
//    findNonZero(mask, points1);
//
//    // Method: 2) Use LineIterator
//    LineIterator lit(image, A, B);
//
//    vector<Point> points2;
//    points2.reserve(lit.count);
//    for (int i = 0; i < lit.count; ++i, ++lit)
//    {
//        points2.push_back(lit.pos());
//    }
//}

tf::Transform read_poses_from_json(std::string camera_name)
{
  tf::Transform worldToCamTransform;
  double translation_x;
  double translation_y;
  double translation_z;
  double rotation_x;
  double rotation_y;
  double rotation_z;
  double rotation_w;

  json pose_config;
  std::string hard_coded_path = "/cfg/poses.json";
  std::cout << "--- detector cfg_callback ---" << std::endl;
  std::string package_path = ros::package::getPath("recognition");
  std::string full_path = package_path + hard_coded_path;
  std::ifstream json_read(full_path);
  json_read >> pose_config;

  //<!-- pc: Jetson-TX2-Ubuntu-18-CUDA-10-NEW -->
  //<!-- sensor: d415 -->
  //<node pkg="tf" type="static_transform_publisher" name="d415_broadcaster"
  //      args="2.18334 0.46989 1.28112 0.522227 0.598362 -0.456386 -0.401191 /world /d415 100" />
  //              tx       ty      tz      rx        ry       rz        rw      
  //double translation_x = pose_config[camera_name]["pose"]["translation"]["x"];
  //double translation_y = pose_config[camera_name]["pose"]["translation"]["y"];
  //double translation_z = pose_config[camera_name]["pose"]["translation"]["z"];
  //double rotation_x = pose_config[camera_name]["pose"]["rotation"]["x"];
  //double rotation_y = pose_config[camera_name]["pose"]["rotation"]["y"];
  //double rotation_z = pose_config[camera_name]["pose"]["rotation"]["z"];
  //double rotation_w = pose_config[camera_name]["pose"]["rotation"]["w"];

  /// cameras: cam0, cam1, cam2
  for (auto& cameras : pose_config.items()){
    if (cameras.key() == camera_name) {
      //pose: pose, other thing, etc. 
      for (auto& pose : cameras.value().items()) {
        if (pose.key() == "pose") {
          //pose_type: translation, rotation
          for (auto& pose_type : pose.value().items()){
            if (pose_type.key() == "translation"){
              // x, y, z 
              for (auto& translations : pose_type.value().items()){
                //x, y, z
                if (translations.key() == "x"){
                  translation_x = translations.value();
                }
                if (translations.key() == "y"){
                  translation_y = translations.value();
                }
                if (translations.key() == "z"){
                  translation_z = translations.value();
                }
              }
            }
            if (pose_type.key() == "rotation"){
              // x, y, z 
              for (auto& rotations : pose_type.value().items()){
                //x, y, z
                if (rotations.key() == "x"){
                  rotation_x = rotations.value();
                }
                if (rotations.key() == "y"){
                  rotation_y = rotations.value();
                }
                if (rotations.key() == "z"){
                  rotation_z = rotations.value();
                }
                if (rotations.key() == "w"){
                  rotation_w = rotations.value();
                }
              }
            }
          }
        }
      }
    }
  }
  std::cout << "translation x: " << translation_x << std::endl;
  worldToCamTransform.setOrigin(tf::Vector3(translation_x, translation_y, translation_z));
  worldToCamTransform.setRotation(tf::Quaternion(rotation_x, rotation_y, rotation_z, rotation_w));

  return worldToCamTransform;
}

/**
 * @brief The TVMPoseNode
 */
class TVMPoseNode {
  private:
    ros::NodeHandle node_;
    std::unique_ptr<PoseFromConfig> tvm_pose_detector;
    // TF listener
    tf::TransformListener tf_listener;
    tf::Transform worldToCamTransform;
    
    // ROS
    dynamic_reconfigure::Server<recognition::GenDetectionConfig> cfg_server;
    ros::ServiceServer camera_info_matrix_server;

    // Publishers
    ros::Publisher detections_pub;
    ros::Publisher skeleton_pub;
    //ros::Publisher image_pub;
    image_transport::Publisher image_pub;

    // Subscribers
    ros::Subscriber rgb_sub;
    ros::Subscriber camera_info_matrix;
    ros::Subscriber detector_sub;

    // Message Filters
    message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub;
    message_filters::Subscriber<sensor_msgs::Image> depth_image_sub;
    message_filters::Subscriber<PointCloudT> cloud_sub;

    // Message Synchronizers 
    typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximatePolicy;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    boost::shared_ptr<ApproximateSync> approximate_sync_;

    //seconday sync??????
    typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, PointCloudT> ImageApproximatePolicy;
    typedef message_filters::Synchronizer<ImageApproximatePolicy> ImageApproximateSync;
    boost::shared_ptr<ImageApproximateSync> image_approximate_sync_;// vars


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
    float max_capable_depth = 10.0; // 6.25 is what the default is;
    /** \brief transforms used for compensating sensor tilt with respect to the ground plane */
    // Initialize transforms to be used to correct sensor tilt to identity matrix:
    Eigen::Affine3f transform, transform_, anti_transform, anti_transform_;
    //transform = transform.Identity();
    //anti_transform = transform.inverse();
    bool estimate_ground_plane = true;
    Eigen::VectorXf ground_coeff;


    // Minimum detection confidence:
    float ground_based_people_detection_min_confidence = -5.0; //-1.75
    // Minimum person height =
    float minimum_person_height = 0.6;
    // Maximum person height =
    float maximum_person_height = 2.3;
    // Max depth range =
    float max_distance = 10.0 ;
    // Point cloud downsampling factor =
    int sampling_factor = 4;
    // Flag stating if classifiers based on RGB image should be used or not =
    bool use_rgb = true;
    // Threshold on image luminance. If luminance is over this threhsold, classifiers on RGB image are also used =
    int minimum_luminance = 0;
    // If true, sensor tilt angle wrt ground plane is compensated to improve people detection =
    bool sensor_tilt_compensation = true ; 
    // Minimum distance between two persons =
    float heads_minimum_distance = 0.3;
    // Voxel size used to downsample the point cloud (lower = detection slower but more precise; higher = detection faster but less precise) =
    float voxel_size = 0.06;
    // Denoising flag. If true, a statistical filter is applied to the point cloud to remove noise =
    bool apply_denoising = true;
    // MeanK for denoising (the higher it is, the stronger is the filtering) =
    int mean_k_denoising = 5;
    // Standard deviation for denoising (the lower it is, the stronger is the filtering) =
    float std_dev_denoising = 0.3;
    open_ptrack::detection::GroundplaneEstimation<PointT> ground_estimator;


   // Initialize transforms to be used to correct sensor tilt to identity matrix:
    //Eigen::Affine3f transform, anti_transform;
    //transform = transform.Identity();
    //anti_transform = transform.inverse();

    /**
     * @brief constructor
     * @param nh node handler
     */
    TVMPoseNode(ros::NodeHandle& nh, std::string sensor_string, json zone_json):
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
          max_capable_depth = master_config["max_capable_depth"];
          std::cout << "n_zones: " << n_zones << std::endl;
          json_found = true;
        }
        catch(const std::exception& e)
        {
          std::cerr << "json master/area not found: "<< e.what() << '\n';
        }
        
        // Publish Messages
        detections_pub = node_.advertise<opt_msgs::DetectionArray>("/objects_detector/detections", 3);
        //objects_detector?
        skeleton_pub = node_.advertise<opt_msgs::SkeletonArrayMsg>("/detector/skeletons", 1);
        //_raw_skeleton_depth_image_pub = nh.advertise<sensor_msgs::Image>
        //      ("/detector/skeletons_depth_image", 1);
        //_raw_skeleton_image_pub = nh.advertise<sensor_msgs::Image>
        //        (_raw_skeleton_image_topic_to_publish, 1);
        //  cv::ellipse(_colored_depth_image,cv::Point(x,y),
        //              cv::Size(_median_search_x_component,_median_search_y_component),0,
        //              0,0,cv::Scalar(0,255,255),5);

        // Subscribe to Messages
        rgb_image_sub.subscribe(node_, sensor_string +"/color/image_rect_color", 1);
        depth_image_sub.subscribe(node_, sensor_string+"/depth/image_rect_raw", 1);
        cloud_sub.subscribe(node_, sensor_string + "/depth_registered/points", 10);
        
        image_pub = it.advertise(sensor_string + "/objects_detector/image", 1);

        // Camera callback for intrinsics matrix update
        camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &TVMPoseNode::camera_info_callback, this);

        //Time sync policies for the subscribers
        //approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(10), rgb_image_sub, depth_image_sub));
        // _1 = rgb_image_sub
        // _2 = depth_image_sub
        // _3 = zone_json or zone_json
        //approximate_sync_->registerCallback(boost::bind(&TVMPoseNode::callback, this, _1, _2, zone_json));


        image_approximate_sync_.reset(new ImageApproximateSync(ImageApproximatePolicy(10), rgb_image_sub, depth_image_sub, cloud_sub));
        image_approximate_sync_->registerCallback(boost::bind(&TVMPoseNode::callback, this, _1, _2, _3, zone_json));


        // create callback config 
        cfg_server.setCallback(boost::bind(&TVMPoseNode::cfg_callback, this, _1, _2));      

        // create object-detector pointer
        //tvm_pose_detector.reset(new YoloTVMGPU256(model_folder_path));
        //tvm_pose_detector.reset(new YoloTVMGPU(model_folder_path));
        // maybe have this in
        // arg one HAS to have / in front of path
        // TODO add that to debugger
        tvm_pose_detector.reset(new PoseFromConfig("/cfg/pose_model.json", "recognition"));
        sensor_name = sensor_string;
        worldToCamTransform = read_poses_from_json(sensor_name);
      
        // 0 == manual
        ground_estimator(1, true);
      }

    void camera_info_callback(const CameraInfo::ConstPtr & msg){
      intrinsics_matrix << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
      _cx = msg->K[2];
      _cy = msg->K[5];
      _constant_x =  1.0f / msg->K[0];
      _constant_y = 1.0f /  msg->K[4];
      camera_info_available_flag = true;
    }

    PointCloudPtr preprocessCloud (PointCloudPtr& input_cloud)
    {
      // Downsample of sampling_factor in every dimension:
      PointCloudPtr cloud_downsampled(new PointCloud);
      PointCloudPtr cloud_denoised(new PointCloud);
      int sampling_factor_ = 4;
      if (sampling_factor_ != 1)
      {
        cloud_downsampled->width = (input_cloud->width)/sampling_factor_;
        cloud_downsampled->height = (input_cloud->height)/sampling_factor_;
        cloud_downsampled->points.resize(cloud_downsampled->height*cloud_downsampled->width);
        cloud_downsampled->is_dense = input_cloud->is_dense;
        cloud_downsampled->header = input_cloud->header;
        for (int j = 0; j < cloud_downsampled->width; j++)
        {
          for (int i = 0; i < cloud_downsampled->height; i++)
          {
            (*cloud_downsampled)(j,i) = (*input_cloud)(sampling_factor_*j,sampling_factor_*i);
          }
        }
      }
      bool apply_denoising_ = true;
      bool isZed_ = false;
      int voxel_size = 0.06;

      if (apply_denoising_)
      {
        // Denoising with statistical filtering:
        pcl::StatisticalOutlierRemoval<PointT> sor;
        if (sampling_factor_ != 1)
          sor.setInputCloud (cloud_downsampled);
        else
          sor.setInputCloud (input_cloud);
        sor.setMeanK (mean_k_denoising_);
        sor.setStddevMulThresh (std_dev_denoising_);
        sor.filter (*cloud_denoised);
      }

      //  // Denoising viewer
      //  int v1(0);
      //  int v2(0);
      //  denoising_viewer_->removeAllPointClouds(v1);
      //  denoising_viewer_->removeAllPointClouds(v2);
      //  denoising_viewer_->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
      //  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(input_cloud);
      //  denoising_viewer_->addPointCloud<PointT> (input_cloud, rgb, "original", v1);
      //  denoising_viewer_->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
      //  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(cloud_denoised);
      //  denoising_viewer_->addPointCloud<PointT> (cloud_denoised, rgb2, "denoised", v2);
      //  denoising_viewer_->spinOnce();

      // Voxel grid filtering:
      PointCloudPtr cloud_filtered(new PointCloud);
      pcl::VoxelGrid<PointT> voxel_grid_filter_object;
      if (apply_denoising_) {
        voxel_grid_filter_object.setInputCloud(cloud_denoised);
      } else {
        if (sampling_factor_ != 1) {
          voxel_grid_filter_object.setInputCloud(cloud_downsampled);
        } else {
          voxel_grid_filter_object.setInputCloud(input_cloud);
        }
      }
      voxel_grid_filter_object.setLeafSize (voxel_size, voxel_size, voxel_size);
      voxel_grid_filter_object.setFilterFieldName("z");
      //if (isZed_)
      //  voxel_grid_filter_object.setFilterLimits(-1 * max_distance, max_distance);
      //else
      voxel_grid_filter_object.setFilterLimits(0.0, max_distance);
      voxel_grid_filter_object.filter (*cloud_filtered);

      return cloud_filtered;
    }

    void set_ground_variables(PointCloudPtr& cloud){
      if (!estimate_ground_plane){
         std::cout << "Ground plane finished already..." << std::endl;
      } else {
        // Ground estimation:
        std::cout << "Ground plane initialization starting..." << std::endl;
        ground_estimator.setInputCloud(cloud);
        //Eigen::VectorXf ground_coeffs = ground_estimator.computeMulticamera(ground_from_extrinsic_calibration, read_ground_from_file,
        //    pointcloud_topic, sampling_factor, voxel_size);
        ground_coeffs = ground_estimator.computeMulticamera(false, false,
                  sensor_string + "/depth_registered/points", 4, 0.06);

        // Point cloud pre-processing (downsampling and filtering):
        PointCloudPtr cloud_filtered(new PointCloud);
        cloud_filtered = preprocessCloud (cloud_);

        // Ground removal and update:
        pcl::IndicesPtr inliers(new std::vector<int>);
        boost::shared_ptr<pcl::SampleConsensusModelPlane<PointT> > ground_model(new pcl::SampleConsensusModelPlane<PointT>(cloud_filtered));
        //if (isZed_)
        //  ground_model->selectWithinDistance(ground_coeffs_, 0.2, *inliers);
        //else
        ground_model->selectWithinDistance(ground_coeffs, voxel_size, *inliers);
        no_ground_cloud_ = PointCloudPtr (new PointCloud);
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*no_ground_cloud_);
        bool debug_flag = false;
        bool sizeCheck = false;
        //if (isZed_) {
        //  if (inliers->size () >= (300 * 0.06 / 0.02 / std::pow (static_cast<double> (sampling_factor_), 2)))
        //    sizeCheck = true;
        //}
        //else {
        if (inliers->size () >= (300 * 0.06 / voxel_size_ / std::pow (static_cast<double> (sampling_factor_), 2))){
            sizeCheck = true;
        }

        if (sizeCheck) {
          ground_model->optimizeModelCoefficients (*inliers, ground_coeffs_, ground_coeffs_);
        }
        //} else {
        //  if (debug_flag)
        //  {
        //    PCL_INFO ("No groundplane update!\n");
        //  }

        // Background Subtraction (optional):
        if (background_subtraction) {
          PointCloudPtr foreground_cloud(new PointCloud);
          for (unsigned int i = 0; i < no_ground_cloud_->points.size(); i++)
          {
            if (not (background_octree_->isVoxelOccupiedAtPoint(no_ground_cloud_->points[i].x, no_ground_cloud_->points[i].y, no_ground_cloud_->points[i].z)))
            {
              foreground_cloud->points.push_back(no_ground_cloud_->points[i]);
            }
          }
          no_ground_cloud_ = foreground_cloud;
        }


        // if (no_ground_cloud_->points.size() > 0)
        // {
          // Euclidean Clustering:
        std::vector<pcl::PointIndices> cluster_indices;
        typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
        tree->setInputCloud(no_ground_cloud_);
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(2 * 0.06);
        ec.setMinClusterSize(min_points_);
        ec.setMaxClusterSize(max_points_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(no_ground_cloud_);
        ec.extract(cluster_indices);

        // Sensor tilt compensation to improve people detection:
        PointCloudPtr no_ground_cloud_rotated(new PointCloud);
        Eigen::VectorXf ground_coeffs_new;
        if(sensor_tilt_compensation)
        {
          // We want to rotate the point cloud so that the ground plane is parallel to the xOz plane of the sensor:
          Eigen::Vector3f input_plane, output_plane;
          input_plane << ground_coeffs_(0), ground_coeffs_(1), ground_coeffs_(2);
          output_plane << 0.0, -1.0, 0.0;

          Eigen::Vector3f axis = input_plane.cross(output_plane);
          float angle = acos( input_plane.dot(output_plane)/ ( input_plane.norm()/output_plane.norm() ) );
          transform_ = Eigen::AngleAxisf(angle, axis);

          // Setting also anti_transform for later
          anti_transform_ = transform_.inverse();
          no_ground_cloud_rotated = rotateCloud(no_ground_cloud_, transform_);
          ground_coeffs_new.resize(4);
          ground_coeffs_new = rotateGround(ground_coeffs_, transform_);
        }
        else
        {
          transform_ = transform_.Identity();
          anti_transform_ = transform_.inverse();
          no_ground_cloud_rotated = no_ground_cloud_;
          ground_coeffs_new = ground_coeffs_;
        }
      }
    }

  private:
    /**
     * @brief callback for camera information that does detection on images
     *  and publishes the detections to specific topics
     * @param rgb_image  the rgb image message
     * @param depth_image  the depth/stereo image message
     * @param zone_json the json that contains the zone information
     */
    //void callback(const sensor_msgs::Image::ConstPtr& rgb_image,
    //              const sensor_msgs::Image::ConstPtr& depth_image,
    //              json zone_json) {

    void callback(const sensor_msgs::Image::ConstPtr& rgb_image,
                  const sensor_msgs::Image::ConstPtr& depth_image,
                  const PointCloudT::ConstPtr& cloud_
                  json zone_json) {


      std::cout << "running algorithm callback" << std::endl;

      if (estimate_ground_plane) {
        set_ground_variables(cloud_)
      }


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
      pose_results* output;
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
      // set skeleton array here not sure which one is correct
      //opt_msgs::SkeletonArrayMsg skeleton_array;
      opt_msgs::SkeletonArrayMsg::Ptr skeleton_array(new opt_msgs::SkeletonArrayMsg);
      skeleton_array->header = rgb_image->header;
      skeleton_array->rgb_header = rgb_image->header;
      // set detection intrinsic matrix from camera variables
      for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
          skeleton_array->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
        }
      }

      //////////////////////////////////////////////////////////////////////////////////////////////
      // Create XYZ cloud:
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      // fails?
      //pcl::fromROSMsg(*cloud_, *pcl_cloud);
      //https://answers.ros.org/question/9515/how-to-convert-between-different-point-cloud-types-using-pcl/
      pcl::PointXYZRGB xyzrgb_point;
      cloud_xyzrgb->points.resize(cloud_->width * cloud_->height, xyzrgb_point);
      cloud_xyzrgb->width = cloud_->width;
      cloud_xyzrgb->height = cloud_->height;
      cloud_xyzrgb->is_dense = false;

      int cloud__is_empty = cloud_->size();
      std::cout << "DEBUG: cloud__is_empty: " << cloud__is_empty << std::endl;
      int cloud_width = cloud_->width;
      std::cout << "DEBUG: cloud_width: " << cloud_width << std::endl;
      int cloud_height = cloud_->height;
      std::cout << "DEBUG: cloud_height: " << cloud_height << std::endl;

      // fill xyzrgb
      for (int i=0;i<cloud_->height;i++)
      {
          for (int j=0;j<cloud_->width;j++)
          {
          cloud_xyzrgb->at(j,i).x = cloud_->at(j,i).x;
          cloud_xyzrgb->at(j,i).y = cloud_->at(j,i).y;
          cloud_xyzrgb->at(j,i).z = cloud_->at(j,i).z;
          }
      }

      int cloud_xyzrgb_is_empty = cloud_xyzrgb->size();
      std::cout << "DEBUG: cloud_xyzrgb_is_empty: " << cloud_xyzrgb_is_empty << std::endl;

      pcl::PointXYZ xyz_point;
      pcl_cloud->points.resize(cloud_->width * cloud_->height, xyz_point);
      pcl_cloud->width = cloud_->width;
      pcl_cloud->height = cloud_->height;
      pcl_cloud->is_dense = false;

      for (size_t i = 0; i < cloud_->points.size(); i++) {
          pcl_cloud->points[i].x = cloud_->points[i].x;
          pcl_cloud->points[i].y = cloud_->points[i].y;
          pcl_cloud->points[i].z = cloud_->points[i].z;
      }


      // define xyz 3d points in cloud
      Eigen::MatrixXd points_3d_in_cam(3, pcl_cloud->size());
      for(int i = 0; i < pcl_cloud->size(); i++)
      {
          points_3d_in_cam(0, i) = (*pcl_cloud)[i].x;
          points_3d_in_cam(1, i) = (*pcl_cloud)[i].y;
          points_3d_in_cam(2, i) = (*pcl_cloud)[i].z;
      }    


      cv::Mat curr_image (cloud_xyzrgb->height, cloud_xyzrgb->width, CV_8UC3);
      for (int i=0;i<cloud_->height;i++)
      {
          for (int j=0;j<cloud_->width;j++)
          {
          curr_image.at<cv::Vec3b>(i,j)[2] = cloud_->at(j,i).r;
          curr_image.at<cv::Vec3b>(i,j)[1] = cloud_->at(j,i).g;
          curr_image.at<cv::Vec3b>(i,j)[0] = cloud_->at(j,i).b;
          }
      }
    
      // define this, but maybe do like the camera transform here????
      Eigen::MatrixXd points_2d_homo = cam_intrins_ * points_3d_in_cam;

      // lets assume that points_2d_homo == world transform...

      Eigen::MatrixXd points_2d(2, pcl_cloud->size());
      for(int i = 0; i < pcl_cloud->size(); i++)
      {
          points_2d(0, i) = points_2d_homo(0, i) / points_2d_homo(2, i);
          points_2d(1, i) = points_2d_homo(1, i) / points_2d_homo(2, i);
      }

      //////////////////////////////////////////////////////////////////////////////////////////////

      open_ptrack::opt_utils::Conversions converter; 
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

      // override and use pointcloud
      cv_image = curr_image.clone();
      cv_image_clone = cv_image.clone();

      // EXTRACT POINTCLOUD-DEPTH HERE...



      Eigen::MatrixXd points_2d(2, pcl_cloud->size());
      for(int i = 0; i < pcl_cloud->size(); i++)
      {
          points_2d(0, i) = points_2d_homo(0, i) / points_2d_homo(2, i);
          points_2d(1, i) = points_2d_homo(1, i) / points_2d_homo(2, i);
      }


      // necessary? or can we just use height/width of cv_image
      int DISPLAY_RESOLUTION_HEIGHT = image_size.height;
      int DISPLAY_RESOLUTION_WIDTH = image_size.width;

      std::cout << "running yolo" << std::endl;
      // forward inference of object detector
      begin = ros::Time::now();
      output = tvm_pose_detector->forward_full(cv_image, .3);
      duration = ros::Time::now().toSec() - begin.toSec();
      printf("yolo detection time: %f\n", duration);
      printf("yolo detections: %ld\n", output->num);
      //std::array gluon_to_rtpose[17] = {0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10, 100};
      // ears/eyes are ignored in rtpose, but not in gluon
      //index == gluon
      //value == rtpose's index
      // 1 == neck, thus not in gluon
      // 14 == chest, thus not in gluon
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
          float score = output->boxes[i].score;
          std::vector<cv::Point3f> points = output->boxes[i].points;
          int num_parts = points.size();

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
          float median_depth = cv_depth_image.at<float>(ymax, median_x) / mm_factor;


          if (median_depth <= 0 || median_depth > max_capable_depth) {
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
              Point3f middle;
              Point3f world_to_temp;
              middle.x = cloud_->at(median_x,median_y).x
              middle.y = cloud_->at(median_x,median_y).y
              middle.z = cloud_->at(median_x,median_y).z
             
              // head
              Point3f top;
              //top.x = cloud_->at(new_x,median_y).x
              //top.y = cloud_->at(new_x,median_y).y
              //top.z = cloud_->at(new_x,median_y).z
              cv::Point3f head = points[0];
              int top_cast_x = static_cast<int>(head.x);
              int top_cast_y = static_cast<int>(head.y);   
              top.x = cloud_->at(top_cast_x,top_cast_y).x
              top.y = cloud_->at(top_cast_x,top_cast_y).y
              top.z = cloud_->at(top_cast_x,top_cast_y).z
              float head_z = cv_depth_image.at<float>(top_cast_y, top_cast_x) / mm_factor;
           
              // just bottom of box should be ok?
              // could just do feet / 2
              Point3f bottom;
              bottom.x = cloud_->at(median_x,new_y).x
              bottom.y = cloud_->at(median_x,new_y).y
              bottom.z = cloud_->at(median_x,new_y).z

              Eigen::Vector3f centroid3d = anti_transform * middle;
              Eigen::Vector3f centroid2d = converter.world2cam(centroid3d, intrinsics_matrix);

              Eigen::Vector3f top3d = anti_transform * top;
              Eigen::Vector3f top2d = converter.world2cam(top3d, intrinsics_matrix);
              // theoretical person bottom point:
              Eigen::Vector3f bottom3d = anti_transform * bottom;
              Eigen::Vector3f bottom2d = converter.world2cam(bottom3d, intrinsics_matrix);

              world_to_temp.x =  static_cast<float>(tmp.x);
              world_to_temp.y =  static_cast<float>(tmp.y);
              world_to_temp.z =  static_cast<float>(tmp.z);

              tf::Vector3 current_world_point(world_to_temp.x, world_to_temp.y, world_to_temp.z);

              float enlarge_factor = 1.1;
              float pixel_xc = centroid2d(0);
              float pixel_yc = centroid2d(1);
              float pixel_height = (bottom2d(1) - top2d(1)) * enlarge_factor;
              float pixel_width = pixel_height / 2;
              detection_msg.box_2D.x = int(centroid2d(0) - pixel_width/2.0);
              detection_msg.box_2D.y = int(centroid2d(1) - pixel_height/2.0);
              detection_msg.box_2D.width = int(pixel_width);
              detection_msg.box_2D.height = int(pixel_height);

              // get height
              float sqrt_ground_coeffs = (ground_coeffs - Eigen::Vector4f(0.0f, 0.0f, 0.0f, ground_coeffs(3))).norm();
              Eigen::Vector4f height_point;
              height_point << top.x, top.y_, top.z_, 1.0f;
              float height = std::fabs(height_point.dot(ground_coeffs));
              height /= sqrt_ground_coeffs;
              //height_ = height;
              distance_ = std::sqrt(top.x * top.x + c_z_ * c_z_);
              
              detection_msg.height = height;
              detection_msg.confidence = score;
              detection_msg.distance = head_z;
              converter.Vector3fToVector3((1+head_centroid_compensation/centroid3d.norm())*centroid3d, detection_msg.centroid);
              converter.Vector3fToVector3((1+head_centroid_compensation/top3d.norm())*top3d, detection_msg.top);
              converter.Vector3fToVector3((1+head_centroid_compensation/bottom3d.norm())*bottom3d, detection_msg.bottom);


              detection_msg.box_3D.p1.x = mx;
              detection_msg.box_3D.p1.y = my;
              detection_msg.box_3D.p1.z = median_depth;
              

              //detection_msg.box_2D.x = median_x;
              //detection_msg.box_2D.y = median_y;
              //detection_msg.box_2D.width = 0;
              //detection_msg.box_2D.height = 0;
              //detection_msg.height = 0;
              //detection_msg.confidence = 10;
              //detection_msg.distance = median_depth;
              
              //detection_msg.centroid.x = mx;
              //detection_msg.centroid.y = my;
              //detection_msg.centroid.z = median_depth;
              
              //detection_msg.top.x = 0;
              //detection_msg.top.y = 0;
              //detection_msg.top.z = 0;
              
              //detection_msg.bottom.x = 0;
              //detection_msg.bottom.y = 0;
              //detection_msg.bottom.z = 0;
              
              // DO POSE HERE
              // do rtpose skelaton detection message here...
              opt_msgs::SkeletonMsg skeleton;
              // we might have to change this part
              skeleton.skeleton_type = opt_msgs::SkeletonMsg::COCO;
              skeleton.joints.resize(num_parts);

              for (size_t i = 0; i < num_parts; i++){
                /* code */
                opt_msgs::Joint3DMsg joint3D;
                int rtpose_part_index = gluon_to_rtpose[i];
          
                // IGNORE eyes/ears
                if (rtpose_part_index == -1){
                  continue;
                } else {
                  cv::Point3f point = points[i];
                  float confidence = point.z;
                  int cast_x = static_cast<int>(point.x);
                  int cast_y = static_cast<int>(point.y);
                  float z = cv_depth_image.at<float>(cast_y, cast_x) / mm_factor;
                  joint3D.x = point.x;
                  joint3D.y = point.y;
                  joint3D.z = z;
                  joint3D.max_height = DISPLAY_RESOLUTION_HEIGHT;
                  joint3D.max_width = DISPLAY_RESOLUTION_WIDTH;
                  joint3D.confidence = confidence;
                  joint3D.header = rgb_image->header;
                  skeleton.joints[rtpose_part_index] = joint3D;
                  // debug this 
                  //cv::circle(cv_image_clone, cv::Point(cast_x, cast_y), 3, (0,255,0));
                }
              }
              float confidence = 0.9f;
              cv::Point3f point_left_shoulder = points[5];
              cv::Point3f point_right_shoulder = points[6];
              cv::Point3f point_left_hip = points[11];
              cv::Point3f point_right_hip = points[12];

              // ******* NECK == joint location 1
              opt_msgs::Joint3DMsg joint3D_neck;
              // center of each shoulder == chest
              float x = (point_left_shoulder.x + point_right_shoulder.x) / 2;
              float y = (point_left_shoulder.y + point_right_shoulder.y) / 2;
              int cast_point_x = static_cast<int>(x);
              int cast_point_y = static_cast<int>(y);              
              float z = cv_depth_image.at<float>(cast_point_y, cast_point_x) / mm_factor;
              joint3D_neck.x = x;
              joint3D_neck.y = y;
              joint3D_neck.z = z;
              joint3D_neck.confidence = confidence;
              joint3D_neck.header = rgb_image->header;
              joint3D_neck.max_height = DISPLAY_RESOLUTION_HEIGHT;
              joint3D_neck.max_width = DISPLAY_RESOLUTION_WIDTH;              
              // NECK == joint location 1
              skeleton.joints[1] = joint3D_neck;
              //cv::circle(cv_image_clone, cv::Point(cast_point_x, cast_point_y), 3, (0,255,0));
              
              // ******** CHEST
              opt_msgs::Joint3DMsg joint3D_chest;
              // weighted mean from rtpose
              // TODO if this looks ugly, we'll just use the neck
              float cx = (point_left_hip.x + point_right_hip.x) * 0.4 + (point_left_shoulder.x + point_right_shoulder.x) * 0.1;
              float cy = (point_left_hip.y + point_right_hip.y) * 0.4 + (point_left_shoulder.y + point_right_shoulder.y) * 0.1;
              int cast_cx = static_cast<int>(cx);
              int cast_cy = static_cast<int>(cy);
              float cz = cv_depth_image.at<float>(cast_cy, cast_cx) / mm_factor;
              joint3D_chest.x = cx;
              joint3D_chest.y = cy;
              joint3D_chest.z = cz;
              joint3D_chest.confidence = confidence; //use confidence from previous
              joint3D_chest.header = rgb_image->header;
              joint3D_chest.max_height = DISPLAY_RESOLUTION_HEIGHT;
              joint3D_chest.max_width = DISPLAY_RESOLUTION_WIDTH; 
              // CHEST == joint location 15, index 14
              skeleton.joints[14] = joint3D_chest;
              //cv::circle(cv_image_clone, cv::Point(cast_cx, cast_cy), 3, (0,0,255));
              draw_skelaton(cv_image_clone, points);
              //index == gluon
              //value == rtpose
              //chest == 100 == LSHOULDER / RSHOULDER == 5 / 2
              //std::array gluon_to_rtpose[17] = 
              //'nose', == 0. Nose, 0
              //'left_eye', 1. LEye == 15
              //'right_eye', == 14
              //'left_ear', == 17
              //'right_ear', == 16
              //'left_shoulder', == LSHOULDER, 5
              //'right_shoulder', = RSHOULDER, 2
              //'left_elbow', == LELBOW, 6
              //'right_elbow', == RELBOW, 3
              //'left_wrist', == LWRIST, 7
              //'right_wrist', == RWRIST, 4
              //'left_hip', == LHIP, 11
              //'right_hip', == RHIP, 8
              //'left_knee', == LKNEE, 12
              //'right_knee', == RKNEE, 9
              //'left_ankle', == LANKLE, 13
              //'right_ankle' == RANKLE, 10
              //  nothing      == CHEST ? left/right shoulders // 2?????

              //modeldesignerfactory.cpp
              //{{0,  "Nose"},
              // {1,  "Neck"},
              // {2,  "RShoulder"},
              // {3,  "RElbow"},
              // {4,  "RWrist"},
              // {5,  "LShoulder"},
              // {6,  "LElbow"},
              // {7,  "LWrist"},
              // {8,  "RHip"},
              // {9,  "RKnee"},
              // {10, "RAnkle"},
              // {11, "LHip"},
              // {12, "LKnee"},
              // {13, "LAnkle"},
              // {14, "REye"},
              // {15, "LEye"},
              // {16, "REar"},
              // {17, "LEar"},
              // {18, "Bkg"}},  

              //'nose', == 0. HEAD, 0
              //'left_eye', -1
              //'right_eye', -1
              //'left_ear', -1
              //'right_ear', -1
              //'left_shoulder', == LSHOULDER, 5
              //'right_shoulder', = RSHOULDER, 2
              //'left_elbow', == LELBOW, 6
              //'right_elbow', == RELBOW, 3
              //'left_wrist', == LWRIST, 7
              //'right_wrist', == RWRIST, 4
              //'left_hip', == LHIP, 11
              //'right_hip', == RHIP, 8
              //'left_knee', == LKNEE, 12
              //'right_knee', == RKNEE, 9
              //'left_ankle', == LANKLE, 13
              //'right_ankle' == RANKLE, 10
              // nothing      == 100 ? left/right shoulders // 2?????

              // opt keypoints
              //enum SkeletonJointId
              //{
              //  HEAD = 0,
              //  NECK,
              //  RSHOULDER,
              //  RELBOW,
              //  RWRIST,
              //  LSHOULDER,
              //  LELBOW,
              //  LWRIST,
              //  RHIP,
              //  RKNEE,
              //  RANKLE,
              //  LHIP,
              //  LKNEE,
              //  LANKLE,
              //  CHEST
              //} ;

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
              skeleton.confidence = 100;
              skeleton.height = 0.0;
              skeleton.distance = 0.0;
              skeleton.occluded = false;
              skeleton_array->skeletons.push_back(skeleton);
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
    skeleton_pub.publish(skeleton_array);
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
  //json master_config;
  //std::string package_path = ros::package::getPath("recognition");
  //std::string master_hard_coded_path = package_path + "/cfg/master.json";
  //std::ifstream json_read(master_hard_coded_path);
  //json_read >> master_config;
  //sensor_name = master_config["sensor_name"]; //the path to the detector model file

  json zone_json;
  std::string area_package_path = ros::package::getPath("recognition");
  std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
  std::ifstream area_json_read(area_hard_coded_path);
  area_json_read >> zone_json;

  std::cout << "--- tvm_detection_node ---" << std::endl;
  ros::init(argc, argv, "tvm_detection_node");
  // something is off here... with the private namespace
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;
  pnh.param("sensor_name", sensor_name, std::string("d435"));
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  TVMPoseNode node(nh, sensor_name, zone_json);
  std::cout << "detection node init " << std::endl;
  ros::spin();
  return 0;
}



