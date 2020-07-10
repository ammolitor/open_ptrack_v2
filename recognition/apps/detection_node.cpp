#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <random> 
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
#include <dynamic_reconfigure/server.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/console/time.h>
#include <pcl/filters/passthrough.h>
#include <open_ptrack/person_clustering/person_cluster.h>
#include <open_ptrack/person_clustering/head_based_subclustering.h>
#include <open_ptrack/ground_segmentation/ground_segmentation.h>
#include <open_ptrack/opt_utils/conversions.h>
#include <opt_msgs/RoiRect.h>
#include <opt_msgs/Rois.h>
#include <std_msgs/String.h>
#include <sensor_msgs/CameraInfo.h>
#include <opt_msgs/Detection.h>
#include <opt_msgs/DetectionArray.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <open_ptrack/hungarian/Hungarian.h>
#include <nlohmann/json.hpp>


/// yolo specific args
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <open_ptrack/nms/nms.h>
#include <torch/torch.h>

using json = nlohmann::json;
typedef sensor_msgs::Image Image;
typedef sensor_msgs::CameraInfo CameraInfo;
// adding this 
using namespace message_filters::sync_policies;
using namespace std;
using namespace cv; // https://github.com/opencv/opencv/issues/6661
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef boost::shared_ptr<PointCloud> PointCloudPtr;
typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;


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
            thresh = model_config["thresh"];

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
            std::cout << "model loaded" << std::endl;
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
            //std::cout << "width: " << detector_width << std::endl;
            //std::cout << "height: " << detector_height << std::endl;
            //std::cout << "total_input: " << total_input << std::endl;
            //std::cout << "device_id: " << device_id << std::endl;
            //std::cout << "dtype_code: " << dtype_code << std::endl;
            //std::cout << "dtype_bits: " << dtype_bits << std::endl;
            //std::cout << "dtype_lanes: " << dtype_lanes << std::endl;
            //std::cout << "device_type: " << device_type << std::endl;

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
            //std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(frame, detector_width, detector_height, true);
            //std::cout << "preprocess finished" << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            //std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) detector_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            //std::cout << "run/getoutput/setinput finished" << std::endl;
  
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
            ////std::cout << "TVMSynchronize finished" << std::endl;  


            // copy to output
            //ulsMatF(int cols, int rows, int channels)
            //at(int channel, int row, int c
            MatF yolo_output(6, n_dets, 1); //ulsMatF yolo_output(1, n_dets, 6);
            TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
            ////std::cout << "TVMSynchronize finished" << std::endl;  
            
            ////std::cout << "starting nms" << std::endl;
            //auto tick = Clock::now();
            std::vector<sortable_result> tvm_results;
            std::vector<sortable_result> proposals;
            proposals.clear();
            tvm_nms_cpu(proposals, yolo_output, thresh, thresh, tvm_results);
            //std::cout << "ending nms" << std::endl;

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
            ////std::cout << "torch part finished" << std::endl; 

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
            // free proposals
            tvm_results = std::vector<sortable_result>();
            proposals = std::vector<sortable_result>();
            return results;
        }

        std::vector<cv::Point3f> pose_forward(cv::Mat bbox_mask, float xmin, float ymin, float xmax, float ymax)
        {
            //std::cout << "running pose forward" << std::endl;
            // get height/width dynamically
            cv::Size image_size = bbox_mask.size();
            float img_height = static_cast<float>(image_size.height);
            float img_width = static_cast<float>(image_size.width);
            int64_t in_shape[4] = {1, 3, pose_height, pose_width};
            int total_input = 3 * pose_width * pose_height;
            
            DLTensor *output_tensor_heatmap;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            //std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(tvm_heatmap_size, pose_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_tensor_heatmap);
            //std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            //std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(bbox_mask, pose_width, pose_height, true);
            cv::Size processed_image_size = processed_image.size();
            //std::cout << "preprocess finished: " << std::endl;
            //std::cout << "preprocess height: " << processed_image_size.height << std::endl;
            //std::cout << "preprocess width: " << processed_image_size.width << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            //std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) pose_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            //std::cout << "run/getoutput/setinput finished" << std::endl;
  
            // https://github.com/apache/incubator-tvm/issues/979?from=timeline
            //"This may give you some ideas to start with.
            //In general you want to use pinned memory and you want
            //to interleave computation with copying; so you want to
            // be upload the next thing while you are computing the
            //current thing while you are downloading the last thing."
            TVMSynchronize(device_type, device_id, nullptr);
            get_output(0, output_tensor_heatmap);
            //std::cout << "TVMSynchronize finished" << std::endl;  

            torch::Tensor ndarray_heat_map_full = torch::zeros({1, 17, 64, 48}, at::kFloat);

            TVMArrayCopyToBytes(output_tensor_heatmap, ndarray_heat_map_full.data_ptr(), 1*17*64*48 * sizeof(float));
            ////std::cout << "saving array output " << std::endl;
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
            ////std::cout << "ndarray_heat_map reshape finished: " << ndarray_heat_map.sizes().size() << std::endl;
            
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L173
            // idx = nd.argmax(heatmaps_reshaped, 2)
            torch::Tensor idx = torch::argmax(ndarray_heat_map, 1);
            //std::vector<int64_t> idxsize = idx.sizes().size();
            ////std::cout << "argmax finished: " << idx.sizes().size() << std::endl;
            
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
            ////std::cout << "pose_forward w: " << w << std::endl;
            ////std::cout << "pose_forward h: " << h << std::endl;
            ////std::cout << "pose_forward center_x: " << center_x << std::endl;
            ////std::cout << "pose_forward center_y: " << center_y << std::endl;
            // https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L168
            // might have to use a diff var name
            for (size_t i = 0; i < 17; i++){
              float index = idx_accessor[i];
              ////std::cout << "index: " << index << std::endl;
              float probability = heat_map_accessor[i][static_cast<int>(index)];
              ////std::cout << "probability: " << probability << std::endl;
              
              //// python modulo vs c++ is dfff
              ////https://stackoverflow.com/questions/1907565/c-and-python-different-behaviour-of-the-modulo-operation
              //preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)
              //preds[:, :, 0] = (preds[:, :, 0]) % width
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L181
              // float modulo_pred = ((index % heatmap_width) + heatmap_width) % heatmap_width;
              // float floor_pred = std::floor(index / heatmap_width);
              int modulo_int = static_cast<int>(index) % static_cast<int>(heatmap_width);
              ////std::cout << "modulo_int: " << modulo_int << std::endl;
              float modulo_pred = static_cast<float>(modulo_int);
              ////std::cout << "modulo_pred: " << modulo_pred << std::endl;
              float floor = index / heatmap_width;
              ////std::cout << "floor: " << floor << std::endl;
              float floor_pred = std::floor(floor);
              ////std::cout << "floor_pred: " << floor_pred << std::endl;
              if (probability <= 0.0) {
                // zero out the pred if the prob is bad...
                //pred_mask = nd.tile(nd.greater(maxvals, 0.0), (1, 1, 2))
                //pred_mask = pred_mask.astype(np.float32)
                //preds *= pred_mask
                modulo_pred = 0.0f;
                floor_pred = 0.0f;
              }
              ////std::cout << "modulo_pred_end: " << modulo_pred << std::endl;
              ////std::cout << "floor_pred_end: " << floor_pred << std::endl;
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L289-L290
              float w_ratio = modulo_pred / heatmap_width;
              float h_ratio = floor_pred / heatmap_height;
              ////std::cout << "w_ratio: " << w_ratio << std::endl;
              ////std::cout << "h_ratio: " << h_ratio << std::endl;              
              cv::Point3f point;
              //https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/pose.py#L291-L292
              //scale = np.array([w, h])
              //preds[i][:, 0] = scale[0] * 2 * w_ratio + center[0] - scale[0]
              //center = np.array([x0 + w, y0 + h])
              point.x = w * 2.0f * w_ratio + center_x - w;
              point.y = h * 2.0f * h_ratio + center_y - h;
              point.z = probability;
              ////std::cout << "point.x: " << point.x << std::endl;
              ////std::cout << "point.y: " << point.y << std::endl;
              points.push_back(point);
            }
            // free outputs
            TVMArrayFree(input);
            TVMArrayFree(output_tensor_heatmap);
            input = nullptr;
            output_tensor_heatmap = nullptr;
            free(data_x);
            data_x = nullptr;
            ////std::cout << "freeing finished" << std::endl;
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
            thresh = model_config["thresh"];
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
            //std::cout << "starting function" << std::endl;
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
            //std::cout << "width: " << detector_width << std::endl;
            //std::cout << "height: " << detector_height << std::endl;
            //std::cout << "total_input: " << total_input << std::endl;
            //std::cout << "device_id: " << device_id << std::endl;
            //std::cout << "dtype_code: " << dtype_code << std::endl;
            //std::cout << "dtype_bits: " << dtype_bits << std::endl;
            //std::cout << "dtype_lanes: " << dtype_lanes << std::endl;
            //std::cout << "device_type: " << device_type << std::endl;

            DLTensor *output_for_nms;
            DLTensor *input;
            float *data_x = (float *) malloc(total_input * sizeof(float));
           
            // allocate memory for results
            yoloresults* results = (yoloresults*)calloc(1, sizeof(yoloresults));
            results->num = 100;
            results->boxes = (bbox_result*)calloc(100, sizeof(bbox_result));

            //std::cout << "about to allocate info" << std::endl;
            // allocate DLTensor memory on device for all the vars needed
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayAlloc(no_nms_output_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_for_nms);
            //std::cout << "allocate info finished" << std::endl;

            //copy processed image to DLTensor
            //std::cout << "about to preprocess" << std::endl;
            cv::Mat processed_image = preprocess_image(frame, detector_width, detector_height, true);
            //std::cout << "preprocess finished" << std::endl;
            cv::Mat split_mat[3];
            cv::split(processed_image, split_mat);
            memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                   processed_image.cols * processed_image.rows * sizeof(float));
            TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
            //std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           
 
            // standard tvm module run
            // get the module, set the module-input, and run the function
            // this is symbolic it ISNT run until TVMSync is performed
            tvm::runtime::Module *mod = (tvm::runtime::Module *) detector_handle.get();
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            //std::cout << "run/getoutput/setinput finished" << std::endl;
  
            // https://github.com/apache/incubator-tvm/issues/979?from=timeline
            //"This may give you some ideas to start with.
            //In general you want to use pinned memory and you want
            //to interleave computation with copying; so you want to
            // be upload the next thing while you are computing the
            //current thing while you are downloading the last thing."
            TVMSynchronize(device_type, device_id, nullptr);
            get_output(0, output_for_nms);
            //std::cout << "TVMSynchronize finished" << std::endl;  

            // copy to output
            //ulsMatF(int cols, int rows, int channels)
            //at(int channel, int row, int c
            MatF yolo_output(6, n_dets, 1); //ulsMatF yolo_output(6, n_dets, 1);
            TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
            //TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
            //std::cout << "TVMSynchronize finished" << std::endl;  
            
            //std::cout << "starting nms" << std::endl;
            //auto tick = Clock::now();
            std::vector<sortable_result> tvm_results;
            std::vector<sortable_result> proposals;
            proposals.clear();
            tvm_nms_cpu(proposals, yolo_output, thresh, thresh, tvm_results);
            //std::cout << "ending nms" << std::endl;

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
            //std::cout << "torch array iter finished" << std::endl;
            tvm_results = std::vector<sortable_result>();
            proposals = std::vector<sortable_result>();           
            return results;
        }
};










/** \brief BaseNode estimates the ground plane equation from a 3D point cloud */
class TVMNode {
  private:
    std::unique_ptr<NoNMSPoseFromConfig> tvm_pose_detector;
    std::unique_ptr<NoNMSYoloFromConfig> tvm_standard_detector;
    // Publishers
    ros::Publisher detections_pub;
    ros::Publisher skeleton_pub;
    image_transport::Publisher image_pub;
    ros::NodeHandle node_;
    ros::Subscriber point_cloud_approximate_sync_;
    image_transport::ImageTransport it;
    ros::ServiceServer camera_info_matrix_server;
    ros::Subscriber camera_info_matrix;

  public:
    tf::TransformListener tf_listener;
    tf::Transform worldToCamTransform;

    typedef pcl::PointCloud<PointT> PointCloud;
    typedef boost::shared_ptr<PointCloud> PointCloudPtr;
    typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

    //############################
    //## camera information ##
    //############################
    Eigen::Matrix3d cam_intrins_;
    Eigen::Matrix3f intrinsics_matrix;
    bool camera_info_available_flag = false;
    double _cx;
    double _cy;
    double _constant_x;
    double _constant_y;
    float max_capable_depth = 6.25; // 6.25 is what the default is;
    std::string sensor_name;

    //############################
    //## zone information ##
    //############################
    json zone_json;
    json master_json;
    int n_zones;
    bool json_found = false;

    //############################
    //## Background subtraction ##
    //############################
    //bool background_subtraction = true;
    int ground_estimation_mode = 2; // automatic
    //# Flag enabling manual grond selection via ssh:
    bool remote_ground_selection = true; //#false
    //# Flag stating if the ground should be read from file, if present:
    bool read_ground_from_file = true;
    //# Flag that locks the ground plane update:
    bool lock_ground = true;
    //# Threshold on the ratio of valid points needed for ground estimation:
    bool valid_points_threshold = 0.0;
    //# Flag enabling/disabling background subtraction:
    bool background_subtraction = true;// #false
    //# Resolution of the octree representing the background:
    float background_octree_resolution =  0.3;
    //# Seconds to use to lear n the background:
    float background_seconds = 3.0;
    // Minimum detection confidence:
    float ground_based_people_detection_min_confidence = -5.0; //-1.75
    // Minimum person height 
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
    bool sensor_tilt_compensation = true; 
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
    double rate_value = 1.0;

    //###################################
    //## Ground + Clustering Variables ##
    //###################################
    /** \brief transforms used for compensating sensor tilt with respect to the ground plane */
    // Initialize transforms to be used to correct sensor tilt to identity matrix:
    Eigen::Affine3f transform, transform_, anti_transform, anti_transform_;
    //Eigen::Affine3f transform, anti_transform;
    bool estimate_ground_plane = true;
    Eigen::VectorXf ground_coeffs;
    Eigen::VectorXf ground_coeffs_new;
    open_ptrack::ground_segmentation::GroundplaneEstimation<PointT> ground_estimator = open_ptrack::ground_segmentation::GroundplaneEstimation<PointT>(ground_estimation_mode, remote_ground_selection);
    pcl::octree::OctreePointCloud<PointT> *background_octree_;
    PointCloudPtr no_ground_cloud_ = PointCloudPtr (new PointCloud);
    PointCloudPtr no_ground_cloud_rotated = PointCloudPtr (new PointCloud);
    PointCloudT::Ptr background_cloud;
    std::vector<pcl::PointIndices> cluster_indices;
    float min_height_ = 1.3;
    float max_height_ = 2.3;
    float heads_minimum_distance_ = 0.3;
    bool vertical_ = false;
    bool use_rgb_ = true;
    int n_frame = 0;
    int n_frames = 15;
    bool setbackground = true;
    float sqrt_ground_coeffs;
    std::vector<cv::Point2f> cluster_centroids2d;
    std::vector<cv::Point3f> cluster_centroids3d;
    std::vector<cv::Point2f> yolo_centroids2d;
    std::vector<cv::Point3f> yolo_centroids3d;
    pcl::PointCloud<pcl::RGB>::Ptr rgb_image_;
    float mm_factor = 1000.0f;
    float median_factor = 0.1;
    bool use_headclusters = false;
    //###################################
    //## Detection Variables ##
    //###################################
    float thresh = 0.3f;
    int gluon_to_rtpose[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
    bool use_pose_model = false;
    //###################################
    //## Transform Listeners ##
    //###################################
    Eigen::Affine3d world2rgb;
    tf::StampedTransform world2rgb_transform;
    tf::StampedTransform world_transform;
    tf::StampedTransform world_inverse_transform;

    TVMNode(ros::NodeHandle& nh, std::string sensor_string, json zone, double max_distance):
      node_(nh), it(node_)
      {
        try
        {
          json master_config;
          std::string master_package_path = ros::package::getPath("recognition");
          std::string master_hard_coded_path = master_package_path + "/cfg/master.json";
          std::ifstream master_json_read(master_hard_coded_path);
          master_json_read >> master_config;
          n_zones = master_config["n_zones"]; //the path to the detector model file
          max_capable_depth = master_config["max_capable_depth"];
          std::cout << "max_capable_depth: " << max_capable_depth << std::endl;
          use_headclusters = master_config["use_headclusters"];
          std::cout << "use_headclusters: " << use_headclusters << std::endl;
          use_pose_model = master_config["use_pose_model"];
          std::cout << "use_pose_model: " << use_pose_model << std::endl;
          ground_estimation_mode = master_config["ground_estimation_mode"];
          std::cout << "ground_estimation_mode: " << ground_estimation_mode << std::endl;
          remote_ground_selection = master_config["remote_ground_selection"];
          std::cout << "remote_ground_selection: " << remote_ground_selection << std::endl;
          read_ground_from_file = master_config["read_ground_from_file"];
          std::cout << "read_ground_from_file: " << read_ground_from_file << std::endl; 
          lock_ground = master_config["lock_ground"]; 
          std::cout << "lock_ground: " << lock_ground << std::endl;
          valid_points_threshold = master_config["valid_points_threshold"];
          std::cout << "valid_points_threshold: " << valid_points_threshold << std::endl;
          background_subtraction = master_config["background_subtraction"];
          std::cout << "background_subtraction: " << background_subtraction << std::endl; 
          background_octree_resolution = master_config["background_octree_resolution"];
          std::cout << "background_octree_resolution: " << background_octree_resolution << std::endl;  
          background_seconds = master_config["background_seconds"]; 
          std::cout << "background_seconds: " << background_seconds << std::endl;
          ground_based_people_detection_min_confidence = master_config["ground_based_people_detection_min_confidence"]; 
          std::cout << "ground_based_people_detection_min_confidence: " << ground_based_people_detection_min_confidence << std::endl;
          minimum_person_height = master_config["minimum_person_height"];
          std::cout << "minimum_person_height: " << minimum_person_height << std::endl; 
          maximum_person_height = master_config["maximum_person_height"];
          std::cout << "maximum_person_height: " << maximum_person_height << std::endl; 
          max_distance = master_config["max_distance"]; 
          std::cout << "max_distance: " << max_distance << std::endl;
          sampling_factor = master_config["sampling_factor"];
          std::cout << "sampling_factor: " << sampling_factor << std::endl; 
          use_rgb = master_config["use_rgb"];
          std::cout << "use_rgb: " << use_rgb << std::endl; 
          minimum_luminance = master_config["minimum_luminance"];
          std::cout << "minimum_luminance: " << minimum_luminance << std::endl; 
          sensor_tilt_compensation = master_config["sensor_tilt_compensation"];
          std::cout << "sensor_tilt_compensation: " << sensor_tilt_compensation << std::endl; 
          heads_minimum_distance = master_config["heads_minimum_distance"]; 
          std::cout << "heads_minimum_distance: " << heads_minimum_distance << std::endl;
          voxel_size = master_config["voxel_size"]; 
          std::cout << "voxel_size: " << voxel_size << std::endl;
          apply_denoising = master_config["apply_denoising"]; 
          std::cout << "apply_denoising: " << apply_denoising << std::endl;
          std_dev_denoising = master_config["std_dev_denoising"];
          std::cout << "std_dev_denoising: " << std_dev_denoising << std::endl;
          rate_value = master_config["rate_value"];
          std::cout << "rate_value: " << rate_value << std::endl;
          json_found = true;
        }
        catch(const std::exception& e)
        {
          std::cerr << "json master/area not found: "<< e.what() << '\n';
        }
        
        // Publish Messages
        detections_pub = node_.advertise<opt_msgs::DetectionArray>("/objects_detector/detections", 3);
        skeleton_pub = node_.advertise<opt_msgs::SkeletonArrayMsg>("/detector/skeletons", 1);

        // Subscribe to Messages
        image_pub = it.advertise(sensor_string + "/objects_detector/image", 1);

        // Camera callback for intrinsics matrix update
        camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &TVMNode::camera_info_callback, this);
        if (use_pose_model) {
          point_cloud_approximate_sync_ = node_.subscribe(sensor_string + "/depth_registered/points", 10, &TVMNode::pose_callback, this);
          use_headclusters = false;
          tvm_pose_detector.reset(new NoNMSPoseFromConfig("/cfg/pose_model.json", "recognition"));
        } else {
          point_cloud_approximate_sync_ = node_.subscribe(sensor_string + "/depth_registered/points", 10, &TVMNode::yolo_callback, this);
          tvm_standard_detector.reset(new NoNMSYoloFromConfig("/cfg/pose_model.json", "recognition"));
        }
      
        sensor_name = sensor_string;
        max_capable_depth = max_distance;

        // maybe...
        transform = transform.Identity();
        anti_transform = transform.inverse();
        zone_json = zone;
        rgb_image_ = pcl::PointCloud<pcl::RGB>::Ptr(new pcl::PointCloud<pcl::RGB>);
      }

    /**
     * \brief gather info on camera intrinsics.
     *
     * \param[in] msg pointer to the camera info.
     */
    void camera_info_callback(const CameraInfo::ConstPtr & msg){
      intrinsics_matrix << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
      cam_intrins_ << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
      _cx = msg->K[2];
      _cy = msg->K[5];
      _constant_x =  1.0f / msg->K[0];
      _constant_y = 1.0f /  msg->K[4];
      camera_info_available_flag = true;
    }

    /**
     * \brief sets the background of the given camera for background removal.
     *
     * \param[in] pointer to the background cloud.
     */
    void set_background (PointCloudPtr& background_cloud){
      // Voxel grid filtering:
      //std::cout << "starting voxel grid filtering: " << std::endl;
      PointCloudT::Ptr cloud_filtered(new PointCloudT);
      pcl::VoxelGrid<PointT> voxel_grid_filter_object;
      voxel_grid_filter_object.setInputCloud(background_cloud);
      voxel_grid_filter_object.setLeafSize (voxel_size, voxel_size, voxel_size);
      voxel_grid_filter_object.filter (*cloud_filtered);
      background_cloud = cloud_filtered;

      // setting octree
      background_octree_ = new pcl::octree::OctreePointCloud<PointT>(background_octree_resolution);
      background_octree_->defineBoundingBox(-max_distance/2, -max_distance/2, 0.0, max_distance/2, max_distance/2, max_distance);
      background_octree_->setInputCloud (background_cloud);
      background_octree_->addPointsFromInputCloud ();

      //std::cout << "saving background file to tmp space: " << std::endl;
      pcl::io::savePCDFileASCII ("/tmp/background_" + sensor_name + ".pcd", *background_cloud);
      //std::cout << "background cloud done." << std::endl << std::endl;
    }
    /**
     * \brief extracts the rbg image from the pointcloud.
     *
     * \param[in] pointer to the input_cloud cloud.
     * \param[in] pointer to the output_cloud cloud.
     */
    void extract_RGB_from_pointcloud (PointCloudPtr input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud){
      // Extract RGB information from a point cloud and output the corresponding RGB point cloud  
      output_cloud->points.resize(input_cloud->height*input_cloud->width);
      output_cloud->width = input_cloud->width;
      output_cloud->height = input_cloud->height;

      pcl::RGB rgb_point;
      for (int j = 0; j < input_cloud->width; j++)
      {
        for (int i = 0; i < input_cloud->height; i++)
        { 
          rgb_point.r = (*input_cloud)(j,i).r;
          rgb_point.g = (*input_cloud)(j,i).g;
          rgb_point.b = (*input_cloud)(j,i).b;    
          (*output_cloud)(j,i) = rgb_point; 
        }
      }
    }
    /**
     * \brief function to compute the background cloud.
     *
     * \param[in] pointer to the point cloud.
     */
    PointCloudT::Ptr compute_background_cloud (PointCloudPtr& cloud){
      //std::cout << "Background acquisition..." << std::flush;
      // Initialization for background subtraction
      if (n_frame == 0){
        background_cloud = PointCloudT::Ptr (new PointCloudT);
      }

      std::string frame_id = cloud->header.frame_id;
      int frames = int(background_seconds * rate_value);
      ros::Rate rate(rate_value);
      //std::cout << "Background subtraction enabled." << std::endl;

      // Try to load the background from file:
      if (pcl::io::loadPCDFile<PointT> ("/tmp/background_" + sensor_name + ".pcd", *background_cloud) == -1)
      {
        // File not found, then background acquisition:
        //compute_background_cloud (max_background_frames, voxel_size, frame_id, rate, background_cloud);
        //std::cout << "could not find background file, begining generation..." << std::endl;
        // Create background cloud:
        background_cloud->header = cloud->header;
        background_cloud->points.clear();

        PointCloudT::Ptr cloud_filtered(new PointCloudT);
        cloud_filtered = preprocess_cloud (cloud);
        *background_cloud += *cloud_filtered;
      }
      n_frame+=1;
      return background_cloud;
    }

    /**
     * \brief function to preprocess a cloud.
     *
     * \param[in] pointer to the input_cloud point cloud.
     */
    PointCloudPtr preprocess_cloud (PointCloudPtr& input_cloud){
      //std::cout << "preprocessing cloud." << std::endl;
      // Downsample of sampling_factor in every dimension:
      PointCloudPtr cloud_downsampled(new PointCloud);
      PointCloudPtr cloud_denoised(new PointCloud);
      bool isZed_ = false;
      //float voxel_size = 0.06; //0.06;
      int sampling_factor_ = 4;//4;
      bool apply_denoising_ = true;//true;
      bool use_voxel = true;

      // Compute mean luminance:
      int n_points = input_cloud->points.size();
      double sumR, sumG, sumB = 0.0;
      for (int j = 0; j < input_cloud->width; j++)
      {
        for (int i = 0; i < input_cloud->height; i++)
        {
          sumR += (*input_cloud)(j,i).r;
          sumG += (*input_cloud)(j,i).g;
          sumB += (*input_cloud)(j,i).b;
        }
      }
      double mean_luminance = 0.3 * sumR/n_points + 0.59 * sumG/n_points + 0.11 * sumB/n_points;
      //std::cout << "mean_luminance: " << mean_luminance << std::endl;

      // Adapt thresholds for clusters points number to the voxel size:
      //max_points_ = int(float(max_points_) * std::pow(0.06/voxel_size_, 2));
      //if (voxel_size_ > 0.06)
      //  min_points_ = int(float(min_points_) * std::pow(0.06/voxel_size_, 2));

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
      //std::cout << "preprocess_cloud downsampled size: " << cloud_downsampled->size() << std::endl;

      if (apply_denoising_)
      {
        // Denoising with statistical filtering:
        pcl::StatisticalOutlierRemoval<PointT> sor;
        if (sampling_factor_ != 1)
          sor.setInputCloud (cloud_downsampled);
        else
          sor.setInputCloud (input_cloud);
        sor.setMeanK (mean_k_denoising);
        sor.setStddevMulThresh (std_dev_denoising);
        sor.filter (*cloud_denoised);
      }
      //std::cout << "preprocess_cloud cloud_denoised size: " << cloud_denoised->size() << std::endl;

      // Voxel grid filtering:
      PointCloudPtr cloud_filtered(new PointCloud);
      pcl::VoxelGrid<PointT> voxel_grid_filter_object;
      if (apply_denoising_)
        voxel_grid_filter_object.setInputCloud(cloud_denoised);
      else
      {
        if (sampling_factor_ != 1)
          voxel_grid_filter_object.setInputCloud(cloud_downsampled);
        else
          voxel_grid_filter_object.setInputCloud(input_cloud);
      }
      voxel_grid_filter_object.setLeafSize (voxel_size, voxel_size, voxel_size);
      voxel_grid_filter_object.setFilterFieldName("z");
      if (isZed_)
        voxel_grid_filter_object.setFilterLimits(-1 * max_distance, max_distance);
      else
        voxel_grid_filter_object.setFilterLimits(0.0, max_distance);
      voxel_grid_filter_object.filter (*cloud_filtered);
      //std::cout << "preprocess_cloud cloud_filtered: " << cloud_filtered->size() << std::endl;

      return cloud_filtered;
    }

    /**
     * \brief function to rotate a cloud.
     *
     * \param[in] pointer to the input_cloud point cloud.
     */
    PointCloudPtr rotate_cloud(PointCloudPtr cloud, Eigen::Affine3f transform ){
      //std::cout << "rotating cloud." << std::endl;
      PointCloudPtr rotated_cloud (new PointCloud);
      pcl::transformPointCloud(*cloud, *rotated_cloud, transform);
      rotated_cloud->header.frame_id = cloud->header.frame_id;
      return rotated_cloud;
      }

    /**
     * \brief function to rotate the ground.
     *
     * \param[in] vector of ground coefficients.
     * \param[in] the affine transform for rotation.
     */
    Eigen::VectorXf rotate_ground( Eigen::VectorXf ground_coeffs, Eigen::Affine3f transform){
      //std::cout << "rotating ground cloud." << std::endl;
      Eigen::VectorXf the_ground_coeffs_new;

      // Create a cloud with three points on the input ground plane:
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr dummy (new pcl::PointCloud<pcl::PointXYZRGB>);

      pcl::PointXYZRGB first = pcl::PointXYZRGB(0.0,0.0,0.0);
      first.x = 1.0;
      pcl::PointXYZRGB second = pcl::PointXYZRGB(0.0,0.0,0.0);
      second.y = 1.0;
      pcl::PointXYZRGB third = pcl::PointXYZRGB(0.0,0.0,0.0);
      third.x = 1.0;
      third.y = 1.0;

      dummy->points.push_back( first );
      dummy->points.push_back( second );
      dummy->points.push_back( third );

      for(uint8_t i = 0; i < dummy->points.size(); i++ )
      { // Find z given x and y:
        dummy->points[i].z = (double) ( -ground_coeffs(3) -(ground_coeffs(0) * dummy->points[i].x) - (ground_coeffs(1) * dummy->points[i].y) ) / ground_coeffs(2);
      }

      // Rotate them:
      dummy = rotate_cloud(dummy, transform);

      // Compute new ground coeffs:
      std::vector<int> indices;
      for(unsigned int i = 0; i < dummy->points.size(); i++)
      {
        indices.push_back(i);
      }
      pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> model_plane(dummy);
      model_plane.computeModelCoefficients(indices, the_ground_coeffs_new);

      return the_ground_coeffs_new;
    }

    /**
     * \brief function to compute blob clusters within the ground-removed point cloud
     *
     * \param[in] no ground pointcloud
     * \param[in] PersonClusers to be filled by this function
     */
    void compute_subclustering(PointCloudPtr no_ground_cloud, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){
      //PointCloudT::Ptr cloud(new PointCloudT);
      //*cloud = *cloud_;      
      //std::cout << "creating people clusters from compute_subclustering" << std::endl;
      // Person clusters creation from clusters indices:
      bool head_centroid = true;
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
      {
        open_ptrack::person_clustering::PersonCluster<PointT> cluster(no_ground_cloud, *it, ground_coeffs, sqrt_ground_coeffs, head_centroid, vertical_); //PersonCluster creation
        clusters.push_back(cluster);
      }

      // To avoid PCL warning:
      if (cluster_indices.size() == 0)
        cluster_indices.push_back(pcl::PointIndices());
      float x;
      float y;
      float z;
      cv::Point2f centroid2d;
      cv::Point3f centroid3d;
      for(typename std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
        {
          it->setPersonConfidence(-100.0);
          Eigen::Vector3f eigen_centroid3d = it->getTCenter();
          x = eigen_centroid3d(0);
          y = eigen_centroid3d(1);
          z = eigen_centroid3d(2);
          //std::cout << "eigen_centroid3d -x: " << x << ", y: " << y << ", z: " << z << std::endl;
          if((!std::isnan(x)) && (!std::isnan(y)) && (!std::isnan(z))){
            centroid2d = cv::Point2f(x, y);
            centroid3d = cv::Point3f(x, y, z);
            cluster_centroids2d.push_back(centroid2d);
            cluster_centroids3d.push_back(centroid3d);
            //std::cout << "centroid2d: " << centroid2d << std::endl;
            //std::cout << "centroid3d: " << centroid3d << std::endl;
            //std::cout << "centroid added. " << std::endl;
          }
        }
      //std::cout << "compute_subclustering - cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
      //std::cout << "compute_subclustering - cluster_centroids3d size: " << cluster_centroids3d.size() << std::endl;
    }

    /**
     * \brief function that creates foreground a cloud and puts blobs found into PersonClusters 
     *
     * \param[in] no ground pointcloud
     * \param[in] PersonClusers to be filled by this function
     */
    void create_foreground_cloud(const PointCloudT::ConstPtr& cloud_, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){
      int min_points = 30;
      int max_points = 5000;
      PointCloudT::Ptr cloud(new PointCloudT);
      *cloud = *cloud_;
      //std::cout << "create_foreground_cloud cloud: " << cloud->size() << std::endl;
      // Point cloud pre-processing (downsampling and filtering):
      PointCloudPtr cloud_filtered(new PointCloud);
      cloud_filtered = preprocess_cloud(cloud);
      //std::cout << "create_foreground_cloud cloud_filtered: " << cloud_filtered->size() << std::endl;

      // set background cloud here

      // Ground removal and update:
      //std::cout << "create_foreground_cloud: removing ground" << std::endl;
      pcl::IndicesPtr inliers(new std::vector<int>);
      boost::shared_ptr<pcl::SampleConsensusModelPlane<PointT> > ground_model(new pcl::SampleConsensusModelPlane<PointT>(cloud_filtered));
      ground_model->selectWithinDistance(ground_coeffs, voxel_size, *inliers);
      PointCloudPtr no_ground_cloud_ = PointCloudPtr (new PointCloud);
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
      if (inliers->size () >= (300 * 0.06 / voxel_size / std::pow (static_cast<double> (sampling_factor), 2))){
          sizeCheck = true;
      }

      if (sizeCheck) {
        ground_model->optimizeModelCoefficients (*inliers, ground_coeffs, ground_coeffs);
      }

      //std::cout << "create_foreground_cloud: ground removed no_ground_cloud_: " << no_ground_cloud_->size() << std::endl;
      // Background Subtraction (optional):
      if (background_subtraction) {
        //std::cout << "removing background" << std::endl;
        PointCloudPtr foreground_cloud(new PointCloud);
        for (unsigned int i = 0; i < no_ground_cloud_->points.size(); i++)
        {
          ////std::cout << "iter: " << i << std::endl;
          if (not (background_octree_->isVoxelOccupiedAtPoint(no_ground_cloud_->points[i].x, no_ground_cloud_->points[i].y, no_ground_cloud_->points[i].z)))
          {
            foreground_cloud->points.push_back(no_ground_cloud_->points[i]);
          }
        }
        no_ground_cloud_ = foreground_cloud;
      }
      //std::cout << "create_foreground_cloud background_subtractionv no_ground_cloud_: " << no_ground_cloud_->size() << std::endl;
      // if (no_ground_cloud_->points.size() > 0)
      // {
        // Euclidean Clustering:
      // moving to global std::vector<pcl::PointIndices> cluster_indices;
      typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
      tree->setInputCloud(no_ground_cloud_);
      pcl::EuclideanClusterExtraction<PointT> ec;
      ec.setClusterTolerance(2 * 0.06);
      ec.setMinClusterSize(min_points);
      ec.setMaxClusterSize(max_points);
      ec.setSearchMethod(tree);
      ec.setInputCloud(no_ground_cloud_);
      ec.extract(cluster_indices);

      // check cluster_indices
      //std::cout << "no_ground_cloud_ final:  " << no_ground_cloud_->size() << std::endl;
      //std::cout << "initial clusters size: " << cluster_indices.size() << std::endl;
      //std::cout << "computing clusters" << std::endl;
      compute_subclustering(no_ground_cloud_, clusters);
      if (use_headclusters){
        //std::cout << ground_coeffs << std::endl;
        //std::cout  << ground_coeffs_new << std::endl; // not being set.. why the f?;
        compute_head_subclustering(clusters);
      }
      //std::cout << "create_foreground_cloud - cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
      //std::cout << "create_foreground_cloud - cluster_centroids3d size: " << cluster_centroids3d.size() << std::endl;
      // Sensor tilt compensation to improve people detection:
      // moving to global PointCloudPtr no_ground_cloud_rotated(new PointCloud);
      // moving to global Eigen::VectorXf ground_coeffs_new;
      if(sensor_tilt_compensation)
      {
        // We want to rotate the point cloud so that the ground plane is parallel to the xOz plane of the sensor:
        Eigen::Vector3f input_plane, output_plane;
        input_plane << ground_coeffs(0), ground_coeffs(1), ground_coeffs(2);
        output_plane << 0.0, -1.0, 0.0;

        Eigen::Vector3f axis = input_plane.cross(output_plane);
        float angle = acos( input_plane.dot(output_plane)/ ( input_plane.norm()/output_plane.norm() ) );
        transform_ = Eigen::AngleAxisf(angle, axis);

        // Setting also anti_transform for later
        anti_transform_ = transform_.inverse();
        no_ground_cloud_rotated = rotate_cloud(no_ground_cloud_, transform_);
        ground_coeffs_new.resize(4);
        ground_coeffs_new = rotate_ground(ground_coeffs, transform_);
      }
      else
      {
        transform_ = transform_.Identity();
        anti_transform_ = transform_.inverse();
        no_ground_cloud_rotated = no_ground_cloud_;
        ground_coeffs_new = ground_coeffs;
      }
    }

    /**
     * \brief function that runs all subfunctions necessary to compute the ground plane 
     *
     * \param[in] input cloud from camera
     */
    void set_ground_variables(const PointCloudT::ConstPtr& cloud_){
      //std::cout << "setting ground variables." << std::endl;
      PointCloudT::Ptr cloud(new PointCloudT);
      *cloud = *cloud_;
      if (!estimate_ground_plane){
          //std::cout << "Ground plane already initialized..." << std::endl;
      } else {
        //std::cout << "background cloud: " << background_cloud->size() << std::endl;

        int min_points = 30;
        int max_points = 5000;

        // Ground estimation:
        //std::cout << "Ground plane initialization starting..." << std::endl;
        ground_estimator.setInputCloud(cloud);
        //Eigen::VectorXf ground_coeffs = ground_estimator.computeMulticamera(ground_from_extrinsic_calibration, read_ground_from_file,
        //    pointcloud_topic, sampling_factor, voxel_size);
        ground_coeffs = ground_estimator.computeMulticamera(false, false,
                  sensor_name + "/depth_registered/points", 4, 0.06);
        sqrt_ground_coeffs = (ground_coeffs - Eigen::Vector4f(0.0f, 0.0f, 0.0f, ground_coeffs(3))).norm();
      // maybe not needed
      estimate_ground_plane = false;

      }
    }

    /**
     * \brief function to further filter the blob clusters based on how similar to a human's height they may be
     *
     * \param[in] the build person clusters
     * \param[in] empty 2d cluster vector
     * \param[in] empty 3d cluster vector
     */
    void compute_head_subclustering(std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){

      // Person clusters creation from clusters indices:
      //for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices_.begin(); it != cluster_indices_.end(); ++it)
      //{
      //  open_ptrack::person_clustering::PersonCluster<PointT> cluster(cloud_, *it, ground_coeffs_, sqrt_ground_coeffs_, head_centroid_, vertical_);  // PersonCluster creation
      //  clusters.push_back(cluster);
      // }

      // reset centroids
      cluster_centroids2d.clear();
      cluster_centroids3d.clear();
      //assert that it's empty

      // To avoid PCL warning:
      if (cluster_indices.size() == 0)
        cluster_indices.push_back(pcl::PointIndices());

      // Head based sub-clustering //
      //std::cout << "compute_head_subclustering: setInputCloud" << std::endl;
      open_ptrack::person_clustering::HeadBasedSubclustering<PointT> subclustering;

      subclustering.setInputCloud(no_ground_cloud_); //no_ground_cloud_rotated
      std::cout << "setInputCloud finished" << std::endl;
      subclustering.setGround(ground_coeffs); //ground_coeffs_new
      std::cout << "setGround finished" << std::endl;
      subclustering.setInitialClusters(cluster_indices);
      std::cout << "setInitialClusters finished" << std::endl;
      subclustering.setHeightLimits(min_height_, max_height_);
      std::cout << "setHeightLimits finished" << std::endl;
      subclustering.setMinimumDistanceBetweenHeads(heads_minimum_distance_);
      std::cout << "setMinimumDistanceBetweenHeads finished" << std::endl;
      subclustering.setSensorPortraitOrientation(vertical_);
      std::cout << "setSensorPortraitOrientation finished" << std::endl;
      subclustering.subcluster(clusters);
      std::cout << "subcluster finished" << std::endl;
      std::cout << "clusters size: " << clusters.size() << std::endl;

      for(typename std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
        {
          it->setPersonConfidence(-100.0);
          cv::Point2f centroid2d;
          cv::Point3f centroid3d;
          Eigen::Vector3f eigen_centroid3d = it->getTCenter();
          centroid2d = cv::Point2f(eigen_centroid3d(0), eigen_centroid3d(1));
          centroid3d = cv::Point3f(eigen_centroid3d(0), eigen_centroid3d(1), eigen_centroid3d(2));
          cluster_centroids2d.push_back(centroid2d);
          cluster_centroids3d.push_back(centroid3d);
        }
    }

    /**
     * \brief callback function to perform detection, pose-rec, etc.
     *
     * \param[in]input cloud from camera
     */
    // void callback(const PointCloudT::ConstPtr& cloud_)
    bool check_detection_msg(opt_msgs::Detection detection_msg){
      bool send_message = false;
      if (std::isfinite(detection_msg.box_2D.x) &&
        std::isfinite(detection_msg.box_2D.y) &&
        std::isfinite(detection_msg.box_2D.width) &&
        std::isfinite(detection_msg.box_2D.height) &&
        std::isfinite(detection_msg.height) &&
        std::isfinite(detection_msg.confidence) &&
        std::isfinite(detection_msg.distance) &&
        std::isfinite(detection_msg.box_3D.p1.x) &&
        std::isfinite(detection_msg.box_3D.p1.y) &&
        std::isfinite(detection_msg.box_3D.p1.z) &&
        std::isfinite(detection_msg.centroid.x) &&
        std::isfinite(detection_msg.centroid.y) &&
        std::isfinite(detection_msg.centroid.z) &&
        std::isfinite(detection_msg.top.x) &&
        std::isfinite(detection_msg.top.y) &&
        std::isfinite(detection_msg.top.z) &&
        std::isfinite(detection_msg.bottom.x) &&
        std::isfinite(detection_msg.bottom.y) &&
        std::isfinite(detection_msg.bottom.z)){
          send_message = true;
        }
      return send_message;
    }

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

    void pose_callback(const PointCloudT::ConstPtr& cloud_) {

      //Calculate direct and inverse transforms between camera and world frame:
      tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
                                  world_transform);
      tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
                                  world_inverse_transform);

      //std::cout << "running algorithm callback" << std::endl;

      if (setbackground){
        //std::cout << "background frame n: " << n_frame << std::endl;
        PointCloudT::Ptr newcloud(new PointCloudT);
        *newcloud = *cloud_;
        background_cloud = compute_background_cloud(newcloud);
        if (n_frame >= n_frames){
          set_background(background_cloud);
          setbackground = false;
        }
      } else { 
        // background is set
        // estimate ground plane and continue with detection process
        if (estimate_ground_plane) {
          set_ground_variables(cloud_);
          estimate_ground_plane = false;
        }

        // set message vars here
        open_ptrack::opt_utils::Conversions converter; 
        std_msgs::Header cloud_header = pcl_conversions::fromPCL(cloud_->header);
        cv_bridge::CvImagePtr cv_ptr_rgb;
        cv_bridge::CvImage::Ptr  cv_ptr_depth;
        cv::Mat cv_image_clone;
        
        // set detection variables here
        pose_results* output;
        cv::Size image_size;
        float height;
        float width;
        ros::Time begin;
        double duration;

        // set detection vars here
        int r, c;
        HungarianAlgorithm HungAlgo;
        std::vector<std::vector<double>> cost_matrix;
        cv::Point2f output_centroid;
        cv::Point3f output_centroid3d;
        std::vector<int> assignment;
        yolo_centroids2d.clear();
        yolo_centroids3d.clear();
        cluster_centroids2d.clear();
        cluster_centroids3d.clear();

        // deallocate and clear
        cluster_centroids2d = std::vector<cv::Point2f>();
        cluster_centroids3d = std::vector<cv::Point3f>();
        yolo_centroids2d = std::vector<cv::Point2f>();
        yolo_centroids3d = std::vector<cv::Point3f>();
        cluster_indices = std::vector<pcl::PointIndices>();

        std::vector<int> valid;
        std::vector<open_ptrack::person_clustering::PersonCluster<PointT> > clusters;   

        // set publication messages vars here
        // generate new detection array message with the header from the rbg image
        opt_msgs::DetectionArray::Ptr detection_array_msg(new opt_msgs::DetectionArray);
        detection_array_msg->header = cloud_header;
        detection_array_msg->confidence_type = std::string("yolo");
        detection_array_msg->image_type = std::string("rgb");
        // set detection intrinsic matrix from camera variables
        for(int i = 0; i < 3; i++){
          for(int j = 0; j < 3; j++){
            detection_array_msg->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
          }
        }
        opt_msgs::SkeletonArrayMsg::Ptr skeleton_array(new opt_msgs::SkeletonArrayMsg);
        skeleton_array->header = cloud_header;
        skeleton_array->rgb_header = cloud_header;
        // set detection intrinsic matrix from camera variables
        for(int i = 0; i < 3; i++){
          for(int j = 0; j < 3; j++){
            skeleton_array->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
          }
        }

        cv::Mat cv_image (cloud_->height, cloud_->width, CV_8UC3);
        cv::Mat cv_depth_image (cloud_->height, cloud_->width, CV_32FC1);
        for (int i=0;i<cloud_->height;i++)
        {
            for (int j=0;j<cloud_->width;j++)
            {
            cv_image.at<cv::Vec3b>(i,j)[2] = cloud_->at(j,i).r;
            cv_image.at<cv::Vec3b>(i,j)[1] = cloud_->at(j,i).g;
            cv_image.at<cv::Vec3b>(i,j)[0] = cloud_->at(j,i).b;
            cv_depth_image.at<cv::Vec3b>(i,j)[0] = cloud_->at(j,i).z;
            }
        }
        // Fill rgb image:
        //rgb_image_->points.clear();                            // clear RGB pointcloud
        //extractRGBFromPointCloud(cloud_, rgb_image_);          // fill RGB pointcloud

        cv_image_clone = cv_image.clone();
        image_size = cv_image.size();
        height = static_cast<float>(image_size.height);
        width = static_cast<float>(image_size.width);

        std::cout << "running inference" << std::endl;
        // forward inference of object detector
        begin = ros::Time::now();
        output = tvm_pose_detector->forward_full(cv_image);
        duration = ros::Time::now().toSec() - begin.toSec();
        std::cout << "inference detection time: " << duration << std::endl;
        std::cout << "inference detections: " << output->num << std::endl;
          
        // build cost matrix
        if (output->num >= 1) {
          float xmin;
          float ymin;
          float xmax;
          float ymax;
          int cast_xmin;
          int cast_ymin;
          int cast_xmax;
          int cast_ymax;
          float median_x;
          float median_y;
          float median_depth;
          float mx;
          float my;
          for (int i = 0; i < output->num; i++) {
            //std::cout << "building inference centroid: " << i+1 << std::endl;
            // there's a rare case when all values == 0...
            xmin = output->boxes[i].xmin;
            ymin = output->boxes[i].ymin;
            xmax = output->boxes[i].xmax;
            ymax = output->boxes[i].ymax;

            if ((xmin == 0) && (ymin == 0) && (xmax == 0) && (ymax == 0)){
              //std::cout << "xmin: " << xmin << std::endl;
              //std::cout << "ymin: " << ymin << std::endl;
              //std::cout << "xmax: " << xmax << std::endl;
              //std::cout << "ymax: " << ymax << std::endl;
              std::cout << "all values zero. passing" << std::endl;
              continue;
            }

            cast_xmin = static_cast<int>(xmin);
            cast_ymin = static_cast<int>(ymin);
            cast_xmax = static_cast<int>(xmax);
            cast_ymax = static_cast<int>(ymax);
            // set the median of the bounding box
            median_x = xmin + ((xmax - xmin) / 2.0);
            median_y = ymin + ((ymax - ymin) / 2.0);
            
            if ( median_x < width*0.02) {
              median_x = width*0.02;
            }
            if (median_x > width*0.98) {
              median_x = width*0.98;
            }

            if ( median_y < height*0.02) {
              median_y = height*0.02;
            }
            if ( median_y > height*0.98) {
              median_y = height*0.98;
            }

            // get x, y, z points
            mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
            my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
            median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

            //std::cout << "yolo centroid - x:" << mx << ", y: " << my << ", z: " << median_depth << std::endl;
            if(std::isfinite(median_depth) && std::isfinite(mx) && std::isfinite(my)){
              output_centroid = cv::Point2f(mx, my); // or median_x, median_y
              output_centroid3d = cv::Point3f(mx, my, median_depth);
              yolo_centroids2d.push_back(output_centroid);
              yolo_centroids3d.push_back(output_centroid3d);
              //std::cout << "centroid added" << std::endl; 
              valid.push_back(i);
            }
          }

          //std::cout << "yolo centroids size: " << yolo_centroids2d.size() << std::endl;

          if (yolo_centroids2d.size() > 0){
            // filter the background and create a filtered cloud
            //std::cout << "creating foreground cloud" << std::endl;
            create_foreground_cloud(cloud_, clusters);

            // compute_head_subclustering(clusters, cluster_centroids, cluster_centroids3d);
            //std::cout << "cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
            // use 3 dimensions
            if (cluster_centroids3d.size() > 0) {
              // Initialize cost matrix for the hungarian algorithm
              //std::cout << "initialize cost matrix for the hungarian algorithm" << std::endl;
              for (int r = 0; r < cluster_centroids3d.size (); r++) {
                std::vector<double> row;
                for (int c = 0; c < yolo_centroids3d.size (); c++) {
                  float dist;
                  dist = cv::norm(cv::Mat(yolo_centroids3d[c]), cv::Mat (cluster_centroids3d[r]));
                  row.push_back(dist);
                }
                cost_matrix.push_back(row);
              }
              
              // only consider min distances???

              // Solve the Hungarian problem to match the distance of the roi centroid
              // to that of the bounding box
              //std::cout << "solving Hungarian problem" << std::endl;
              HungAlgo.Solve(cost_matrix, assignment);
              // rows == pcl centroids index
              // values ==  yolo index
              // assignment size == cluster_centroids2d size:
              // value at each == yolo
              //std::cout << "assignment shape: " <<  assignment.size() << std::endl;
              int negs = 0;
              int poss = 0;
              for (int i = 0; i < assignment.size(); i++){
                if (assignment[i] == -1){
                  negs+=1;
                } else {
                  poss+=1;
                  //std::cout << "assignment i: " << i << " value: " << assignment[i] << std::endl;
                }
              }
              //std::cout << "assignment positives: " <<  poss << std::endl;
              //std::cout << "assignment negatives: " <<  negs << std::endl;

              for (int x = 0; x < assignment.size(); x++) {
                if (assignment[x] == -1){
                  continue;
                }
                else
                {
                  // cluster_centroids3d 36 maps to yolo_centroids3d 0.
                  // yolo_centroids3d 0 maps to valid 2
                  // valid 2 == output->boxes[2]
                  // 
                  // output->boxes 0 and 1 are both bad detections...
                  int i = valid[assignment[x]];
                  //std::cout << "cluster: " << x << " to yolo number: " << i << std::endl;
                  open_ptrack::person_clustering::PersonCluster<PointT> person_cluster = clusters[x];
                  float xmin = output->boxes[i].xmin;
                  float ymin = output->boxes[i].ymin;
                  float xmax = output->boxes[i].xmax;
                  float ymax = output->boxes[i].ymax;
                  float score = output->boxes[i].score;
                  //std::cout << "xmin: " << xmin << std::endl;
                  //std::cout << "ymin: " << ymin << std::endl;
                  //std::cout << "xmax: " << xmax << std::endl;
                  //std::cout << "ymax: " << ymax << std::endl;
                  //std::cout << "score: " << score << std::endl;

                  //std::cout << "yolo xmin check " << xmin << std::endl;
                  // make sure nothing == 0 or MAX so no display errors happen
                  if (xmin <= 1.0f){
                    xmin = 1.0f;
                  }   
                  if (ymin <= 1.0f){
                    ymin = 1.0f;
                  }
                  if (xmax >= width){
                    xmax = height-1.0f;
                  }
                  if (ymax >= height){
                    ymax = height-1.0f;
                  }                     
                  //std::cout << "cleaned xmin: " << xmin << std::endl;
                  //std::cout << "cleaned ymin: " << ymin << std::endl;
                  //std::cout << "cleaned xmax: " << xmax << std::endl;
                  //std::cout << "cleaned ymax: " << ymax << std::endl;                  

                  float label = static_cast<float>(output->boxes[i].id);
                  std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];
                  //std::cout << "object_name: " << object_name << std::endl;
                  // get the coordinate information
                  int cast_xmin = static_cast<int>(xmin);
                  int cast_ymin = static_cast<int>(ymin);
                  int cast_xmax = static_cast<int>(xmax);
                  int cast_ymax = static_cast<int>(ymax);

                  //std::cout << "cast_xmin: " << cast_xmin << std::endl;
                  //std::cout << "cast_ymin: " << cast_ymin << std::endl;
                  //std::cout << "cast_xmax: " << cast_xmax << std::endl;
                  //std::cout << "cast_ymax: " << cast_ymax << std::endl; 

                  std::vector<cv::Point3f> points = output->boxes[i].points;
                  int num_parts = points.size();
                  //std::cout << "num_parts: " << num_parts << std::endl;

                  // set the median of the bounding box
                  float median_x = xmin + ((xmax - xmin) / 2.0);
                  float median_y = ymin + ((ymax - ymin) / 2.0);
                
                  if ( median_x < width*0.02) {
                    median_x = width*0.02;
                  }
                  if (median_x > width*0.98) {
                    median_x = width*0.98;
                  }

                  if ( median_y < height*0.02) {
                    median_y = height*0.02;
                  }
                  if ( median_y > height*0.98) {
                    median_y = height*0.98;
                  }
                  //std::cout << "cleaned median_x: " << median_x << std::endl;
                  //std::cout << "cleaned median_y: " << median_y << std::endl;
                  // set the new coordinates of the image so that the boxes are set
                  int new_x = static_cast<int>(median_x - (median_factor * (median_x - xmin)));
                  int new_y = static_cast<int>(median_y - (median_factor * (median_y - ymin)));
                  int new_width = static_cast<int>(2 * (median_factor * (median_x - xmin)));
                  int new_height = static_cast<int>(2 * (median_factor * (median_y - ymin)));
                  
                  //std::cout << "new_x: " << new_x << std::endl;
                  //std::cout << "new_y: " << new_y << std::endl;
                  //std::cout << "new_width: " << new_width << std::endl;
                  //std::cout << "new_height: " << new_height << std::endl;

                  // get x, y, z points
                  float mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
                  float my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
                  float median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

                  //std::cout << "mx: " << mx << std::endl;
                  //std::cout << "my: " << my << std::endl;
                  //std::cout << "median_depth: " << median_depth << std::endl;

                  // Create detection message: -- literally tatken ground_based_people_detector_node
                  float skeleton_distance;
                  float skeleton_height;
                  opt_msgs::Detection detection_msg;
                  converter.Vector3fToVector3(anti_transform * person_cluster.getMin(), detection_msg.box_3D.p1);
                  converter.Vector3fToVector3(anti_transform * person_cluster.getMax(), detection_msg.box_3D.p2);
                      
                  float head_centroid_compensation = 0.05;
                  // theoretical person centroid:
                  Eigen::Vector3f centroid3d = anti_transform * person_cluster.getTCenter();
                  Eigen::Vector3f centroid2d = converter.world2cam(centroid3d, intrinsics_matrix);
                  // theoretical person top point:
                  Eigen::Vector3f top3d = anti_transform * person_cluster.getTTop();
                  Eigen::Vector3f top2d = converter.world2cam(top3d, intrinsics_matrix);
                  // theoretical person bottom point:
                  Eigen::Vector3f bottom3d = anti_transform * person_cluster.getTBottom();
                  Eigen::Vector3f bottom2d = converter.world2cam(bottom3d, intrinsics_matrix);
                  float enlarge_factor = 1.1;
                  float pixel_xc = centroid2d(0);
                  float pixel_yc = centroid2d(1);
                  float pixel_height = (bottom2d(1) - top2d(1)) * enlarge_factor;
                  float pixel_width = pixel_height / 2;
                  detection_msg.box_2D.x = int(centroid2d(0) - pixel_width/2.0);
                  detection_msg.box_2D.y = int(centroid2d(1) - pixel_height/2.0);
                  detection_msg.box_2D.width = int(pixel_width);
                  detection_msg.box_2D.height = int(pixel_height);
                  detection_msg.height = person_cluster.getHeight();
                  detection_msg.confidence = score;//use yolo score, not pcl //person_cluster.getPersonConfidence();
                  detection_msg.distance = person_cluster.getDistance();
                  converter.Vector3fToVector3((1+head_centroid_compensation/centroid3d.norm())*centroid3d, detection_msg.centroid);
                  converter.Vector3fToVector3((1+head_centroid_compensation/top3d.norm())*top3d, detection_msg.top);
                  converter.Vector3fToVector3((1+head_centroid_compensation/bottom3d.norm())*bottom3d, detection_msg.bottom);

                  opt_msgs::SkeletonMsg skeleton;
                  skeleton.skeleton_type = opt_msgs::SkeletonMsg::COCO;
                  skeleton.joints.resize(num_parts);
                  skeleton_height = int(pixel_height);;
                  skeleton_distance = person_cluster.getDistance();

                  for (size_t i = 0; i < num_parts; i++){
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
                      joint3D.x = cloud_->at(static_cast<int>(cast_x), static_cast<int>(cast_y)).x;
                      joint3D.y = cloud_->at(static_cast<int>(cast_x), static_cast<int>(cast_y)).y;
                      joint3D.z = cloud_->at(static_cast<int>(cast_x), static_cast<int>(cast_y)).z;
                      joint3D.max_height = image_size.height;
                      joint3D.max_width = image_size.width;
                      joint3D.confidence = confidence;
                      joint3D.header = cloud_header;
                      skeleton.joints[rtpose_part_index] = joint3D;
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
                  joint3D_neck.x = cloud_->at(static_cast<int>(cast_point_x), static_cast<int>(cast_point_y)).x;
                  joint3D_neck.y = cloud_->at(static_cast<int>(cast_point_x), static_cast<int>(cast_point_y)).y;
                  joint3D_neck.z = cloud_->at(static_cast<int>(cast_point_x), static_cast<int>(cast_point_y)).z;
                  joint3D_neck.confidence = confidence;
                  joint3D_neck.header = cloud_header;
                  joint3D_neck.max_height = image_size.height;
                  joint3D_neck.max_width = image_size.width;              
                  // NECK == joint location 1
                  skeleton.joints[1] = joint3D_neck;
                  
                  // ******** CHEST
                  opt_msgs::Joint3DMsg joint3D_chest;
                  // weighted mean from rtpose
                  float cx = (point_left_hip.x + point_right_hip.x) * 0.4 + (point_left_shoulder.x + point_right_shoulder.x) * 0.1;
                  float cy = (point_left_hip.y + point_right_hip.y) * 0.4 + (point_left_shoulder.y + point_right_shoulder.y) * 0.1;
                  int cast_cx = static_cast<int>(cx);
                  int cast_cy = static_cast<int>(cy);
                  joint3D_chest.x = cloud_->at(static_cast<int>(cast_cx), static_cast<int>(cast_cy)).x;
                  joint3D_chest.y = cloud_->at(static_cast<int>(cast_cx), static_cast<int>(cast_cy)).y;
                  joint3D_chest.z = cloud_->at(static_cast<int>(cast_cx), static_cast<int>(cast_cy)).z;
                  joint3D_chest.confidence = confidence; //use confidence from previous
                  joint3D_chest.header = cloud_header;
                  joint3D_chest.max_height = image_size.height;
                  joint3D_chest.max_width = image_size.width; 
                  // CHEST == joint location 15, index 14
                  skeleton.joints[14] = joint3D_chest;
                  draw_skelaton(cv_image_clone, points);


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
                    double world_x_min;
                    double world_y_min;
                    double world_z_min;
                    double world_x_max;
                    double world_y_max;
                    double world_z_max;
                    for (zone_id = 0; zone_id < n_zones; zone_id++)
                    {
                      // need a world view here bc each detection was transformed
                      // this will work for a singular cam, but would mean each cam would have to tune
                      // to the specific area; which I think would be fine. // but will need
                      // to test to be sure
                      // a given detection can be in only one place at one time, thus it can't be in
                      // multiple zones
                      zone_string = std::to_string(zone_id);
                      //std::cout << "zone_string: " << zone_string << std::endl;
                      // type must be number but is null...
                      //https://github.com/nlohmann/json/issues/1593

                      // translate between world and frame
                      world_x_min = zone_json[zone_string]["min"]["world"]["x"];
                      world_y_min = zone_json[zone_string]["min"]["world"]["y"];
                      world_z_min = zone_json[zone_string]["min"]["world"]["z"];
                      world_x_max = zone_json[zone_string]["max"]["world"]["x"];
                      world_y_max = zone_json[zone_string]["max"]["world"]["y"];
                      world_z_max = zone_json[zone_string]["max"]["world"]["z"];

                      //std::cout << "world_x_min: " << world_x_min << std::endl;
                      //std::cout << "world_y_min: " << world_y_min << std::endl;
                      //std::cout << "world_z_min: " << world_z_min << std::endl;
                      //std::cout << "world_x_max: " << world_x_max << std::endl;
                      //std::cout << "world_y_max: " << world_y_max << std::endl;
                      //std::cout << "world_z_max: " << world_z_max << std::endl;

                      Eigen::Vector3d min_vec;
                      Eigen::Vector3d max_vec;
                      tf::Vector3 min_point(world_x_min, world_y_min, world_z_min);
                      tf::Vector3 max_point(world_x_max, world_y_max, world_z_max);
                      
                      min_point = world_transform(min_point);
                      max_point = world_transform(max_point);

                      x_min = min_point.getX();
                      y_min = min_point.getY();
                      z_min = min_point.getZ();
                      x_max = min_point.getX();
                      y_max = min_point.getY();
                      z_max = min_point.getZ();

                      //std::cout << "x_min: " << x_min << std::endl;
                      //std::cout << "y_min: " << y_min << std::endl;
                      //std::cout << "z_min: " << z_min << std::endl;
                      //std::cout << "x_max: " << x_max << std::endl;
                      //std::cout << "y_max: " << y_max << std::endl;
                      //std::cout << "z_max: " << z_max << std::endl;
                      //std::cout << "mx: " << mx << std::endl;
                      //std::cout << "my: " << my << std::endl;
                      //std::cout << "median_depth: " << median_depth << std::endl;

                      inside_area_cube = (mx <= x_max && mx >= x_min) && (my <= y_max && my >= y_min) && (median_depth <= z_max && median_depth >= z_min);
                      //std::cout << "inside_cube: " << inside_area_cube << std::endl;
                      if (inside_area_cube) {
                        break;
                      }
                    }

                    if (inside_area_cube) {
                      detection_msg.zone_id = zone_id;
                      //std::cout << "DEBUG -- INSIDE ZONE: " << zone_id << std::endl;
                    } else {
                      detection_msg.zone_id = 1000;
                    } 
                  }
                  skeleton.confidence = 100;
                  skeleton.height = skeleton_height;
                  skeleton.distance = skeleton_distance;
                  skeleton.occluded = false;
                
                  // final check here 
                  // only add to message if no nans exist
                  if (check_detection_msg(detection_msg)){
                    std::cout << "valid detection!" << std::endl;
                    skeleton_array->skeletons.push_back(skeleton);
                    detection_msg.object_name=object_name;            
                    detection_array_msg->detections.push_back(detection_msg);
                
                  cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
                  cv::putText(cv_image_clone, object_name, cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
                }
              }
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
    }


    void yolo_callback(const PointCloudT::ConstPtr& cloud_) {

      //Calculate direct and inverse transforms between camera and world frame:
      tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
                                  world_transform);
      tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
                                  world_inverse_transform);

      std::cout << "running algorithm callback" << std::endl;

      if (setbackground){
        //std::cout << "background frame n: " << n_frame << std::endl;
        PointCloudT::Ptr newcloud(new PointCloudT);
        *newcloud = *cloud_;
        background_cloud = compute_background_cloud(newcloud);
        if (n_frame >= n_frames){
          set_background(background_cloud);
          setbackground = false;
        }
      } else { 
        // background is set
        // estimate ground plane and continue with detection process
        if (estimate_ground_plane) {
          set_ground_variables(cloud_);
          estimate_ground_plane = false;
        }

        // set message vars here
        open_ptrack::opt_utils::Conversions converter; 
        std_msgs::Header cloud_header = pcl_conversions::fromPCL(cloud_->header);
        cv_bridge::CvImagePtr cv_ptr_rgb;
        cv_bridge::CvImage::Ptr  cv_ptr_depth;
        cv::Mat cv_image_clone;
        
        // set detection variables here
        yoloresults* output;
        cv::Size image_size;
        float height;
        float width;
        ros::Time begin;
        double duration;

        // set detection vars here
        int r, c;
        HungarianAlgorithm HungAlgo;
        std::vector<std::vector<double>> cost_matrix;
        cv::Point2f output_centroid;
        cv::Point3f output_centroid3d;
        std::vector<int> assignment;
        yolo_centroids2d.clear();
        yolo_centroids3d.clear();
        cluster_centroids2d.clear();
        cluster_centroids3d.clear();

        // deallocate and clear
        cluster_centroids2d = std::vector<cv::Point2f>();
        cluster_centroids3d = std::vector<cv::Point3f>();
        yolo_centroids2d = std::vector<cv::Point2f>();
        yolo_centroids3d = std::vector<cv::Point3f>();
        cluster_indices = std::vector<pcl::PointIndices>();

        std::vector<int> valid;
        std::vector<open_ptrack::person_clustering::PersonCluster<PointT> > clusters;   

        // set publication messages vars here
        // generate new detection array message with the header from the rbg image
        opt_msgs::DetectionArray::Ptr detection_array_msg(new opt_msgs::DetectionArray);
        detection_array_msg->header = cloud_header;
        detection_array_msg->confidence_type = std::string("yolo");
        detection_array_msg->image_type = std::string("rgb");
        // set detection intrinsic matrix from camera variables
        for(int i = 0; i < 3; i++){
          for(int j = 0; j < 3; j++){
            detection_array_msg->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
          }
        }
        opt_msgs::SkeletonArrayMsg::Ptr skeleton_array(new opt_msgs::SkeletonArrayMsg);
        skeleton_array->header = cloud_header;
        skeleton_array->rgb_header = cloud_header;
        // set detection intrinsic matrix from camera variables
        for(int i = 0; i < 3; i++){
          for(int j = 0; j < 3; j++){
            skeleton_array->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
          }
        }

        cv::Mat cv_image (cloud_->height, cloud_->width, CV_8UC3);
        cv::Mat cv_depth_image (cloud_->height, cloud_->width, CV_32FC1);
        for (int i=0;i<cloud_->height;i++)
        {
            for (int j=0;j<cloud_->width;j++)
            {
            cv_image.at<cv::Vec3b>(i,j)[2] = cloud_->at(j,i).r;
            cv_image.at<cv::Vec3b>(i,j)[1] = cloud_->at(j,i).g;
            cv_image.at<cv::Vec3b>(i,j)[0] = cloud_->at(j,i).b;
            cv_depth_image.at<cv::Vec3b>(i,j)[0] = cloud_->at(j,i).z;
            }
        }
        // Fill rgb image:
        //rgb_image_->points.clear();                            // clear RGB pointcloud
        //extractRGBFromPointCloud(cloud_, rgb_image_);          // fill RGB pointcloud

        cv_image_clone = cv_image.clone();
        image_size = cv_image.size();
        height = static_cast<float>(image_size.height);
        width = static_cast<float>(image_size.width);

        std::cout << "running inference" << std::endl;
        // forward inference of object detector
        begin = ros::Time::now();
        output = tvm_standard_detector->forward_full(cv_image);
        duration = ros::Time::now().toSec() - begin.toSec();
        std::cout << "inference detection time: " << duration << std::endl;
        std::cout << "inference detections: " << output->num << std::endl;
          
        // build cost matrix
        if (output->num >= 1) {
          float xmin;
          float ymin;
          float xmax;
          float ymax;
          int cast_xmin;
          int cast_ymin;
          int cast_xmax;
          int cast_ymax;
          float median_x;
          float median_y;
          float median_depth;
          float mx;
          float my;
          for (int i = 0; i < output->num; i++) {
            //std::cout << "building inference centroid: " << i+1 << std::endl;
            // there's a rare case when all values == 0...
            xmin = output->boxes[i].xmin;
            ymin = output->boxes[i].ymin;
            xmax = output->boxes[i].xmax;
            ymax = output->boxes[i].ymax;

            if ((xmin == 0) && (ymin == 0) && (xmax == 0) && (ymax == 0)){
              //std::cout << "xmin: " << xmin << std::endl;
              //std::cout << "ymin: " << ymin << std::endl;
              //std::cout << "xmax: " << xmax << std::endl;
              //std::cout << "ymax: " << ymax << std::endl;
              //std::cout << "all values zero. passing" << std::endl;
              continue;
            }

            cast_xmin = static_cast<int>(xmin);
            cast_ymin = static_cast<int>(ymin);
            cast_xmax = static_cast<int>(xmax);
            cast_ymax = static_cast<int>(ymax);
            // set the median of the bounding box
            median_x = xmin + ((xmax - xmin) / 2.0);
            median_y = ymin + ((ymax - ymin) / 2.0);
            
            if ( median_x < width*0.02) {
              median_x = width*0.02;
            }
            if (median_x > width*0.98) {
              median_x = width*0.98;
            }

            if ( median_y < height*0.02) {
              median_y = height*0.02;
            }
            if ( median_y > height*0.98) {
              median_y = height*0.98;
            }

            // get x, y, z points
            mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
            my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
            median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

            //std::cout << "yolo centroid - x:" << mx << ", y: " << my << ", z: " << median_depth << std::endl;
            if(std::isfinite(median_depth) && std::isfinite(mx) && std::isfinite(my)){
              output_centroid = cv::Point2f(mx, my); // or median_x, median_y
              output_centroid3d = cv::Point3f(mx, my, median_depth);
              yolo_centroids2d.push_back(output_centroid);
              yolo_centroids3d.push_back(output_centroid3d);
              //std::cout << "centroid added" << std::endl; 
              valid.push_back(i);
            }
          }

          //std::cout << "yolo centroids size: " << yolo_centroids2d.size() << std::endl;

          if (yolo_centroids2d.size() > 0){
            // filter the background and create a filtered cloud
            //std::cout << "creating foreground cloud" << std::endl;
            create_foreground_cloud(cloud_, clusters);

            // compute_head_subclustering(clusters, cluster_centroids, cluster_centroids3d);
            //std::cout << "cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
            // use 3 dimensions
            if (cluster_centroids3d.size() > 0) {
              // Initialize cost matrix for the hungarian algorithm
              //std::cout << "initialize cost matrix for the hungarian algorithm" << std::endl;
              for (int r = 0; r < cluster_centroids3d.size (); r++) {
                std::vector<double> row;
                for (int c = 0; c < yolo_centroids3d.size (); c++) {
                  float dist;
                  dist = cv::norm(cv::Mat(yolo_centroids3d[c]), cv::Mat (cluster_centroids3d[r]));
                  //std::cout << "c: " << c << " r: " << r << " dist: " << dist << std::endl;
                  row.push_back(dist);
                }
                cost_matrix.push_back(row);
              }
              
              // Solve the Hungarian problem to match the distance of the roi centroid
              // to that of the bounding box
              //std::cout << "solving Hungarian problem" << std::endl;
              HungAlgo.Solve(cost_matrix, assignment);
              // rows == pcl centroids index
              // values ==  yolo index
              // assignment size == cluster_centroids2d size:
              // value at each == yolo
              //std::cout << "assignment shape: " <<  assignment.size() << std::endl;
              int negs = 0;
              int poss = 0;
              for (int i = 0; i < assignment.size(); i++){
                if (assignment[i] == -1){
                  negs+=1;
                } else {
                  poss+=1;
                  //std::cout << "assignment i: " << i << " value: " << assignment[i] << std::endl;
                }
              }
              //std::cout << "assignment positives: " <<  poss << std::endl;
              //std::cout << "assignment negatives: " <<  negs << std::endl;

              for (int x = 0; x < assignment.size(); x++) {
                if (assignment[x] == -1){
                  continue;
                }
                else
                {
                  // cluster_centroids3d 36 maps to yolo_centroids3d 0.
                  // yolo_centroids3d 0 maps to valid 2
                  // valid 2 == output->boxes[2]
                  // 
                  // output->boxes 0 and 1 are both bad detections...
                  int i = valid[assignment[x]];
                  //std::cout << "cluster: " << x << " to yolo number: " << i << std::endl;
                  open_ptrack::person_clustering::PersonCluster<PointT> person_cluster = clusters[x];
                  float xmin = output->boxes[i].xmin;
                  float ymin = output->boxes[i].ymin;
                  float xmax = output->boxes[i].xmax;
                  float ymax = output->boxes[i].ymax;
                  float score = output->boxes[i].score;
                  //std::cout << "xmin: " << xmin << std::endl;
                  //std::cout << "ymin: " << ymin << std::endl;
                  //std::cout << "xmax: " << xmax << std::endl;
                  //std::cout << "ymax: " << ymax << std::endl;
                  //std::cout << "score: " << score << std::endl;

                  //std::cout << "yolo xmin check " << xmin << std::endl;
                  // make sure nothing == 0 or MAX so no display errors happen
                  if (xmin <= 1.0f){
                    xmin = 1.0f;
                  }   
                  if (ymin <= 1.0f){
                    ymin = 1.0f;
                  }
                  if (xmax >= width){
                    xmax = height-1.0f;
                  }
                  if (ymax >= height){
                    ymax = height-1.0f;
                  }                     
                  //std::cout << "cleaned xmin: " << xmin << std::endl;
                  //std::cout << "cleaned ymin: " << ymin << std::endl;
                  //std::cout << "cleaned xmax: " << xmax << std::endl;
                  //std::cout << "cleaned ymax: " << ymax << std::endl;                  

                  float label = static_cast<float>(output->boxes[i].id);
                  std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];
                  //std::cout << "object_name: " << object_name << std::endl;
                  // get the coordinate information
                  int cast_xmin = static_cast<int>(xmin);
                  int cast_ymin = static_cast<int>(ymin);
                  int cast_xmax = static_cast<int>(xmax);
                  int cast_ymax = static_cast<int>(ymax);

                  //std::cout << "cast_xmin: " << cast_xmin << std::endl;
                  //std::cout << "cast_ymin: " << cast_ymin << std::endl;
                  //std::cout << "cast_xmax: " << cast_xmax << std::endl;
                  //std::cout << "cast_ymax: " << cast_ymax << std::endl; 

                  //std::vector<cv::Point3f> points = output->boxes[i].points;
                  //int num_parts = points.size();
                  //std::cout << "num_parts: " << num_parts << std::endl;

                  // set the median of the bounding box
                  float median_x = xmin + ((xmax - xmin) / 2.0);
                  float median_y = ymin + ((ymax - ymin) / 2.0);
                
                  if ( median_x < width*0.02) {
                    median_x = width*0.02;
                  }
                  if (median_x > width*0.98) {
                    median_x = width*0.98;
                  }

                  if ( median_y < height*0.02) {
                    median_y = height*0.02;
                  }
                  if ( median_y > height*0.98) {
                    median_y = height*0.98;
                  }
                  //std::cout << "cleaned median_x: " << median_x << std::endl;
                  //std::cout << "cleaned median_y: " << median_y << std::endl;
                  // set the new coordinates of the image so that the boxes are set
                  int new_x = static_cast<int>(median_x - (median_factor * (median_x - xmin)));
                  int new_y = static_cast<int>(median_y - (median_factor * (median_y - ymin)));
                  int new_width = static_cast<int>(2 * (median_factor * (median_x - xmin)));
                  int new_height = static_cast<int>(2 * (median_factor * (median_y - ymin)));
                  
                  //std::cout << "new_x: " << new_x << std::endl;
                  //std::cout << "new_y: " << new_y << std::endl;
                  //std::cout << "new_width: " << new_width << std::endl;
                  //std::cout << "new_height: " << new_height << std::endl;

                  // get x, y, z points
                  float mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
                  float my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
                  float median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

                  //std::cout << "mx: " << mx << std::endl;
                  //std::cout << "my: " << my << std::endl;
                  //std::cout << "median_depth: " << median_depth << std::endl;

                  // Create detection message: -- literally tatken ground_based_people_detector_node
                  float skeleton_distance;
                  float skeleton_height;
                  opt_msgs::Detection detection_msg;
                  converter.Vector3fToVector3(anti_transform * person_cluster.getMin(), detection_msg.box_3D.p1);
                  converter.Vector3fToVector3(anti_transform * person_cluster.getMax(), detection_msg.box_3D.p2);
                      
                  float head_centroid_compensation = 0.05;
                  // theoretical person centroid:
                  Eigen::Vector3f centroid3d = anti_transform * person_cluster.getTCenter();
                  Eigen::Vector3f centroid2d = converter.world2cam(centroid3d, intrinsics_matrix);
                  // theoretical person top point:
                  Eigen::Vector3f top3d = anti_transform * person_cluster.getTTop();
                  Eigen::Vector3f top2d = converter.world2cam(top3d, intrinsics_matrix);
                  // theoretical person bottom point:
                  Eigen::Vector3f bottom3d = anti_transform * person_cluster.getTBottom();
                  Eigen::Vector3f bottom2d = converter.world2cam(bottom3d, intrinsics_matrix);
                  float enlarge_factor = 1.1;
                  float pixel_xc = centroid2d(0);
                  float pixel_yc = centroid2d(1);
                  float pixel_height = (bottom2d(1) - top2d(1)) * enlarge_factor;
                  float pixel_width = pixel_height / 2;
                  detection_msg.box_2D.x = int(centroid2d(0) - pixel_width/2.0);
                  detection_msg.box_2D.y = int(centroid2d(1) - pixel_height/2.0);
                  detection_msg.box_2D.width = int(pixel_width);
                  detection_msg.box_2D.height = int(pixel_height);
                  detection_msg.height = person_cluster.getHeight();
                  detection_msg.confidence = score;//use yolo score, not pcl //person_cluster.getPersonConfidence();
                  detection_msg.distance = person_cluster.getDistance();
                  converter.Vector3fToVector3((1+head_centroid_compensation/centroid3d.norm())*centroid3d, detection_msg.centroid);
                  converter.Vector3fToVector3((1+head_centroid_compensation/top3d.norm())*top3d, detection_msg.top);
                  converter.Vector3fToVector3((1+head_centroid_compensation/bottom3d.norm())*bottom3d, detection_msg.bottom);

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
                    double world_x_min;
                    double world_y_min;
                    double world_z_min;
                    double world_x_max;
                    double world_y_max;
                    double world_z_max;
                    for (zone_id = 0; zone_id < n_zones; zone_id++)
                    {
                      // need a world view here bc each detection was transformed
                      // this will work for a singular cam, but would mean each cam would have to tune
                      // to the specific area; which I think would be fine. // but will need
                      // to test to be sure
                      // a given detection can be in only one place at one time, thus it can't be in
                      // multiple zones
                      zone_string = std::to_string(zone_id);
                      //std::cout << "zone_string: " << zone_string << std::endl;
                      // type must be number but is null...
                      //https://github.com/nlohmann/json/issues/1593

                      // translate between world and frame
                      world_x_min = zone_json[zone_string]["min"]["world"]["x"];
                      world_y_min = zone_json[zone_string]["min"]["world"]["y"];
                      world_z_min = zone_json[zone_string]["min"]["world"]["z"];
                      world_x_max = zone_json[zone_string]["max"]["world"]["x"];
                      world_y_max = zone_json[zone_string]["max"]["world"]["y"];
                      world_z_max = zone_json[zone_string]["max"]["world"]["z"];

                      //std::cout << "world_x_min: " << world_x_min << std::endl;
                      //std::cout << "world_y_min: " << world_y_min << std::endl;
                      //std::cout << "world_z_min: " << world_z_min << std::endl;
                      //std::cout << "world_x_max: " << world_x_max << std::endl;
                      //std::cout << "world_y_max: " << world_y_max << std::endl;
                      //std::cout << "world_z_max: " << world_z_max << std::endl;

                      Eigen::Vector3d min_vec;
                      Eigen::Vector3d max_vec;
                      tf::Vector3 min_point(world_x_min, world_y_min, world_z_min);
                      tf::Vector3 max_point(world_x_max, world_y_max, world_z_max);
                      
                      min_point = world_transform(min_point);
                      max_point = world_transform(max_point);

                      x_min = min_point.getX();
                      y_min = min_point.getY();
                      z_min = min_point.getZ();
                      x_max = min_point.getX();
                      y_max = min_point.getY();
                      z_max = min_point.getZ();

                      //std::cout << "x_min: " << x_min << std::endl;
                      //std::cout << "y_min: " << y_min << std::endl;
                      //std::cout << "z_min: " << z_min << std::endl;
                      //std::cout << "x_max: " << x_max << std::endl;
                      //std::cout << "y_max: " << y_max << std::endl;
                      //std::cout << "z_max: " << z_max << std::endl;
                      //std::cout << "mx: " << mx << std::endl;
                      //std::cout << "my: " << my << std::endl;
                      //std::cout << "median_depth: " << median_depth << std::endl;

                      inside_area_cube = (mx <= x_max && mx >= x_min) && (my <= y_max && my >= y_min) && (median_depth <= z_max && median_depth >= z_min);
                      //std::cout << "inside_cube: " << inside_area_cube << std::endl;
                      if (inside_area_cube) {
                        break;
                      }
                    }

                    if (inside_area_cube) {
                      detection_msg.zone_id = zone_id;
                      //std::cout << "DEBUG -- INSIDE ZONE: " << zone_id << std::endl;
                    } else {
                      detection_msg.zone_id = 1000;
                    } 
                  }
                
                  // final check here 
                  // only add to message if no nans exist
                  if (check_detection_msg(detection_msg)){
                    std::cout << "valid detection!" << std::endl;
                    detection_msg.object_name=object_name;
                    detection_array_msg->detections.push_back(detection_msg);
                
                  cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
                  cv::putText(cv_image_clone, object_name, cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
                }
              }
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
    }

};

int main(int argc, char** argv) {
  std::string sensor_name;
  double max_distance;
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
  pnh.param("max_distance", max_distance, 6.25);
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  TVMNode node(nh, sensor_name, zone_json, max_distance);
  std::cout << "TVMNode init " << std::endl;
  ros::spin();
  return 0;
}
