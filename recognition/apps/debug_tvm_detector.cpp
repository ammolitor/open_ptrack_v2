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
//#include <opencv2/opencv.hpp>

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
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/console/time.h>
#include <pcl/filters/passthrough.h>

// getting errors from here...????? (I might be able to recreate it...) stemming from pcl_visualizer
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

// technically not using any person clustering
////#include <pcl/people/person_cluster.h>
////#include <pcl/people/head_based_subcluster.h>
#include <pcl/octree/octree.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <visualization_msgs/MarkerArray.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <open_ptrack/hungarian/Hungarian.h>


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
using namespace cv; // https://github.com/opencv/opencv/issues/6661
using namespace std;
using namespace cv; // https://github.com/opencv/opencv/issues/6661
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
                if (label == 0){
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
                }
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
                
                if (label == 0){
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
                }
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

/**
 * @brief The TVMDetectionNode
 */
class TVMDetectionNode {
  private:
    ros::NodeHandle node_;
    std::unique_ptr<YoloTVMFromConfig> tvm_detector;
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

    ros::Subscriber point_cloud_approximate_sync_;

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
    Eigen::Matrix3d cam_intrins_;
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
    float max_capable_depth = 6.25; // 6.25 is what the default is;
    /** \brief transforms used for compensating sensor tilt with respect to the ground plane */
    // Initialize transforms to be used to correct sensor tilt to identity matrix:
    Eigen::Affine3f transform, transform_, anti_transform, anti_transform_;
    //Eigen::Affine3f transform, anti_transform;
    bool estimate_ground_plane = true;
    Eigen::VectorXf ground_coeffs;

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

    //############################
    //## Background subtraction ##
    //############################
    //# Flag enabling/disabling background subtraction:
    bool background_subtraction = true;// #false
    //# Resolution of the octree representing the background:
    float background_octree_resolution =  0.3;
    //# Seconds to use to lear n the background:
    float background_seconds = 3.0;
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
    open_ptrack::ground_segmentation::GroundplaneEstimation<PointT> ground_estimator = open_ptrack::ground_segmentation::GroundplaneEstimation<PointT>(ground_estimation_mode, remote_ground_selection);
    PointCloudPtr no_ground_cloud_ = PointCloudPtr (new PointCloud);
    // out of frame resources at 60./
    // try 30, then 15
    double rate_value = 1.0;
    bool use_pointcloud = false;
    int centroid_argument;
    int mode_;

    // mode 1 args
    std::vector<pcl::PointIndices> cluster_indices;
    PointCloudPtr no_ground_cloud_rotated = PointCloudPtr (new PointCloud);
    Eigen::VectorXf ground_coeffs_new;
    float min_height_ = 1.3;
    float max_height_ = 2.3;
    float heads_minimum_distance_ = 0.3;
    bool vertical_ = false;
    bool use_rgb_ = true;
    std::map<std::string, std::pair<double, double>> area_thres_;
    int gluon_to_rtpose[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
    // Image to "world" transforms
    Eigen::Affine3d world2rgb;
    tf::StampedTransform world2rgb_transform;
    tf::StampedTransform world_transform;
    tf::StampedTransform world_inverse_transform;
    PointCloudT::Ptr background_cloud;
    //json zone_json;
    int n_frame = 0;
    int n_frames = 15;
    bool set_background = true;
    float sqrt_ground_coeffs;
    pcl::octree::OctreePointCloud<PointT> *background_octree_;
    // Initialize transforms to be used to correct sensor tilt to identity matrix:
    //Eigen::Affine3f transform, anti_transform;
    //transform = transform.Identity();
    //anti_transform = transform.inverse();
    pcl::PointCloud<pcl::RGB>::Ptr rgb_image_;
    std::vector<cv::Point2f> cluster_centroids2d;
    std::vector<cv::Point3f> cluster_centroids3d;
    std::vector<cv::Point2f> yolo_centroids2d;
    std::vector<cv::Point3f> yolo_centroids3d;
    /**
     * @brief constructor
     * @param nh node handler
     */
    TVMDetectionNode(ros::NodeHandle& nh, std::string sensor_string, json zone, double max_distance):
      node_(nh), it(node_)
      {
        
        try
        {
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

        // Subscribe to Messages
        rgb_image_sub.subscribe(node_, sensor_string +"/color/image_rect_color", 1);
        depth_image_sub.subscribe(node_, sensor_string+"/depth/image_rect_raw", 1);
        cloud_sub.subscribe(node_, sensor_string + "/depth_registered/points", 10);
        
        image_pub = it.advertise(sensor_string + "/objects_detector/image", 1);

        // Camera callback for intrinsics matrix update
        camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &TVMDetectionNode::camera_info_callback, this);


        point_cloud_approximate_sync_ = node_.subscribe(sensor_string + "/depth_registered/points", 10, &TVMDetectionNode::mode_1_callback_cloud_only, this);

        // create callback config 
        //cfg_server.setCallback(boost::bind(&TVMPoseNode::cfg_callback, this, _1, _2));      

        // create object-detector pointer
        //tvm_detector.reset(new YoloTVMGPU256(model_folder_path));
        //tvm_detector.reset(new YoloTVMGPU(model_folder_path));
        // maybe have this in
        // arg one HAS to have / in front of path
        // TODO add that to debugger
        tvm_detector.reset(new YoloTVMFromConfig("/cfg/model.json", "recognition"));
        sensor_name = sensor_string;
        //worldToCamTransform = read_poses_from_json(sensor_name);
        max_capable_depth = max_distance;
        area_thres_["person"] = pair<double, double>(1.8, 0.5);

        // maybe...
        transform = transform.Identity();
        anti_transform = transform.inverse();
        zone_json = zone;
        // 0 == manual
        rgb_image_ = pcl::PointCloud<pcl::RGB>::Ptr(new pcl::PointCloud<pcl::RGB>);
      }

    void camera_info_callback(const CameraInfo::ConstPtr & msg){
      intrinsics_matrix << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
      cam_intrins_ << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
      _cx = msg->K[2];
      _cy = msg->K[5];
      _constant_x =  1.0f / msg->K[0];
      _constant_y = 1.0f /  msg->K[4];
      camera_info_available_flag = true;
    }


    void extractRGBFromPointCloud (PointCloudPtr input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud)
    {
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

    void setBackground (PointCloudPtr& background_cloud)
    {
      // Voxel grid filtering:
      std::cout << "starting voxel grid filtering: " << std::endl;
      PointCloudT::Ptr cloud_filtered(new PointCloudT);
      //cloud_filtered(new PointCloudT);
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

      std::cout << "saving background file to tmp space: " << std::endl;
      pcl::io::savePCDFileASCII ("/tmp/background_" + sensor_name + ".pcd", *background_cloud);
      std::cout << "background cloud done." << std::endl << std::endl;
    }

    PointCloudT::Ptr computeBackgroundCloud (PointCloudPtr& cloud){
      std::cout << "Background acquisition..." << std::flush;
      // Initialization for background subtraction:
      //PointCloudT::Ptr background_cloud = PointCloudT::Ptr (new PointCloudT);
      if (n_frame == 0){
        background_cloud = PointCloudT::Ptr (new PointCloudT);
      }

      std::string frame_id = cloud->header.frame_id;
      int frames = int(background_seconds * rate_value);
      ros::Rate rate(rate_value);
      std::cout << "Background subtraction enabled." << std::endl;

      // Try to load the background from file:
      if (pcl::io::loadPCDFile<PointT> ("/tmp/background_" + sensor_name + ".pcd", *background_cloud) == -1)
      {
        // File not found, then background acquisition:
        //computeBackgroundCloud (max_background_frames, voxel_size, frame_id, rate, background_cloud);
        std::cout << "could not find background file, begining generation..." << std::endl;
        // Create background cloud:
        background_cloud->header = cloud->header;
        background_cloud->points.clear();

        PointCloudT::Ptr cloud_filtered(new PointCloudT);
        cloud_filtered = preprocessCloud (cloud);
        *background_cloud += *cloud_filtered;
      }
      n_frame+=1;
      return background_cloud;
    }

    PointCloudPtr preprocessCloud (PointCloudPtr& input_cloud)
    {
      std::cout << "preprocessing cloud." << std::endl;
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
      //    mean_luminance_ = 0.2126 * sumR/n_points + 0.7152 * sumG/n_points + 0.0722 * sumB/n_points;
      std::cout << "mean_luminance: " << mean_luminance << std::endl;


      // Adapt thresholds for clusters points number to the voxel size:
      //max_points_ = int(float(max_points_) * std::pow(0.06/voxel_size_, 2));
      //if (voxel_size_ > 0.06)
      //  min_points_ = int(float(min_points_) * std::pow(0.06/voxel_size_, 2));

      //yolo centroid - x:0.595159, y: -1.07777, z: 5.883
      //centroid added
      //checking yolo centroids size: 1
      //checking yolo centroids empty: 0
      //creating foreground cloud
      //create_foreground_cloud cloud: 307200
      //preprocessing cloud.
      //preprocessCloud downsampled size: 19200
      //preprocessCloud cloud_denoised size: 15652
      //create_foreground_cloud cloud_filtered: 2
      //create_foreground_cloud: removing background
      //create_foreground_cloud no_ground_cloud_: 2

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
      std::cout << "preprocessCloud downsampled size: " << cloud_downsampled->size() << std::endl;

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
      std::cout << "preprocessCloud cloud_denoised size: " << cloud_denoised->size() << std::endl;
      
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
      std::cout << "preprocessCloud cloud_filtered: " << cloud_filtered->size() << std::endl;

      return cloud_filtered;
    }

    PointCloudPtr rotateCloud(PointCloudPtr cloud, Eigen::Affine3f transform ){
      std::cout << "rotating cloud." << std::endl;
        PointCloudPtr rotated_cloud (new PointCloud);
        pcl::transformPointCloud(*cloud, *rotated_cloud, transform);
        rotated_cloud->header.frame_id = cloud->header.frame_id;
        return rotated_cloud;
      }

    Eigen::VectorXf rotateGround( Eigen::VectorXf ground_coeffs, Eigen::Affine3f transform){
      std::cout << "rotating ground cloud." << std::endl;
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
      dummy = rotateCloud(dummy, transform);

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

    void compute_subclustering(PointCloudPtr no_ground_cloud, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){
      //PointCloudT::Ptr cloud(new PointCloudT);
      //*cloud = *cloud_;      
      std::cout << "creating people clusters from compute_subclustering" << std::endl;
      // Person clusters creation from clusters indices:
      bool head_centroid = true;
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
      {
        open_ptrack::person_clustering::PersonCluster<PointT> cluster(no_ground_cloud, *it, ground_coeffs, sqrt_ground_coeffs, head_centroid, vertical_); //PersonCluster creation
        clusters.push_back(cluster);
        //std::cout << "debug person_cluster ground_coeffs: " << ground_coeffs << std::endl;
        //std::cout << "debug person_cluster sqrt_ground_coeffs: " << sqrt_ground_coeffs << std::endl;
        //std::cout << "debug person_cluster getTCenter: " << cluster.getTCenter() << std::endl;
        //std::cout << "debug person_cluster getCenter: " << cluster.getCenter() << std::endl;
        //std::cout << "debug person_cluster getHeight: " << cluster.getHeight() << std::endl;
        //std::cout << "debug person_cluster getDistance: " << cluster.getDistance() << std::endl;
        //std::cout << "debug person_cluster getTTop: " << cluster.getTTop() << std::endl;
        //std::cout << "debug person_cluster getTBottom: " << cluster.getTBottom() << std::endl;
        //std::cout << "debug person_cluster getTop: " << cluster.getTop() << std::endl;
        //std::cout << "debug person_cluster getBottom: " << cluster.getBottom() << std::endl;
        //std::cout << "debug person_cluster getMin: " << cluster.getMin() << std::endl;
        //std::cout << "debug person_cluster getMax: " << cluster.getMax() << std::endl;
        //std::cout << "debug person_cluster getAngle: " << cluster.getAngle() << std::endl;
        //std::cout << "debug person_cluster getNumberPoints: " << cluster.getNumberPoints() << std::endl;
        //std::cout << "debug person_cluster getPersonConfidence: " << cluster.getPersonConfidence() << std::endl;
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
          std::cout << "eigen_centroid3d -x: " << x << ", y: " << y << ", z: " << z << std::endl;
          if((!std::isnan(x)) && (!std::isnan(y)) && (!std::isnan(z))){
            centroid2d = cv::Point2f(x, y);
            centroid3d = cv::Point3f(x, y, z);
            cluster_centroids2d.push_back(centroid2d);
            cluster_centroids3d.push_back(centroid3d);
            std::cout << "centroid2d: " << centroid2d << std::endl;
            std::cout << "centroid3d: " << centroid3d << std::endl;
            std::cout << "centroid added. " << std::endl;
          }
        }
      std::cout << "compute_subclustering - cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
      std::cout << "compute_subclustering - cluster_centroids3d size: " << cluster_centroids3d.size() << std::endl;
  }

    void create_foreground_cloud(const PointCloudT::ConstPtr& cloud_, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){
      int min_points = 30;
      int max_points = 5000;
      PointCloudT::Ptr cloud(new PointCloudT);
      *cloud = *cloud_;
      std::cout << "create_foreground_cloud cloud: " << cloud->size() << std::endl;
      // Point cloud pre-processing (downsampling and filtering):
      PointCloudPtr cloud_filtered(new PointCloud);
      cloud_filtered = preprocessCloud(cloud);
      std::cout << "create_foreground_cloud cloud_filtered: " << cloud_filtered->size() << std::endl;

      // set background cloud here

      // Ground removal and update:
      std::cout << "create_foreground_cloud: removing ground" << std::endl;
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

      std::cout << "create_foreground_cloud: ground removed no_ground_cloud_: " << no_ground_cloud_->size() << std::endl;
      // Background Subtraction (optional):
      if (background_subtraction) {
        std::cout << "removing background" << std::endl;
        PointCloudPtr foreground_cloud(new PointCloud);
        for (unsigned int i = 0; i < no_ground_cloud_->points.size(); i++)
        {
          //std::cout << "iter: " << i << std::endl;
          if (not (background_octree_->isVoxelOccupiedAtPoint(no_ground_cloud_->points[i].x, no_ground_cloud_->points[i].y, no_ground_cloud_->points[i].z)))
          {
            foreground_cloud->points.push_back(no_ground_cloud_->points[i]);
          }
        }
        no_ground_cloud_ = foreground_cloud;
      }
      std::cout << "create_foreground_cloud background_subtractionv no_ground_cloud_: " << no_ground_cloud_->size() << std::endl;
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
      std::cout << "no_ground_cloud_ final:  " << no_ground_cloud_->size() << std::endl;
      std::cout << "initial clusters size: " << cluster_indices.size() << std::endl;
      std::cout << "computing clusters" << std::endl;
      compute_subclustering(no_ground_cloud_, clusters);
      std::cout << "create_foreground_cloud - cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
      std::cout << "create_foreground_cloud - cluster_centroids3d size: " << cluster_centroids3d.size() << std::endl;
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
        no_ground_cloud_rotated = rotateCloud(no_ground_cloud_, transform_);
        ground_coeffs_new.resize(4);
        ground_coeffs_new = rotateGround(ground_coeffs, transform_);
      }
      else
      {
        transform_ = transform_.Identity();
        anti_transform_ = transform_.inverse();
        no_ground_cloud_rotated = no_ground_cloud_;
        ground_coeffs_new = ground_coeffs;
      }
    }

    void set_ground_variables(const PointCloudT::ConstPtr& cloud_){
      std::cout << "setting ground variables." << std::endl;
      PointCloudT::Ptr cloud(new PointCloudT);
      *cloud = *cloud_;
      if (!estimate_ground_plane){
         std::cout << "Ground plane already initialized..." << std::endl;
      } else {
        std::cout << "background cloud: " << background_cloud->size() << std::endl;
        //sampling_factor_ = 1;
        //voxel_size_ = 0.06;
        //max_distance_ = 50.0;
        //vertical_ = false;
        //head_centroid_ = true;
        //min_height_ = 1.3;
        //max_height_ = 2.3;
        //min_points_ = 30;     // this value is adapted to the voxel size in method "compute"
        //max_points_ = 5000;   // this value is adapted to the voxel size in method "compute"
        //dimension_limits_set_ = false;
        //heads_minimum_distance_ = 0.3;
        //use_rgb_ = true;
        //mean_luminance_ = 0.0;
        //sensor_tilt_compensation_ = false;
        //background_subtraction_ = false;
        int min_points = 30;
        int max_points = 5000;

        // set flag vales for mandatory parameters:
        //sqrt_ground_coeffs_ = std::numeric_limits<float>::quiet_NaN();
        //person_classifier_set_flag_ = false;
        //frame_counter_ = 0;

        // Ground estimation:
        std::cout << "Ground plane initialization starting..." << std::endl;
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


    // mode_ 0 specific utilities
    vector<Point3f> clusterPoints(vector<Point3f>& points)
    {
        Mat labels;
        cv::kmeans(points, 2,  labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS);

        vector<Point3f> points_class[2];
        double z[2];
        for(int i = 0; i < points.size(); i++ )
        {
            points_class[labels.at<int>(i)].push_back(points[i]);
            z[labels.at<int>(i)] += points[i].z;
        }
        for(int i = 0; i < 2; i ++)
        {
            z[i] /= points_class[i].size();
        }
        // int foreground_label = 0;
        // for(int i = 1; i < 2; i++ )
        // {
        //     if(z[i] < z[foreground_label])
        //     {
        //         foreground_label = i;
        //     }
        // }
        int foreground_label = points_class[0].size() > points_class[1].size() ? 0 : 1;
        return points_class[foreground_label];
    }
    //https://github.com/EpsAvlc/cam_lidar_fusion
    bool filterBboxByArea(int xmin, int ymin, int xmax, int ymax, double range)
    {
        
        int bbox_area = (xmax - xmin) * (ymax - ymin);
        //TODO add box class
        std::string bbox_class = "person"; //bbox.Class;
        if(area_thres_.count(bbox_class) < 1)
            return false;
        Eigen::MatrixXd rect_corners(3, 4);
        rect_corners.block(0,0,3,1) << 0, 0, range;
        rect_corners.block(0,1,3,1) << area_thres_[bbox_class].first, 0, range;
        rect_corners.block(0,2,3,1) << area_thres_[bbox_class].first, area_thres_[bbox_class].second, range;
        rect_corners.block(0,3,3,1) << 0, area_thres_[bbox_class].second, range;
        
        Eigen::MatrixXd rect_corners_2d_homo = cam_intrins_ * rect_corners;
        vector<Point2d> rect_corners_2d;
        for(int i = 0; i < 4; i ++)
        {
            Point2d tmp;
            tmp.x = rect_corners_2d_homo(0, i) / rect_corners_2d_homo(2, i);
            tmp.y = rect_corners_2d_homo(1, i) / rect_corners_2d_homo(2, i);
            rect_corners_2d.push_back(tmp);
        }
        double width = rect_corners_2d[1].x - rect_corners_2d[0].x;
        double height = rect_corners_2d[2].y - rect_corners_2d[1].y;
        int hypo_area = static_cast<int>(width * height);
        // cout << bbox_area << ", " << hypo_area << endl;
        if(bbox_area < hypo_area * 0.0 || bbox_area > hypo_area * 3)
        {
            // cout << hypo_area * 0.5 << endl;
            return false;
        }
        return true;
    }

    void drawCube(Mat& img, Point3d min_xyz, Point3d max_xyz)
    {
        Eigen::MatrixXd corners(3,8);
        double min_max_x[2] = {min_xyz.x, max_xyz.x};
        double min_max_y[2] = {min_xyz.y, max_xyz.y};
        double min_max_z[2] = {min_xyz.z, max_xyz.z};
        int corner_index = 0;
        for(int i = 0; i < 2; i++)
        {
            for(int j = 0; j < 2; j++)
            {
                for(int k = 0; k < 2; k++)
                {
                    corners(0, corner_index) = min_max_x[i]; 
                    corners(1, corner_index) = min_max_y[j];
                    corners(2, corner_index) = min_max_z[k];    
                    corner_index ++;      
                }
            }
        }
        Eigen::MatrixXd corners_2d_homo = cam_intrins_ * corners;
        vector<Point2d> corners_2d;
        for(int i = 0; i < 8; i++)
        {
            Point2d tmp;
            tmp.x = corners_2d_homo(0, i) / corners_2d_homo(2, i);
            tmp.y = corners_2d_homo(1, i) / corners_2d_homo(2, i);
            corners_2d.push_back(tmp);
        }
        int thickness = 2;
        cv::line(img, corners_2d[0], corners_2d[1], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[0], corners_2d[2], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[0], corners_2d[4], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[1], corners_2d[3], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[1], corners_2d[5], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[2], corners_2d[3], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[2], corners_2d[6], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[3], corners_2d[7], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[4], corners_2d[5], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[4], corners_2d[6], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[5], corners_2d[7], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[6], corners_2d[7], cv::Scalar(197,75,30), thickness);
    }

    void extract_cube(Mat& img, Point3d min_xyz, Point3d max_xyz)
    {
        // CREATE FUNCTION HERE
        Eigen::MatrixXd corners(3,8);
        double min_max_x[2] = {min_xyz.x, max_xyz.x};
        double min_max_y[2] = {min_xyz.y, max_xyz.y};
        double min_max_z[2] = {min_xyz.z, max_xyz.z};
        int corner_index = 0;
        for(int i = 0; i < 2; i++)
        {
            for(int j = 0; j < 2; j++)
            {
                for(int k = 0; k < 2; k++)
                {
                    corners(0, corner_index) = min_max_x[i]; 
                    corners(1, corner_index) = min_max_y[j];
                    corners(2, corner_index) = min_max_z[k];    
                    corner_index ++;      
                }
            }
        }
        Eigen::MatrixXd corners_2d_homo = cam_intrins_ * corners;
        vector<Point2d> corners_2d;
        for(int i = 0; i < 8; i++)
        {
            Point2d tmp;
            tmp.x = corners_2d_homo(0, i) / corners_2d_homo(2, i);
            tmp.y = corners_2d_homo(1, i) / corners_2d_homo(2, i);
            corners_2d.push_back(tmp);
        }
        int thickness = 2;
        cv::line(img, corners_2d[0], corners_2d[1], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[0], corners_2d[2], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[0], corners_2d[4], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[1], corners_2d[3], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[1], corners_2d[5], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[2], corners_2d[3], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[2], corners_2d[6], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[3], corners_2d[7], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[4], corners_2d[5], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[4], corners_2d[6], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[5], corners_2d[7], cv::Scalar(197,75,30), thickness);
        cv::line(img, corners_2d[6], corners_2d[7], cv::Scalar(197,75,30), thickness);
    }

  //json export_cube(blah blah){
  // do stuff here  
  //} end
    void extract_cube_boundries(vector<Point3f> points_fg, Point3d min_xyz, Point3d max_xyz) {
      for(int i = 0; i < points_fg.size(); i++) {
        pcl::PointXYZ tmp_pt;
        tmp_pt.x = points_fg[i].x;
        tmp_pt.y = points_fg[i].y;
        tmp_pt.z = points_fg[i].z;
        
        if(min_xyz.x > tmp_pt.x)
            min_xyz.x = tmp_pt.x;
        if(min_xyz.y > tmp_pt.y)
            min_xyz.y = tmp_pt.y;
        if(min_xyz.z > tmp_pt.z)
            min_xyz.z = tmp_pt.z;

        if(max_xyz.x < tmp_pt.x)
            max_xyz.x = tmp_pt.x;
        if(max_xyz.y < tmp_pt.y)
            max_xyz.y = tmp_pt.y;
        if(max_xyz.z < tmp_pt.z)
            max_xyz.z = tmp_pt.z;
        }
      }


    void compute_head_subclustering(std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters, std::vector<cv::Point2f> cluster_centroids2d, std::vector<cv::Point3f> cluster_centroids3d){

      // Person clusters creation from clusters indices:
      //for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices_.begin(); it != cluster_indices_.end(); ++it)
      //{
      //  open_ptrack::person_clustering::PersonCluster<PointT> cluster(cloud_, *it, ground_coeffs_, sqrt_ground_coeffs_, head_centroid_, vertical_);  // PersonCluster creation
      //  clusters.push_back(cluster);
     // }


      // To avoid PCL warning:
      if (cluster_indices.size() == 0)
        cluster_indices.push_back(pcl::PointIndices());

      // Head based sub-clustering //
      std::cout << "compute_head_subclustering: setInputCloud" << std::endl;
      open_ptrack::person_clustering::HeadBasedSubclustering<PointT> subclustering;
      subclustering.setInputCloud(no_ground_cloud_rotated);
      subclustering.setGround(ground_coeffs_new);
      subclustering.setInitialClusters(cluster_indices);
      subclustering.setHeightLimits(min_height_, max_height_);
      subclustering.setMinimumDistanceBetweenHeads(heads_minimum_distance_);
      subclustering.setSensorPortraitOrientation(vertical_);
      subclustering.subcluster(clusters);

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


  private:
    /**
     * @brief callback for camera information that does detection on images
     *  and publishes the detections to specific topics
     * @param rgb_image  the rgb image message
     * @param depth_image  the depth/stereo image message
     * @param zone_json the json that contains the zone information
     */

  
  void mode_1_callback_cloud_only(const PointCloudT::ConstPtr& cloud_) {//,
                                  //json zone_json) {


      //Calculate direct and inverse transforms between camera and world frame:
      tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
                                 world_transform);
      tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
                                 world_inverse_transform);

      std::cout << "running algorithm callback" << std::endl;

      if (set_background){
        std::cout << "background frame n: " << n_frame << std::endl;
        PointCloudT::Ptr newcloud(new PointCloudT);
        *newcloud = *cloud_;
        background_cloud = computeBackgroundCloud(newcloud);
        if (n_frame >= n_frames){
          setBackground(background_cloud);
          set_background = false;
        }
      }
      else
      {
        if (estimate_ground_plane) {
          set_ground_variables(cloud_);
          estimate_ground_plane = false;
        }

        // set message vars here
        std_msgs::Header cloud_header = pcl_conversions::fromPCL(cloud_->header);
        cv_bridge::CvImagePtr cv_ptr_rgb;
        cv_bridge::CvImage::Ptr  cv_ptr_depth;
        //cv::Mat cv_image;
        //cv::Mat cv_depth_image;
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
        detection_array_msg->header = cloud_header;
        detection_array_msg->confidence_type = std::string("yolo");
        detection_array_msg->image_type = std::string("rgb");
        // set detection intrinsic matrix from camera variables
        for(int i = 0; i < 3; i++){
          for(int j = 0; j < 3; j++){
            detection_array_msg->intrinsic_matrix.push_back(intrinsics_matrix(i, j));
          }
        }
        open_ptrack::opt_utils::Conversions converter; 
        // NOTE
        // convert message to usable formats
        // available encoding types:
        // ---- sensor_msgs::image_encodings::BGR8
        // ---- sensor_msgs::image_encodings::TYPE_16UC1;
        // ---- sensor_msgs::image_encodings::TYPE_32UC1;
      
        //cv_ptr_rgb = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
        //cv_image = cv_ptr_rgb->image;
        //cv_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
        //cv_depth_image = cv_ptr_depth->image;
        //image_size = cv_image.size();
        //height =  static_cast<float>(image_size.height);
        //width =  static_cast<float>(image_size.width);
        //cv_image_clone = cv_image.clone();

        //////////////////////////////////////////////////////////////////////////////////////////////
  //////      // Create XYZ cloud:
  //////      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  //////      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  //////      // fails?
  //////      //pcl::fromROSMsg(*cloud_, *pcl_cloud);
  //////      //https://answers.ros.org/question/9515/how-to-convert-between-different-point-cloud-types-using-pcl/
  //////      pcl::PointXYZRGB xyzrgb_point;
  //////      cloud_xyzrgb->points.resize(cloud_->width * cloud_->height, xyzrgb_point);
  //////      cloud_xyzrgb->width = cloud_->width;
  //////      cloud_xyzrgb->height = cloud_->height;
  //////      cloud_xyzrgb->is_dense = false;

  //////      int cloud__is_empty = cloud_->size();
  //////      std::cout << "DEBUG: cloud__is_empty: " << cloud__is_empty << std::endl;
  //////      int cloud_width = cloud_->width;
  //////      std::cout << "DEBUG: cloud_width: " << cloud_width << std::endl;
  //////      int cloud_height = cloud_->height;
  //////      std::cout << "DEBUG: cloud_height: " << cloud_height << std::endl;

  //////      // fill xyzrgb
  //////      for (int i=0;i<cloud_->height;i++)
  //////      {
  //////          for (int j=0;j<cloud_->width;j++)
  //////          {
  //////          cloud_xyzrgb->at(j,i).x = cloud_->at(j,i).x;
  //////          cloud_xyzrgb->at(j,i).y = cloud_->at(j,i).y;
  //////          cloud_xyzrgb->at(j,i).z = cloud_->at(j,i).z;
  //////          }
  //////      }

  //////      int cloud_xyzrgb_is_empty = cloud_xyzrgb->size();
  //////      std::cout << "DEBUG: cloud_xyzrgb_is_empty: " << cloud_xyzrgb_is_empty << std::endl;

  //////      pcl::PointXYZ xyz_point;
  //////      pcl_cloud->points.resize(cloud_->width * cloud_->height, xyz_point);
  //////      pcl_cloud->width = cloud_->width;
  //////      pcl_cloud->height = cloud_->height;
  //////      pcl_cloud->is_dense = false;

  //////      for (size_t i = 0; i < cloud_->points.size(); i++) {
  //////          pcl_cloud->points[i].x = cloud_->points[i].x;
  //////          pcl_cloud->points[i].y = cloud_->points[i].y;
  //////          pcl_cloud->points[i].z = cloud_->points[i].z;
  //////      }

  //////      // define xyz 3d points in cloud
  //////      Eigen::MatrixXd points_3d_in_cam(3, pcl_cloud->size());
  //////      for(int i = 0; i < pcl_cloud->size(); i++)
  //////      {
  //////          points_3d_in_cam(0, i) = (*pcl_cloud)[i].x;
  //////          points_3d_in_cam(1, i) = (*pcl_cloud)[i].y;
  //////          points_3d_in_cam(2, i) = (*pcl_cloud)[i].z;
  //////      }    

      
  //////      // define this, but maybe do like the camera transform here????
  //////      Eigen::MatrixXd points_2d_homo = cam_intrins_ * points_3d_in_cam;

  //////      // lets assume that points_2d_homo == world transform...

  //////      Eigen::MatrixXd points_2d(2, pcl_cloud->size());
  //////      for(int i = 0; i < pcl_cloud->size(); i++)
  //////      {
  //////          points_2d(0, i) = points_2d_homo(0, i) / points_2d_homo(2, i);
  //////          points_2d(1, i) = points_2d_homo(1, i) / points_2d_homo(2, i);
  //////      }

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

        //////////////////////////////////////////////////////////////////////////////////////////////

        // override and use pointcloud
        //cv_image = curr_image.clone();
        cv_image_clone = cv_image.clone();
        image_size = cv_image.size();
        height =  static_cast<float>(image_size.height);
        width =  static_cast<float>(image_size.width);
        // override and use pointcloud

        // necessary? or can we just use height/width of cv_image
        int DISPLAY_RESOLUTION_HEIGHT = image_size.height;
        int DISPLAY_RESOLUTION_WIDTH = image_size.width;

        std::cout << "running yolo" << std::endl;
        // forward inference of object detector
        begin = ros::Time::now();
        output = tvm_detector->forward_full(cv_image, .3);
        duration = ros::Time::now().toSec() - begin.toSec();
        std::cout << "yolo detection time: " << duration << std::endl;
        std::cout << "yolo detections: " << output->num << std::endl;
        
        std::cout << "initializing clusters" << std::endl;
        std::vector<open_ptrack::person_clustering::PersonCluster<PointT> > clusters;   // vector containing persons clusters
        // we run ground-based-people-detector pcl subclustering operation

        int r, c;
        // don't forget to import hungarian algo
        HungarianAlgorithm HungAlgo;
        std::vector<std::vector<double>> cost_matrix;
        cv::Point2f output_centroid;
        cv::Point3f output_centroid3d;
        std::vector<int> assignment;
        // empty each of the centroid arrays.
        yolo_centroids2d.clear();
        yolo_centroids3d.clear();
        cluster_centroids2d.clear();
        cluster_centroids3d.clear();
        std::vector<int> valid;

        // fall back on subclusters?????
        // no detections? no forward...
          
        // build cost matrix
        std::cout << "checking yolo output" << std::endl;
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
          std::cout << "building yolo centroids" << std::endl;
          for (int i = 0; i < output->num; i++) {
            std::cout << "building yolo centroid: " << i+1 << std::endl;
            std::cout << "xmin check: " << output->boxes[i].xmin << std::endl;
            xmin = output->boxes[i].xmin;
            ymin = output->boxes[i].ymin;
            xmax = output->boxes[i].xmax;
            ymax = output->boxes[i].ymax;
            cast_xmin = static_cast<int>(xmin);
            cast_ymin = static_cast<int>(ymin);
            cast_xmax = static_cast<int>(xmax);
            cast_ymax = static_cast<int>(ymax);
            // set the median of the bounding box
            median_x = xmin + ((xmax - xmin) / 2.0);
            median_y = ymin + ((ymax - ymin) / 2.0);
            // If the detect box coordinat is near edge of image, it will return a error 'Out of im.size().'
            // lets resize the detection until it's inside the width...
            
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

            std::cout << "yolo centroid - x:" << mx << ", y: " << my << ", z: " << median_depth << std::endl;
            if(std::isfinite(median_depth) && std::isfinite(mx) && std::isfinite(my)){
              output_centroid = cv::Point2f(mx, my); // or median_x, median_y
              output_centroid3d = cv::Point3f(mx, my, median_depth);
              yolo_centroids2d.push_back(output_centroid);
              yolo_centroids3d.push_back(output_centroid3d);
              std::cout << "centroid added" << std::endl; 
              valid.push_back(i);
            }
          }

          std::cout << "checking yolo centroids size: " << yolo_centroids2d.size() << std::endl;
          std::cout << "checking yolo centroids empty: " << yolo_centroids2d.empty() << std::endl;

          if (yolo_centroids2d.size() > 0){
          // filter the background and create a filtered cloud
            std::cout << "creating foreground cloud" << std::endl;
            create_foreground_cloud(cloud_, clusters);

            //compute_head_subclustering(clusters, cluster_centroids, cluster_centroids3d);
            std::cout << "cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
            // use 3 dimensions
            if (cluster_centroids3d.size() > 0) {
              // Initialize cost matrix for the hungarian algorithm
              std::cout << "initialize cost matrix for the hungarian algorithm" << std::endl;
              for (int r = 0; r < cluster_centroids3d.size (); r++) {
                std::vector<double> row;
                for (int c = 0; c < yolo_centroids3d.size (); c++) {
                  float dist;
                  dist = cv::norm(cv::Mat(yolo_centroids3d[c]), cv::Mat (cluster_centroids3d[r]));
                  row.push_back(dist);
                }
                cost_matrix.push_back(row);
              }
              
              // Solve the Hungarian problem to match the distance of the roi centroid
              // to that of the bounding box
              std::cout << "solving Hungarian problem" << std::endl;
              HungAlgo.Solve(cost_matrix, assignment);
              //rows == pcl centroids index
              // values ==  yolo index
              // assignment size == cluster_centroids2d size:
              // value at each == yolo
              std::cout << "assignment shape: " <<  assignment.size() << std::endl;
              int negs = 0;
              int poss = 0;
              for (int i = 0; i < assignment.size(); i++){
                if (assignment[i] == -1){
                  negs+=1;
                } else {
                  poss+=1;
                  std::cout << "assignment i: " << i << " value: " << assignment[i] << std::endl;
                }
              }
              std::cout << "assignment positives: " <<  poss << std::endl;
              std::cout << "assignment negatives: " <<  negs << std::endl;

              //for(typename std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
              for (int x = 0; x < assignment.size(); x++) {
                if (assignment[x] == -1){
                  continue;
                }
                else
                {
                  int i = valid[assignment[x]];
                  std::cout << "cluster: " << x << " to yolo number: " << i << std::endl;
                  open_ptrack::person_clustering::PersonCluster<PointT> person_cluster = clusters[x];
                  float xmin = output->boxes[i].xmin;
                  float ymin = output->boxes[i].ymin;
                  float xmax = output->boxes[i].xmax;
                  float ymax = output->boxes[i].ymax;
                  float score = output->boxes[i].score;
                  float label = static_cast<float>(output->boxes[i].id);
                  std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];
                  // get the coordinate information
                  int cast_xmin = static_cast<int>(xmin);
                  int cast_ymin = static_cast<int>(ymin);
                  int cast_xmax = static_cast<int>(xmax);
                  int cast_ymax = static_cast<int>(ymax);
                  //std::vector<cv::Point3f> points = output->boxes[i].points;
                  //int num_parts = points.size();

                  // set the median of the bounding box
                  float median_x = xmin + ((xmax - xmin) / 2.0);
                  float median_y = ymin + ((ymax - ymin) / 2.0);

                  // If the detect box coordinat is near edge of image, it will return a error 'Out of im.size().'
                  //if ( median_x < width*0.02 || median_x > width*0.98) continue;
                  //if ( median_y < height*0.02 || median_y > height*0.98) continue;
                
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

                  // set the new coordinates of the image so that the boxes are set
                  int new_x = static_cast<int>(median_x - (median_factor * (median_x - xmin)));
                  int new_y = static_cast<int>(median_y - (median_factor * (median_y - ymin)));
                  int new_width = static_cast<int>(2 * (median_factor * (median_x - xmin)));
                  int new_height = static_cast<int>(2 * (median_factor * (median_y - ymin)));
                
                  //float median_depth = cv_depth_image.at<float>(median_y, median_x) / mm_factor;
                  // set the mx/my wtr the intrinsic camera matrix
                  //float mx = (median_x - _cx) * median_depth * _constant_x;
                  //float my = (median_y - _cy) * median_depth * _constant_y;

                  // get x, y, z points
                  float mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
                  float my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
                  float median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

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
                  detection_msg.confidence = score;//person_cluster.getPersonConfidence();
                  detection_msg.distance = person_cluster.getDistance();
                  converter.Vector3fToVector3((1+head_centroid_compensation/centroid3d.norm())*centroid3d, detection_msg.centroid);
                  converter.Vector3fToVector3((1+head_centroid_compensation/top3d.norm())*top3d, detection_msg.top);
                  converter.Vector3fToVector3((1+head_centroid_compensation/bottom3d.norm())*bottom3d, detection_msg.bottom);

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
                      std::cout << "zone_string: " << zone_string << std::endl;
                      // type must be number but is null...
                      //https://github.com/nlohmann/json/issues/1593

                      // translate between world and frame
                      world_x_min = zone_json[zone_string]["min"]["world"]["x"];
                      world_y_min = zone_json[zone_string]["min"]["world"]["y"];
                      world_z_min = zone_json[zone_string]["min"]["world"]["z"];
                      world_x_max = zone_json[zone_string]["max"]["world"]["x"];
                      world_y_max = zone_json[zone_string]["max"]["world"]["y"];
                      world_z_max = zone_json[zone_string]["max"]["world"]["z"];

                      std::cout << "world_x_min: " << world_x_min << std::endl;
                      std::cout << "world_y_min: " << world_y_min << std::endl;
                      std::cout << "world_z_min: " << world_z_min << std::endl;
                      std::cout << "world_x_max: " << world_x_max << std::endl;
                      std::cout << "world_y_max: " << world_y_max << std::endl;
                      std::cout << "world_z_max: " << world_z_max << std::endl;

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

                      std::cout << "x_min: " << x_min << std::endl;
                      std::cout << "y_min: " << y_min << std::endl;
                      std::cout << "z_min: " << z_min << std::endl;
                      std::cout << "x_max: " << x_max << std::endl;
                      std::cout << "y_max: " << y_max << std::endl;
                      std::cout << "z_max: " << z_max << std::endl;
                      std::cout << "mx: " << mx << std::endl;
                      std::cout << "my: " << my << std::endl;
                      std::cout << "median_depth: " << median_depth << std::endl;

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
                
                  // final check here 
                  // only add to message if no nans exist
                  if (check_detection_msg(detection_msg)){
                    std::cout << "valid detection!" << std::endl;
                    detection_msg.object_name=object_name;            
                    detection_array_msg->detections.push_back(detection_msg);
                
                  cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
                  cv::putText(cv_image_clone, object_name, cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
                  // cv::imwrite("/home/nvidia/OUTPUTIMAGE.JPG", cv_image);
                }
              }
            }
          }
        }
      }
    
    // this will publish empty detections if nothing is found
    sensor_msgs::ImagePtr imagemsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_clone).toImageMsg();
    detections_pub.publish(detection_array_msg);
    //skeleton_pub.publish(skeleton_array);
    image_pub.publish(imagemsg);
    free(output->boxes);
    free(output);
    }  
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
  double max_distance;
  bool use_pointcloud;
  int centroid_arg; //0, 1, 2, 3
  int cluster_mode; //0, 1, 2, 3
  bool pointcloud_only;
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
  pnh.param("max_distance", max_distance, 6.25);
  pnh.param("use_pointcloud", use_pointcloud, false);
  pnh.param("centroid_arg", centroid_arg, 0);
  pnh.param("cluster_mode", cluster_mode, 0);
  pnh.param("pointcloud_only", pointcloud_only, true);
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  TVMDetectionNode node(nh, sensor_name, zone_json, max_distance);
  std::cout << "detection node init " << std::endl;
  ros::spin();
  return 0;
}
