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

#include <open_ptrack/yolo_tvm.hpp>
#include <dynamic_reconfigure/server.h>
#include <recognition/GenDetectionConfig.h>

// not sure if this is the correct json reading code
// but will be easier than continually recompiling t
// import header files
#include <nlohmann/json.hpp>
using json = nlohmann::json;

typedef sensor_msgs::Image Image;
typedef sensor_msgs::CameraInfo CameraInfo;
using namespace message_filters::sync_policies;

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

/**
 * @brief The TVMDetectionNode
 */
class TVMDetectionNode {
  private:
    ros::NodeHandle node_;
    //std::unique_ptr<YoloTVMGPU256> tvm_object_detector;
    std::unique_ptr<YoloTVMGPU> tvm_object_detector;
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
    TVMDetectionNode(ros::NodeHandle& nh, std::string sensor_string):
      node_(nh), it(node_)
      {
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
        approximate_sync_->registerCallback(boost::bind(&TVMDetectionNode::callback, this, _1, _2));

        // create callback config 
        cfg_server.setCallback(boost::bind(&TVMDetectionNode::cfg_callback, this, _1, _2));      

        // create object-detector pointer
        //tvm_object_detector.reset(new YoloTVMGPU256(model_folder_path));
        tvm_object_detector.reset(new YoloTVMGPU(model_folder_path));
        sensor_name = sensor_string;

        try
        {
          json zone_json;
          std::string area_package_path = ros::package::getPath("recognition");
          std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
          std::ifstream area_json_read(area_hard_coded_path);
          area_json_read >> zone_json;
          
          // get the number of zones to scan.
          json master_config;
          std::string master_package_path = ros::package::getPath("recognition");
          std::string master_hard_coded_path = master_package_path + "/cfg/master.json";
          std::ifstream master_json_read(master_hard_coded_path);
          master_json_read >> master_config;
          n_zones = master_config["n_zones"]; //the path to the detector model file
          json_found = true;
        }
        catch(const std::exception& e)
        {
          std::cerr << "json master/area not found: "<< e.what() << '\n';
        }
         

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
     */
    void callback(const sensor_msgs::Image::ConstPtr& rgb_image,
                  const sensor_msgs::Image::ConstPtr& depth_image) {
        
      std::cout << "running algorithm callback" << std::endl;
    
      //tf_listener.waitForTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
      //tf_listener.lookupTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ir2rgb_transform);
      //tf_listener.waitForTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
      //tf_listener.lookupTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), world2rgb_transform);

      // transform to eigen
      //tf::transformTFToEigen(world2rgb_transform, world2rgb);
      //tf::transformTFToEigen(ir2rgb_transform, ir2rgb);




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
                for (zone_id = 0; zone_id < n_zones; zone_id++)
                {
                  // need a world view here bc each detection was transformed
                  // this will work for a singular cam, but would mean each cam would have to tune
                  // to the specific area; which I think would be fine. // but will need
                  // to test to be sure
                  // a given detection can be in only one place at one time, thus it can't be in
                  // multiple zones
                  double x_min = zone_json[zone_id][sensor_name]["min"][sensor_name]["x"];
                  double y_min = zone_json[zone_id][sensor_name]["min"][sensor_name]["y"];
                  double z_min = zone_json[zone_id][sensor_name]["min"][sensor_name]["z"];
                  double x_max = zone_json[zone_id][sensor_name]["max"][sensor_name]["x"];
                  double y_max = zone_json[zone_id][sensor_name]["max"][sensor_name]["y"];
                  double z_max = zone_json[zone_id][sensor_name]["max"][sensor_name]["z"];
                  inside_area_cube = (mx <= x_max && mx >= x_min) && (my <= y_max && my >= y_min) && (median_depth <= z_max && median_depth >= z_min);
                  // I think this works. 
                  if (inside_area_cube) {
                    break;
                  }
                }
                if (inside_area_cube) {
                  detection_msg.zone_id = zone_id;
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
  std::string sensor_name;
  json master_config;
  std::string package_path = ros::package::getPath("recognition");
  std::string master_hard_coded_path = package_path + "/cfg/master.json";
  std::ifstream json_read(master_hard_coded_path);
  json_read >> master_config;
  sensor_name = master_config["sensor_name"]; //the path to the detector model file

  std::cout << "--- tvm_detection_node ---" << std::endl;
  ros::init(argc, argv, "tvm_detection_node");
  // something is off here... with the private namespace
  ros::NodeHandle nh;
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  TVMDetectionNode node(nh, sensor_name);
  std::cout << "detection node init " << std::endl;
  ros::spin();
  return 0;
}