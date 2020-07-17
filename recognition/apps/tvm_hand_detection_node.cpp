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
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <pcl/filters/statistical_outlier_removal.h>

// not sure if this is the correct json reading code
// but will be easier than continually recompiling t
// import header files
#include <nlohmann/json.hpp>
using json = nlohmann::json;

typedef sensor_msgs::Image Image;
typedef sensor_msgs::CameraInfo CameraInfo;
using namespace message_filters::sync_policies;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef boost::shared_ptr<PointCloud> PointCloudPtr;
typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;
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
 * @brief The TVMHandDetectionNode
 */
class TVMHandDetectionNode {
  private:
    ros::NodeHandle node_;
    //std::unique_ptr<YoloTVMGPU256> tvm_object_detector;
    //std::unique_ptr<YoloTVMGPU320> tvm_object_detector;
    std::unique_ptr<NoNMSYoloFromConfig> tvm_object_detector;

    // TF listener
    tf::TransformListener tf_listener;
    // only need this if I need to debug
    //image_transport::ImageTransport it;
    
    // ROS
    //dynamic_reconfigure::Server<recognition::HandDetectionConfig> cfg_server;
    ros::ServiceServer camera_info_matrix_server;
    dynamic_reconfigure::Server<recognition::GenDetectionConfig> cfg_server;


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
    ros::Subscriber point_cloud_approximate_sync_;

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
    json zone_json;
    bool json_found;
    int n_zones;
    float override_threshold = 0.5;
    float max_capable_depth = 6.25;
    bool use_pointcloud;


    double rate_value = 1.0;
  public:
    // Set camera matrix transforms
    Eigen::Matrix3f intrinsics_matrix;
    bool camera_info_available_flag = false;
    double _cx;
    double _cy;
    double _constant_x;
    double _constant_y;
    image_transport::ImageTransport it;
    // Image to "world" transforms
    Eigen::Affine3d world2rgb;
    tf::StampedTransform world2rgb_transform;
    tf::StampedTransform world_transform;
    tf::StampedTransform world_inverse_transform;
    /**
     * @brief constructor
     * @param nh node handler
     */
    TVMHandDetectionNode(ros::NodeHandle& nh, std::string sensor_string, bool use_dynamic_reconfigure):
      node_(nh), it(node_)
      {
        
        try
        {
          json master_config;
          std::string package_path = ros::package::getPath("recognition");
          std::string master_hard_coded_path = package_path + "/cfg/hand_detector.json";
          std::ifstream master_json_read(master_hard_coded_path);
          master_json_read >> master_config;
          n_zones = master_config["n_zones"]; //the path to the detector model file
          max_capable_depth = master_config["max_capable_depth"];
          use_pointcloud = master_config["use_pointcloud"];
          std::cout << "max_capable_depth: " << max_capable_depth << std::endl;
          override_threshold = master_config["override_threshold"];
          std::string zone_json_path = master_config["zone_json_path"];
          std::string area_hard_coded_path = package_path + zone_json_path;
          std::ifstream area_json_read(area_hard_coded_path);
          area_json_read >> zone_json;
          json_found = true;
        }
        catch(const std::exception& e)
        {
          std::cerr << "json master/area not found: "<< e.what() << '\n';
        }        
        
        // Publish Messages
        detections_pub = node_.advertise<opt_msgs::DetectionArray>("/hand_detector/detections", 3);

        // Subscribe to Messages
        if (use_pointcloud){
          point_cloud_approximate_sync_ = node_.subscribe(sensor_string + "/depth_registered/points", 1, &TVMHandDetectionNode::cloud_callback, this);
        } else {
          rgb_image_sub.subscribe(node_, sensor_string +"/color/image_rect_color", 1);
          depth_image_sub.subscribe(node_, sensor_string+"/depth/image_rect_raw", 1);
          //Time sync policies for the subscribers
          approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(10), rgb_image_sub, depth_image_sub));
          approximate_sync_->registerCallback(boost::bind(&TVMHandDetectionNode::callback, this, _1, _2));
        }
        
        image_pub = it.advertise(sensor_string + "/hand_detector/image", 1);

        // Camera callback for intrinsics matrix update
        camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &TVMHandDetectionNode::camera_info_callback, this);

        // create callback config 
        if (use_dynamic_reconfigure){
          cfg_server.setCallback(boost::bind(&TVMHandDetectionNode::cfg_callback, this, _1, _2));      
        }

        // create object-detector pointer
        //tvm_object_detector.reset(new YoloTVMGPU256(model_folder_path));
        tvm_object_detector.reset(new NoNMSYoloFromConfig("/cfg/hand_detector.json", "recognition"));
        sensor_name = sensor_string;
        std::cout << "detector loaded!" << std::endl;

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
      //Calculate direct and inverse transforms between camera and world frame:
      tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
                                  world_transform);
      tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
                                  world_inverse_transform);
      // set message vars here
      cv_bridge::CvImagePtr cv_ptr_rgb;
      cv_bridge::CvImage::Ptr  cv_ptr_depth;
      cv::Mat cv_image;
      cv::Mat cv_depth_image;
      
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
      cv::Mat cv_image_clone;

      // forward inference of object detector
      begin = ros::Time::now();
      output = tvm_object_detector->forward_full(cv_image, override_threshold);
      duration = ros::Time::now().toSec() - begin.toSec();
      printf("yolo detection time: %f\n", duration);
      printf("yolo detections: %ld\n", output->num);

      if (output->num >= 1) {
        for (int i = 0; i < output->num; i++) {
          // get the label and the object name
          float label = static_cast<float>(output->boxes[i].id);
          //std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];

          // get the coordinate information
          float xmin = output->boxes[i].xmin;
          float ymin = output->boxes[i].ymin;
          float xmax = output->boxes[i].xmax;
          float ymax = output->boxes[i].ymax;
          float score = output->boxes[i].score;

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
            
            tf::Vector3 point_3D(mx, my, median_depth);
            
            world_transformed_point = world_inverse_transform(point_3D);

            opt_msgs::Detection detection_msg;
            detection_msg.box_3D.p1.x = world_transformed_point.x;
            detection_msg.box_3D.p1.y = world_transformed_point.y;
            detection_msg.box_3D.p1.z = world_transformed_point.z;
            
            detection_msg.box_3D.p2.x = mx;
            detection_msg.box_3D.p2.y = my;
            detection_msg.box_3D.p2.z = median_depth;
            
            detection_msg.box_2D.x = median_x;
            detection_msg.box_2D.y = median_y;
            detection_msg.box_2D.width = 0;
            detection_msg.box_2D.height = 0;
            detection_msg.height = 0;
            detection_msg.confidence = score;
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
                
                min_point = world_inverse_transform(min_point);
                max_point = world_inverse_transform(max_point);

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

            detection_msg.object_name="hand";              
            detection_array_msg->detections.push_back(detection_msg);

            // sensor_msgs::ImagePtr image_msg_aligned = cv_bridge::CvImage(std_msgs::Header(), "bgr8", aligned_clone).toImageMsg();
            cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
            cv::putText(cv_image_clone, "hand", cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
            // cv::imwrite("/home/nvidia/OUTPUTIMAGE.JPG", cv_image);

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


    /**
     * @brief callback for camera information that does detection on images
     *  and publishes the detections to specific topics
     * @param rgb_image  the rgb image message
     * @param depth_image  the depth/stereo image message
     */
    void cloud_callback(const PointCloudT::ConstPtr& cloud_) {
        
      std::cout << "running algorithm callback" << std::endl;
      //Calculate direct and inverse transforms between camera and world frame:
      tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
                                  world_transform);
      tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
                                  world_inverse_transform);
      // set message vars here
      //cv_bridge::CvImagePtr cv_ptr_rgb;
      //cv_bridge::CvImage::Ptr  cv_ptr_depth;
      //cv::Mat cv_image;
      //cv::Mat cv_depth_image;
      
      // set message vars here
      //open_ptrack::opt_utils::Conversions converter; 
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

      cv_image_clone = cv_image.clone();
      image_size = cv_image.size();
      height = static_cast<float>(image_size.height);
      width = static_cast<float>(image_size.width);

      // forward inference of object detector
      begin = ros::Time::now();
      output = tvm_object_detector->forward_full(cv_image, override_threshold);
      duration = ros::Time::now().toSec() - begin.toSec();
      printf("yolo detection time: %f\n", duration);
      printf("yolo detections: %ld\n", output->num);

      if (output->num >= 1) {
        for (int i = 0; i < output->num; i++) {
          // get the label and the object name
          float label = static_cast<float>(output->boxes[i].id);
          //std::string object_name = COCO_CLASS_NAMES[output->boxes[i].id];

          // get the coordinate information
          float xmin = output->boxes[i].xmin;
          float ymin = output->boxes[i].ymin;
          float xmax = output->boxes[i].xmax;
          float ymax = output->boxes[i].ymax;
          float score = output->boxes[i].score;

          // set the median of the bounding box
          float median_x = xmin + ((xmax - xmin) / 2.0);
          float median_y = ymin + ((ymax - ymin) / 2.0);

          // If the detect box coordinat is near edge of image, it will return a error 'Out of im.size().'
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
        
          // TODO we'll have to vet this strategy but for now, we're trying it
          // this will get the median depth of the object at the center of the object,
          // however, we're moving towards the median depth as where the box hits the ground/
          // where the feet are..
          //float median_depth = cv_depth_image.at<float>(median_y, median_x) / 1000.0f;
          //float median_depth = cv_depth_image.at<float>(new_y, median_x) / mm_factor;
          float mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
          float my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
          float median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

          if (median_depth <= 0 || median_depth > 6.25) {
            std::cout << "median_depth " << median_depth << " rejecting" << std::endl;
            continue;
            }			
            
          // set the mx/my wtr the intrinsic camera matrix
          //float mx = (median_x - _cx) * median_depth * _constant_x;
          //float my = (median_y - _cy) * median_depth * _constant_y;

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
            detection_msg.confidence = score;
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
                
                min_point = world_inverse_transform(min_point);
                max_point = world_inverse_transform(max_point);

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

            detection_msg.object_name="hand";            
            detection_array_msg->detections.push_back(detection_msg);

            // sensor_msgs::ImagePtr image_msg_aligned = cv_bridge::CvImage(std_msgs::Header(), "bgr8", aligned_clone).toImageMsg();
            cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
            cv::putText(cv_image_clone, "hand", cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
            // cv::imwrite("/home/nvidia/OUTPUTIMAGE.JPG", cv_image);

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
      //std::string package_path = ros::package::getPath("recognition");
      std::cout << "Updating detector configuration!!!" << std::endl;
      max_capable_depth = config.max_capable_depth;
      override_threshold = config.override_threshold;
    }
};

int main(int argc, char** argv) {
  // read json to get the master config file
  std::string sensor_name;
  bool use_dynamic_reconfigure;
  //json master_config;
  //std::string package_path = ros::package::getPath("recognition");
  //std::string master_hard_coded_path = package_path + "/cfg/master.json";
  //std::ifstream json_read(master_hard_coded_path);
  //json_read >> master_config;
  //sensor_name = master_config["sensor_name"]; //the path to the detector model file


  std::cout << "--- tvm_hand_detection_node ---" << std::endl;
  ros::init(argc, argv, "tvm_hand_detection_node");
  // something is off here... with the private namespace
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;
  pnh.param("sensor_name", sensor_name, std::string("d435"));
  pnh.param("use_dynamic_reconfigure", use_dynamic_reconfigure, false);
  // odd private v. public structure. lame.
  // one config, no need for multiple config layers
  //nh.getParam("sensor_name", sensor_name);
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  TVMHandDetectionNode node(nh, sensor_name, use_dynamic_reconfigure);
  std::cout << "TVMHandDetectionNode init " << std::endl;
  ros::spin();
  return 0;
}