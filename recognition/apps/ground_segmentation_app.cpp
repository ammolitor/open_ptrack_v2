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
#include <opt_msgs/GroundCoeffs.h>
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
#include <nlohmann/json.hpp>
#include <recognition/GenDetectionConfig.h>

/// yolo specific args
//#include <open_ptrack/tvm_detection_helpers.hpp>
#include <pcl/visualization/pcl_visualizer.h>

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


/** \brief BaseNode estimates the ground plane equation from a 3D point cloud */
class GroundEstimationNode {
  private:
    // Publishers
    ros::Publisher coeffs_pub;
    ros::Publisher skeleton_pub;
    image_transport::Publisher image_pub;
    ros::NodeHandle node_;
    ros::Subscriber point_cloud_approximate_sync_;
    image_transport::ImageTransport it;
    ros::ServiceServer camera_info_matrix_server;
    ros::Subscriber camera_info_matrix;
    dynamic_reconfigure::Server<recognition::GenDetectionConfig> cfg_server;

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
    float valid_points_threshold = 0.0;
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
    float override_threshold = 0.5;
    float nms_threshold = 0.5;
    bool fast_no_clustering = false;
    bool view_pointcloud = false;
    bool ground_from_extrinsic_calibration = false;

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
    bool filter_height = false; // var is auto set if you're not using head clusters
    //###################################
    //## Detection Variables ##
    //###################################
    float thresh = 0.3f;
    int gluon_to_rtpose[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
    bool use_pose_model = false;
    bool use_3D_clusters = false;
    //###################################
    //## Transform Listeners ##
    //###################################
    Eigen::Affine3d world2rgb;
    tf::StampedTransform world2rgb_transform;
    tf::StampedTransform world_transform;
    tf::StampedTransform world_inverse_transform;
    pcl::visualization::PCLVisualizer viewer = pcl::visualization::PCLVisualizer ("3D Viewer");

    GroundEstimationNode(ros::NodeHandle& nh, std::string sensor_string):
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
          use_3D_clusters = master_config["use_3D_clusters"];
          override_threshold = master_config["override_threshold"];
          nms_threshold = master_config["nms_threshold"];
          filter_height = master_config["filter_height"];
          fast_no_clustering = master_config["fast_no_clustering"];
          std::cout << "Fast Mode: " << fast_no_clustering << std::endl;
          view_pointcloud = master_config["view_pointcloud"];
          std::cout << "view_pointcloud: " << view_pointcloud << std::endl;
          ground_from_extrinsic_calibration = master_config["ground_from_extrinsic_calibration"];
          json_found = true;
        }
        catch(const std::exception& e)
        {
          std::cerr << "json master/area not found: "<< e.what() << '\n';
        }
        coeffs_pub = node_.advertise<opt_msgs::GroundCoeffs>(sensor_string + "/ground_coeffs/", 1);
        
        // Camera callback for intrinsics matrix update
        camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &GroundEstimationNode::camera_info_callback, this);

        point_cloud_approximate_sync_ = node_.subscribe(sensor_string + "/depth_registered/points", 1, &GroundEstimationNode::ground_plane_callback, this);
        // maybe...
        transform = transform.Identity();
        anti_transform = transform.inverse();
        sensor_name = sensor_string;
        rgb_image_ = pcl::PointCloud<pcl::RGB>::Ptr(new pcl::PointCloud<pcl::RGB>);
        // reset here after vars have been called...
        ground_estimator = open_ptrack::ground_segmentation::GroundplaneEstimation<PointT>(ground_estimation_mode, remote_ground_selection);
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

      //std::cout << "saving background file to tmp space: " << std::endl;
      pcl::io::savePCDFileASCII ("/tmp/background_" + sensor_name + ".pcd", *background_cloud);
      //std::cout << "background cloud done." << std::endl << std::endl;
    }

    void set_octree(){
      // setting octree
      background_octree_ = new pcl::octree::OctreePointCloud<PointT>(background_octree_resolution);
      background_octree_->defineBoundingBox(-max_distance/2, -max_distance/2, 0.0, max_distance/2, max_distance/2, max_distance);
      background_octree_->setInputCloud (background_cloud);
      background_octree_->addPointsFromInputCloud ();
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
    void compute_subclustering(PointCloudPtr no_ground_cloud, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters, bool compute_height){
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
          
          float height = it->getHeight();
          if (compute_height && height > min_height_ && height < max_height_){
            // only take blobs that are person sized.
            continue;
          }
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
      //std::cout << "[create_foreground_cloud] cloud_filtered size: " << cloud_filtered->size() << std::endl;
      //std::cout << "[create_foreground_cloud] cloud_filtered height: " << cloud_filtered->height << std::endl;


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
      //std::cout << "[create_foreground_cloud] no_ground_cloud_ size: " << no_ground_cloud_->size() << std::endl;
      //std::cout << "[create_foreground_cloud] no_ground_cloud_ height: " << no_ground_cloud_->height << std::endl;
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
        //std::cout << "[create_foreground_cloud::background_subtraction] foreground_cloud: " << foreground_cloud->size() << std::endl;
        //std::cout << "[create_foreground_cloud::background_subtraction] no_ground_cloud_: " << no_ground_cloud_->size() << std::endl;
      }
      
      //std::cout << "[create_foreground_cloud] no_ground_cloud_ post-background subtraction: " << no_ground_cloud_->size() << std::endl;
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
      compute_subclustering(no_ground_cloud_, clusters, filter_height);
      //std::cout << "[create_foreground_cloud] no_ground_cloud_ post compute_subclustering: " << no_ground_cloud_->size() << std::endl;
      if (use_headclusters){
        //std::cout << ground_coeffs << std::endl;
        //std::cout  << ground_coeffs_new << std::endl; // not being set.. why the f?;
        compute_head_subclustering(no_ground_cloud_, clusters);
      }
      //std::cout << "create_foreground_cloud - cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
      //std::cout << "create_foreground_cloud - cluster_centroids3d size: " << cluster_centroids3d.size() << std::endl;
      // Sensor tilt compensation to improve people detection:
      // moving to global PointCloudPtr no_ground_cloud_rotated(new PointCloud);
      // moving to global Eigen::VectorXf ground_coeffs_new;
      //ground_coeffs
      PointCloudPtr no_ground_cloud_rotated_pre(new PointCloud);
      Eigen::VectorXf ground_coeffs_;
      Eigen::VectorXf ground_coeffs_new_pre;
      Eigen::Affine3f transform, transform_pre, anti_transform_pre;
      if(sensor_tilt_compensation)
      {
        // We want to rotate the point cloud so that the ground plane is parallel to the xOz plane of the sensor:
        Eigen::Vector3f input_plane, output_plane;
        input_plane << ground_coeffs(0), ground_coeffs(1), ground_coeffs(2);
        output_plane << 0.0, -1.0, 0.0;

        Eigen::Vector3f axis = input_plane.cross(output_plane);
        float angle = acos( input_plane.dot(output_plane)/ ( input_plane.norm()/output_plane.norm() ) );
        transform_pre = Eigen::AngleAxisf(angle, axis);

        // Setting also anti_transform for later
        anti_transform_pre = transform_pre.inverse();
        no_ground_cloud_rotated_pre = rotate_cloud(no_ground_cloud_, transform_pre);
        ground_coeffs_new_pre.resize(4);
        ground_coeffs_new_pre = rotate_ground(ground_coeffs, transform_pre);
      }
      else
      {
        transform_ = transform_.Identity();
        anti_transform_pre = transform_.inverse();
        no_ground_cloud_rotated_pre = no_ground_cloud_;
        ground_coeffs_new_pre = ground_coeffs;
      }
      // now do them at the global level...
      anti_transform_ = anti_transform_pre;
      transform_ = transform_pre;
      ground_coeffs_new = ground_coeffs_new_pre;
      no_ground_cloud_rotated = no_ground_cloud_rotated_pre;
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
        std::cout << "Ground plane initialization starting..." << std::endl;
        ground_estimator.setInputCloud(cloud);
        //Eigen::VectorXf ground_coeffs = ground_estimator.computeMulticamera(ground_from_extrinsic_calibration, read_ground_from_file,
        //    pointcloud_topic, sampling_factor, voxel_size);
        ground_coeffs = ground_estimator.computeMulticamera(ground_from_extrinsic_calibration, false, sensor_name + "/depth_registered/points", 4, 0.06);
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
    void compute_head_subclustering(PointCloudPtr no_ground_cloud, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){

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
      //std::cout << "[compute_head_subclustering] no_ground_cloud check: " << no_ground_cloud->size() << std::endl;
      subclustering.setInputCloud(no_ground_cloud); //no_ground_cloud_rotated
      //std::cout << "setInputCloud finished" << std::endl;
      subclustering.setGround(ground_coeffs); //ground_coeffs_new
      //std::cout << "[compute_head_subclustering] ground_coeffs check: " << ground_coeffs << std::endl;
      //std::cout << "setGround finished" << std::endl;
      subclustering.setInitialClusters(cluster_indices);
      //std::cout << "setInitialClusters finished" << std::endl;
      subclustering.setHeightLimits(min_height_, max_height_);
      //std::cout << "setHeightLimits finished" << std::endl;
      subclustering.setMinimumDistanceBetweenHeads(heads_minimum_distance_);
      //std::cout << "setMinimumDistanceBetweenHeads finished" << std::endl;
      subclustering.setSensorPortraitOrientation(vertical_);
      //std::cout << "setSensorPortraitOrientation finished" << std::endl;
      subclustering.subcluster(clusters);
      //std::cout << "subcluster finished" << std::endl;
      //std::cout << "clusters size: " << clusters.size() << std::endl;

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



    // this is going to be for non-pcl-clusters
    void drawTBoundingBox (pcl::visualization::PCLVisualizer& viewer, Eigen::Vector3f tcenter_, Eigen::Vector3f ttop_, float  height_, float person_confidence_, int person_number, bool add_info)
    {
      // draw theoretical person bounding box in the PCL viewer:
      pcl::ModelCoefficients coeffs;
      // translation
      coeffs.values.push_back (tcenter_[0]);
      coeffs.values.push_back (tcenter_[1]);
      coeffs.values.push_back (tcenter_[2]);
      // rotation
      coeffs.values.push_back (0.0);
      coeffs.values.push_back (0.0);
      coeffs.values.push_back (0.0);
      coeffs.values.push_back (1.0);
      // size
      if (vertical_)
      {
        coeffs.values.push_back (height_);
        coeffs.values.push_back (0.5);
        coeffs.values.push_back (0.5);
      }
      else
      {
        coeffs.values.push_back (0.5);
        coeffs.values.push_back (height_);
        coeffs.values.push_back (0.5);
      }

      std::stringstream bbox_name;
      bbox_name << "bbox_person_" << person_number;
      viewer.removeShape (bbox_name.str());
      viewer.addCube (coeffs, bbox_name.str());
      viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, bbox_name.str());
      viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, bbox_name.str());
      
      if (add_info){
        std::stringstream confid;
        confid << person_confidence_;
        PointT position;
        position.x = tcenter_[0]- 0.2;
        position.y = ttop_[1];
        position.z = tcenter_[2];
        viewer.addText3D(confid.str().substr(0, 4), position, 0.1);
      }
      //std::cout << "debug stats:" << std::endl;
      //std::cout << "tcenter_: " << tcenter_ << std::endl;
      //std::cout << "height_: " << height_ << std::endl;
      //std::cout << "ttop_: " << ttop_ << std::endl;

    }

    void point_cloud_visulizer (const PointCloudT::ConstPtr& cloud_, Eigen::VectorXf& ground_coeffs_ ,pcl::visualization::PCLVisualizer& viewer, std::vector<open_ptrack::person_clustering::PersonCluster<PointT>>& clusters, std::vector<int> viz_indicies)
    {
      pcl::ModelCoefficients::Ptr plane_ (new pcl::ModelCoefficients); 
      plane_->values.resize (10); 
      plane_->values[0] = ground_coeffs_(0); 
      plane_->values[1] = ground_coeffs_(1); 
      plane_->values[2] = ground_coeffs_(2); 
      plane_->values[3] = ground_coeffs_(3); 

      viewer.addPlane (*plane_, "plane_", 0); 
      viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.9, 0.1, 0.1 /*R,G,B*/, "plane_", 0); 
      viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.6, "plane_", 0); 
      viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "plane_", 0); 

      int off_set = 20;
      
      for (size_t i = 0; i < viz_indicies.size(); i++)
      {
        int x = viz_indicies[i];
        open_ptrack::person_clustering::PersonCluster<PointT> person_cluster = clusters[x];
        pcl::ModelCoefficients::Ptr sphere_ (new pcl::ModelCoefficients); 
        sphere_->values.resize (1); 
        sphere_->values[0] = person_cluster.getTCenter()(0); 
        sphere_->values[1] = person_cluster.getTCenter()(1); 
        sphere_->values[2] = person_cluster.getTCenter()(2); 
        sphere_->values[3] = 0.05; 

        viewer.addSphere (*sphere_, "sphere_", 0); 
        viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.9, 0.1, 0.1 /*R,G,B*/, "sphere_", 0); 
        viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.6, "sphere_", 0); 
        viewer.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "sphere_", 0); 

        person_cluster.drawTBoundingBox(viewer, 1);

        std::string f_str = "PersonConfidence : " + std::to_string(person_cluster.getPersonConfidence());
        viewer.addText(f_str,off_set,20,f_str,0);

        //Evaluate confidence for the current PersonCluster:
        Eigen::Vector3f centroid = intrinsics_matrix * (anti_transform_ * person_cluster.getTCenter());
        centroid /= centroid(2);
        Eigen::Vector3f top = intrinsics_matrix * (anti_transform_ * person_cluster.getTTop());
        top /= top(2);
        Eigen::Vector3f bottom = intrinsics_matrix * (anti_transform_ * person_cluster.getTBottom());
        bottom /= bottom(2);

        // Eigen::Vector3f centroid = person_cluster.getTCenter();
        // // centroid /= centroid(2);
        // Eigen::Vector3f top = person_cluster.getTTop();
        // // top /= top(2);
        // Eigen::Vector3f bottom = person_cluster.getTBottom();
        // // bottom /= bottom(2);


        float pixel_height;
        float pixel_width;

        if (!vertical_)
        {
          pixel_height = bottom(1) - top(1);
          pixel_width = pixel_height / 2.0f;
          std::string f_str_1 = "person_height : " + std::to_string(fabs(pixel_height));
          viewer.addText(f_str_1,off_set,40,f_str_1,0);
        }
        else
        {
          pixel_width = top(0) - bottom(0);
          pixel_height = pixel_width / 2.0f;
          std::string f_str_1 = "person_height : " + std::to_string(fabs(pixel_width));
          viewer.addText(f_str_1,off_set,40,f_str_1,0);
        }
        off_set = off_set + 145;
      }
  
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_);
      viewer.addPointCloud<PointT> (cloud_, rgb, "temp_cloud");
      
      viewer.addCoordinateSystem (0.5, "axis", 0); 
      viewer.setBackgroundColor (0, 0, 0, 0); 
      viewer.setPosition (800, 400); 
      viewer.setCameraPosition(-9, 0, -5,     10, 0, 5,     0, -1, 0,      0);
      viewer.spinOnce ();
      viewer.removeAllShapes();
      viewer.removeAllPointClouds();
    }

    void ground_plane_callback(const PointCloudT::ConstPtr& cloud_) {

      //Calculate direct and inverse transforms between camera and world frame:
      //tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
      //                            world_transform);
      //tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
      //                            world_inverse_transform);

      //std::cout << "running algorithm callback" << std::endl;
      //ros::Time start = ros::Time::now();
      if (setbackground){
        //std::cout << "background frame n: " << n_frame << std::endl;
        PointCloudT::Ptr newcloud(new PointCloudT);
        *newcloud = *cloud_;
        if (!ground_estimator.tooManyNaN(newcloud, 1 - valid_points_threshold)){
          background_cloud = compute_background_cloud(newcloud);
          if (n_frame >= n_frames){
            set_background(background_cloud);
            set_octree();
            setbackground = false;
          }
        }
      } else { 
        // background is set
        // estimate ground plane and continue with detection process
        if (estimate_ground_plane) {
          set_ground_variables(cloud_);
          estimate_ground_plane = false;
          std::cout << "DEBUG sqrt_ground_coeffs : " << sqrt_ground_coeffs << std::endl;
          std::cout << "DEBUG ground_coeffs : " <<ground_coeffs << std::endl;
        }
      
        if (!estimate_ground_plane){
          opt_msgs::GroundCoeffs::Ptr coeffs_msg(new opt_msgs::GroundCoeffs);
          coeffs_msg->ground_coeffs.x = ground_coeffs(0);
          coeffs_msg->ground_coeffs.y = ground_coeffs(1);
          coeffs_msg->ground_coeffs.z = ground_coeffs(2);
          coeffs_msg->ground_coeffs.w = ground_coeffs(3);
          coeffs_pub.publish(coeffs_msg);
          //std::cout << "press cntrl+c anytime to exit" << std::endl;
        }
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
      //std::string package_path = ros::package::getPath("recognition");
      std::cout << "Updating detector configuration!!!" << std::endl;
      max_capable_depth = config.max_capable_depth;
      std::cout << "max_capable_depth: " << max_capable_depth << std::endl;
      ground_estimation_mode = config.ground_estimation_mode;
      std::cout << "ground_estimation_mode: " << ground_estimation_mode << std::endl;
      remote_ground_selection = config.remote_ground_selection;
      std::cout << "remote_ground_selection: " << remote_ground_selection << std::endl;
      read_ground_from_file = config.read_ground_from_file;
      std::cout << "read_ground_from_file: " << read_ground_from_file << std::endl; 
      lock_ground = config.lock_ground; 
      std::cout << "lock_ground: " << lock_ground << std::endl;
      valid_points_threshold = config.valid_points_threshold;
      std::cout << "valid_points_threshold: " << valid_points_threshold << std::endl;
      background_subtraction = config.background_subtraction;
      std::cout << "background_subtraction: " << background_subtraction << std::endl; 
      background_octree_resolution = config.background_octree_resolution;
      std::cout << "background_octree_resolution: " << background_octree_resolution << std::endl;  
      background_seconds = config.background_seconds; 
      std::cout << "background_seconds: " << background_seconds << std::endl;
      ground_based_people_detection_min_confidence = config.ground_based_people_detection_min_confidence; 
      std::cout << "ground_based_people_detection_min_confidence: " << ground_based_people_detection_min_confidence << std::endl;
      minimum_person_height = config.minimum_person_height;
      std::cout << "minimum_person_height: " << minimum_person_height << std::endl; 
      maximum_person_height = config.maximum_person_height;
      std::cout << "maximum_person_height: " << maximum_person_height << std::endl; 
      sampling_factor = config.sampling_factor;
      std::cout << "sampling_factor: " << sampling_factor << std::endl; 
      use_rgb = config.use_rgb;
      std::cout << "use_rgb: " << use_rgb << std::endl; 
      minimum_luminance = config.minimum_luminance;
      std::cout << "minimum_luminance: " << minimum_luminance << std::endl; 
      sensor_tilt_compensation = config.sensor_tilt_compensation;
      std::cout << "sensor_tilt_compensation: " << sensor_tilt_compensation << std::endl; 
      heads_minimum_distance = config.heads_minimum_distance; 
      std::cout << "heads_minimum_distance: " << heads_minimum_distance << std::endl;
      voxel_size = config.voxel_size; 
      std::cout << "voxel_size: " << voxel_size << std::endl;
      apply_denoising = config.apply_denoising; 
      std::cout << "apply_denoising: " << apply_denoising << std::endl;
      std_dev_denoising = config.std_dev_denoising;
      std::cout << "std_dev_denoising: " << std_dev_denoising << std::endl;
      rate_value = config.rate_value;
      std::cout << "rate_value: " << rate_value << std::endl;
      // can only turn on when first starting... can't redeclare a callback
      view_pointcloud = config.view_pointcloud;
      //std::cout << "Fast Mode: " << fast_no_clustering << std::endl;

      override_threshold = config.override_threshold;
      nms_threshold = config.nms_threshold;
    }
};

int main(int argc, char** argv) {
  std::string sensor_name;
  //double max_distance;
  //json zone_json;
  bool use_dynamic_reconfigure;
  //std::string area_package_path = ros::package::getPath("recognition");
  //std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
  //std::ifstream area_json_read(area_hard_coded_path);
  //area_json_read >> zone_json;

  std::cout << "--- tvm_detection_node ---" << std::endl;
  ros::init(argc, argv, "tvm_detection_node");
  // something is off here... with the private namespace
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;
  pnh.param("sensor_name", sensor_name, std::string("d435"));
  pnh.param("use_dynamic_reconfigure", use_dynamic_reconfigure, false);
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  GroundEstimationNode node(nh, sensor_name);
  std::cout << "GroundEstimationNode init " << std::endl;
  ros::spin();
  return 0;
}
