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
#include <ros/console.h>
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
#include <opt_msgs/GroundCoeffs.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
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
#include <recognition/GenDetectionConfig.h>
#include <pcl/common/colors.h>
#include <regex> 
/// yolo specific args
//#include <open_ptrack/tvm_detection_helpers.hpp>
//#include <open_ptrack/NoNMSPoseFromConfig.hpp>
//#include <open_ptrack/NoNMSYoloFromConfig.hpp>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>

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

// change as needed...
#define MAX_NSENSORS 100

Eigen::Matrix4d
readMatrixFromFile (std::string filename)
{
  Eigen::Matrix4d matrix;
  std::string line;
  std::ifstream myfile (filename.c_str());
  if (myfile.is_open())
  {
    int k = 0;
    std::string number;
    while (myfile >> number)
    {
      matrix(int(k/4), int(k%4)) = std::atof(number.c_str());
      k++;
    }
    myfile.close();
  }

  std::cout << matrix << std::endl;

  return matrix;
}

struct CalibrationData {
    std::string       frame_id;
    //CameraIntrinsics  intrinsics;
    std::vector<double> distortion_coeffs;
    std::vector<double> intrinsics; // K Matrix
    std::vector<int>    resolution;
    std::string distortion_model;
    std::string camera_model;
    // extrinsics
    Eigen::Matrix<double, 4, 4>  T_cam_imu;
    Eigen::Matrix<double, 4, 4>  T_cn_cnm1;
    std::string       rostopic;
    int               tagCount{0};
    int               rotateCode{-1};
    int               flipCode{2};
    bool              fixIntrinsics{false};
    bool              fixExtrinsics{false};
    bool              active{true};
    Eigen::Affine3d pose_inverse_transform;
    // 
  };

std::map<std::string, CalibrationData> read_calibration_data(std::string local_filepath){
  std::map<std::string, CalibrationData> lookup;
  json kalibr_config;
  std::string package_path = ros::package::getPath("recognition");
  std::string hard_coded_path = package_path + local_filepath; //"/cfg/kalibr.json";
  std::ifstream kalibr_json_read(hard_coded_path); 
  kalibr_json_read >> kalibr_config;
  int cam_index = 0;
  bool process = true;
  while (process){
  // for (size_t cam_index = 0; cam_index < n_cams; cam_index++){
    CalibrationData calibData;
    std::string cam = "cam" + std::to_string(cam_index);
    std::cout << "starting read of " << cam << std::endl;
    if (cam_index == 0) {
      
      std::vector<double> distortion_coeffs = kalibr_config[cam]["distortion_coeffs"];
      std::cout << "read: distortion_coeffs" << std::endl;
      std::vector<double> intrinsics = kalibr_config[cam]["intrinsics"];
      std::cout << "read: intrinsics" << std::endl;
      std::string rostopic = kalibr_config[cam]["rostopic"];
      std::cout << "read: " << rostopic << std::endl;
      std::string frame_id_tmp = rostopic;
      int pos = frame_id_tmp.find("/color/image_raw");
      if (pos != std::string::npos)
        frame_id_tmp.replace(pos, std::string("/color/image_raw").size(), "");
      pos = frame_id_tmp.find("/");
      if (pos != std::string::npos)
        frame_id_tmp.replace(pos, std::string("/").size(), "");    
      calibData.frame_id = frame_id_tmp;
      calibData.distortion_coeffs  = distortion_coeffs;
      calibData.intrinsics = intrinsics;
      calibData.T_cn_cnm1 = Eigen::Matrix<double,4,4>::Identity();
      //if (calibData.T_cn_cnm1 != identity()) {
      //  ROS_WARN_STREAM("Cam0 had a non-identity T_cn_cnm1 specified!");
      //  calibData.T_cn_cnm1 = calibData.T_cn_cnm1.identity();
      //}
    } else {

      try
      {
        std::vector<std::vector<double>> values = kalibr_config[cam]["T_cn_cnm1"];
        std::cout << "read: values " << std::endl;
        calibData.T_cn_cnm1 << values[0][0], values[0][1], values[0][2], values[0][3],
              values[1][0], values[1][1], values[1][2], values[1][3],
              values[2][0], values[2][1], values[2][2], values[2][3],
              values[3][0], values[3][1], values[3][2], values[3][3];      
        std::vector<double> distortion_coeffs = kalibr_config[cam]["distortion_coeffs"];
        std::vector<double> intrinsics = kalibr_config[cam]["intrinsics"];
        std::string rostopic = kalibr_config[cam]["rostopic"];
        std::string frame_id_tmp = rostopic;
        int pos = frame_id_tmp.find("/color/image_raw");
        if (pos != std::string::npos)
          frame_id_tmp.replace(pos, std::string("/color/image_raw").size(), "");
        pos = frame_id_tmp.find("/");
        if (pos != std::string::npos)
          frame_id_tmp.replace(pos, std::string("/").size(), "");    
        calibData.frame_id = frame_id_tmp;
        calibData.distortion_coeffs  = distortion_coeffs;
        calibData.intrinsics = intrinsics;

        Eigen::Matrix<double,3,3> R = calibData.T_cn_cnm1.block<3,3>(0,0);
        Eigen::Matrix<double,3,3> I = Eigen::Matrix<double,3,3>::Identity();
        if (R == I) {
          //ROS_ERROR_STREAM(cam << " cannot have an identity rotation in T_cn_cnm1. Perturb values as needed.");
          Eigen::AngleAxisd a(0.00001,Eigen::Vector3d::UnitX());
          calibData.T_cn_cnm1.block<3,3>(0,0) = a.toRotationMatrix();
        }
      }
      catch(const std::exception& e)
      {
        std::cerr << e.what() << '\n';
        process = false;
      }
    }
    lookup[calibData.frame_id] = calibData;
    cam_index++;
  }
  return lookup;
}


/** \brief BaseNode estimates the ground plane equation from a 3D point cloud */
class VisNode {
  private:
    // Publishers
    ros::Publisher detections_pub;
    ros::Publisher cloud_pub;
    image_transport::Publisher image_pub;
    ros::NodeHandle node_;
    ros::Subscriber point_cloud_approximate_sync_;
    image_transport::ImageTransport it;
    ros::ServiceServer camera_info_matrix_server;
    ros::Subscriber camera_info_matrix;
    ros::Subscriber ground_coeffs_sub;
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
    bool zone_json_found = false;

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
    bool use_saved_ground_file = false;

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
    //pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    bool listen_for_ground = false;
    PointCloudPtr clouds_stacked = PointCloudPtr (new PointCloud);
    std::map<std::string, Eigen::Affine3d> frame_transforms;
    std::map<std::string, pcl::PointCloud<pcl::PointXYZRGB>> last_received_cloud;
    bool calibration_refinement = true;
    std::map<std::string, Eigen::Matrix4d> registration_matrices;

    std::map<std::string, CalibrationData> kalibr_lookup;

    VisNode(ros::NodeHandle& nh, bool use_kalibr):
      node_(nh), it(node_)
      {
      
        //n cameras??
        cloud_pub = node_.advertise<sensor_msgs::PointCloud2>("/world_cloud", 1);
        
        if (use_kalibr){
          point_cloud_approximate_sync_ = node_.subscribe("/cleaned_clouds", 1, &VisNode::kalibr_callback, this);
          kalibr_lookup = read_calibration_data("/cfg/kalibr.json");
        } else {
          point_cloud_approximate_sync_ = node_.subscribe("/cleaned_clouds", 1, &VisNode::callback, this);
        }

      }


    void kalibr_callback(const PointCloudT::ConstPtr& cloud_) {
      // Read message header information:
      pcl::GlasbeyLUT colors;
      PointCloudPtr clouds_stacked(new PointCloud);
      std_msgs::Header cloud_header = pcl_conversions::fromPCL(cloud_->header);
      std::string frame_id = cloud_header.frame_id;
      ros::Time frame_time = cloud_header.stamp;

      std::string frame_id_tmp = frame_id;
      int pos = frame_id_tmp.find("_color_optical_frame");
      if (pos != std::string::npos)
        frame_id_tmp.replace(pos, std::string("_color_optical_frame").size(), "");
      pos = frame_id_tmp.find("_depth_optical_frame");
      if (pos != std::string::npos)
      frame_id_tmp.replace(pos, std::string("_depth_optical_frame").size(), "");
      frame_id = frame_id_tmp;
      CalibrationData calibData = kalibr_lookup[frame_id];

      Eigen::Affine3d pose_inverse_transform;

      // debug info
      //Eigen::Matrix3d m = pose_inverse_transform.rotation();
      //Eigen::Vector3d v = pose_inverse_transform.translation();
      //std::cout << "Rotation: " << std::endl << m << std::endl;
      //std::cout << "Translation: " << std::endl << v << std::endl;
      //std::cout << "Matrix: " << pose_inverse_transform.matrix() << std::endl;

      std::cout << "cloud_ size: " << cloud_->size() << std::endl;
      pcl::PointCloud < pcl::PointXYZRGB > cloud_xyzrgb;
      pcl::copyPointCloud(*cloud_, cloud_xyzrgb);

      pose_inverse_transform.matrix() << calibData.T_cn_cnm1;
      pcl::transformPointCloud(cloud_xyzrgb, cloud_xyzrgb, pose_inverse_transform);

      // Detection correction by means of calibration refinement:
      //if (calibration_refinement)
      //{
      //  if (strcmp(frame_id.substr(0,1).c_str(), "/") == 0)
      //  {
      //    frame_id = frame_id.substr(1, frame_id.size() - 1);
      //  }
      //
      //  Eigen::Matrix4d registration_matrix;
      //  std::map<std::string, Eigen::Matrix4d>::iterator registration_matrices_iterator = registration_matrices.find(frame_id);
      //  if (registration_matrices_iterator != registration_matrices.end())
      //  { // camera already present
      //    registration_matrix = registration_matrices_iterator->second;
      //  }
      //  else
      //  { // camera not present
      //    std::cout << "Reading refinement matrix of " << frame_id << " from file." << std::endl;
      //    std::string refinement_filename = ros::package::getPath("opt_calibration") + "/conf/registration_" + frame_id + ".txt";
      //    std::ifstream f(refinement_filename.c_str());
      //    if (f.good()) // if the file exists
      //    {
      //      f.close();
      //     registration_matrix = readMatrixFromFile (refinement_filename);
      //      registration_matrices.insert(std::pair<std::string, Eigen::Matrix4d> (frame_id, registration_matrix));
      //    }
      //    else  // if the file does not exist
      //    {
      //      // insert the identity matrix
      //      std::cout << "Refinement file not found! Not doing refinement for this sensor." << std::endl;
      //      registration_matrices.insert(std::pair<std::string, Eigen::Matrix4d> (frame_id, Eigen::Matrix4d::Identity()));
      //    }
      //  }
      //  //cloud_xyzrgb = registration_matrix * cloud_xyzrgb;
      //  Eigen::Affine3d registration_transform;
      //  registration_transform.matrix() << registration_matrix;
      //  pcl::transformPointCloud(cloud_xyzrgb, cloud_xyzrgb, registration_transform);
      //}

      std::cout << "cloud_xyzrgb size: " << cloud_xyzrgb.size() << std::endl;
      //PointCloudT::ConstPtr::iterator it2 = cloud_->points.begin();
      //pcl::PointCloud<pcl::PointXYZRGB>::iterator it;
      //pcl::PointCloud<pcl::PointXYZ>::iterator it2;
      //for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = cloud_xyzrgb.points.begin(); it != cloud_xyzrgb.points.end(); ++it,++it2)
  
      // debug output -- all inf
      //for (pcl::PointCloud<pcl::PointXYZRGB>::iterator cloud_it(cloud_xyzrgb.begin()); cloud_it != cloud_xyzrgb.end(); ++cloud_it)
      //{
      //  std::cout << "cloud_it->x: " << cloud_it->x << std::endl;
      //  std::cout << "cloud_it->y: " << cloud_it->y << std::endl;
      //  std::cout << "cloud_it->z: " << cloud_it->z << std::endl;
      //}

      //for (pcl::PointCloud<pcl::PointXYZRGB>::iterator cloud_it(cloud_xyzrgb.begin()); cloud_it != cloud_xyzrgb.end();
      //    ++cloud_it)
      //  cloud_it->rgb = colors.at(0).rgb;
      last_received_cloud[frame_id] = cloud_xyzrgb;
      std::map<std::string, pcl::PointCloud<pcl::PointXYZRGB>>::iterator it = last_received_cloud.begin();

      while (it != last_received_cloud.end())
      {
          std::string frame_identifier = it->first;
          pcl::PointCloud<pcl::PointXYZRGB> cloud_xyzrgb = it->second;
          *clouds_stacked += cloud_xyzrgb;
          it++;
      }
      
      std::cout << "clouds stacked size: " << clouds_stacked->size() << std::endl;
      // Publish point clouds
      clouds_stacked->header.frame_id = "world";
      cloud_pub.publish(clouds_stacked);


      // Create XYZ cloud for viz
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_for_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
      //https://answers.ros.org/question/9515/how-to-convert-between-different-point-cloud-types-using-pcl/
      //pcl::PointXYZRGB xyzrgb_point;
      //cloud_for_vis->points.resize(cloud_xyzrgb.width * cloud_xyzrgb.height, xyzrgb_point);
      //cloud_for_vis->width = cloud_xyzrgb.width;
      //cloud_for_vis->height = cloud_xyzrgb.height;
      //cloud_for_vis->is_dense = false;

      // fill xyzrgb
      //for (int i=0;i<cloud_xyzrgb.height;i++)
     // {
      //    for (int j=0;j<cloud_xyzrgb.width;j++)
      //    {
      //    cloud_for_vis->at(j,i).x = cloud_xyzrgb.at(j,i).x;
      //    cloud_for_vis->at(j,i).y = cloud_xyzrgb.at(j,i).y;
      //    cloud_for_vis->at(j,i).z = cloud_xyzrgb.at(j,i).z;
      //    }
      //}
      pcl::copyPointCloud(*clouds_stacked, *cloud_for_vis);

      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_for_vis);
      viewer.addPointCloud<PointT> (cloud_for_vis, rgb, "temp_cloud");
      //viewer.showCloud (clouds_stacked);
      viewer.addCoordinateSystem (0.5, "axis", 0); 
      viewer.setBackgroundColor (0, 0, 0, 0); 
      //viewer.setPosition (800, 400); 
      //viewer.setCameraPosition(0,0,-2,0,-1,0,0);;
      viewer.spinOnce ();
      viewer.removeAllShapes();
      viewer.removeAllPointClouds();  
    }


    void callback(const PointCloudT::ConstPtr& cloud_) {
      // Read message header information:
      pcl::GlasbeyLUT colors;
      PointCloudPtr clouds_stacked(new PointCloud);
      std_msgs::Header cloud_header = pcl_conversions::fromPCL(cloud_->header);
      std::string frame_id = cloud_header.frame_id;
      ros::Time frame_time = cloud_header.stamp;

      std::string frame_id_tmp = frame_id;
      int pos = frame_id_tmp.find("_color_optical_frame");
      if (pos != std::string::npos)
        frame_id_tmp.replace(pos, std::string("_color_optical_frame").size(), "");
      pos = frame_id_tmp.find("_depth_optical_frame");
      if (pos != std::string::npos)
      frame_id_tmp.replace(pos, std::string("_depth_optical_frame").size(), "");
      
      frame_id = frame_id_tmp;

      // add to pointcloud vis
      //pcl::transformPointCloud(cloud_, PtrPointCloud, inverse_transform.matrix());
      tf::StampedTransform transform;
      tf::StampedTransform inverse_transform;
      Eigen::Affine3d pose_transform;
      Eigen::Affine3d pose_inverse_transform;
      //try {
      //  pose_inverse_transform = frame_transforms[frame_id];
      //} catch(const std::exception& e) {
      //  //Calculate direct and inverse transforms between camera and world frame:
      //  tf_listener.lookupTransform("/world", frame_id, ros::Time(0), transform);
      //  tf_listener.lookupTransform(frame_id, "/world", ros::Time(0), inverse_transform);
      //
      //  tf::transformTFToEigen(transform, pose_transform);
      //  tf::transformTFToEigen(inverse_transform, pose_inverse_transform);
      //  frame_transforms[frame_id] = pose_inverse_transform;
      //}
      
      tf_listener.lookupTransform("/world", frame_id, ros::Time(0), transform);
      tf_listener.lookupTransform(frame_id, "/world", ros::Time(0), inverse_transform);

      tf::transformTFToEigen(transform, pose_transform);
      tf::transformTFToEigen(inverse_transform, pose_inverse_transform);
      //frame_transforms[frame_id] = pose_inverse_transform;
      
      // debug info
      //Eigen::Matrix3d m = pose_inverse_transform.rotation();
      //Eigen::Vector3d v = pose_inverse_transform.translation();
      //std::cout << "Rotation: " << std::endl << m << std::endl;
      //std::cout << "Translation: " << std::endl << v << std::endl;
      //std::cout << "Matrix: " << pose_inverse_transform.matrix() << std::endl;

      std::cout << "cloud_ size: " << cloud_->size() << std::endl;
      pcl::PointCloud < pcl::PointXYZRGB > cloud_xyzrgb;
      pcl::copyPointCloud(*cloud_, cloud_xyzrgb);
      pcl::transformPointCloud(cloud_xyzrgb, cloud_xyzrgb, pose_inverse_transform);


      // Detection correction by means of calibration refinement:
      if (calibration_refinement)
      {
        if (strcmp(frame_id.substr(0,1).c_str(), "/") == 0)
        {
          frame_id = frame_id.substr(1, frame_id.size() - 1);
        }

        Eigen::Matrix4d registration_matrix;
        std::map<std::string, Eigen::Matrix4d>::iterator registration_matrices_iterator = registration_matrices.find(frame_id);
        if (registration_matrices_iterator != registration_matrices.end())
        { // camera already present
          registration_matrix = registration_matrices_iterator->second;
        }
        else
        { // camera not present
          std::cout << "Reading refinement matrix of " << frame_id << " from file." << std::endl;
          std::string refinement_filename = ros::package::getPath("opt_calibration") + "/conf/registration_" + frame_id + ".txt";
          std::ifstream f(refinement_filename.c_str());
          if (f.good()) // if the file exists
          {
            f.close();
            registration_matrix = readMatrixFromFile (refinement_filename);
            registration_matrices.insert(std::pair<std::string, Eigen::Matrix4d> (frame_id, registration_matrix));
          }
          else  // if the file does not exist
          {
            // insert the identity matrix
            std::cout << "Refinement file not found! Not doing refinement for this sensor." << std::endl;
            registration_matrices.insert(std::pair<std::string, Eigen::Matrix4d> (frame_id, Eigen::Matrix4d::Identity()));
          }
        }
        //cloud_xyzrgb = registration_matrix * cloud_xyzrgb;
        Eigen::Affine3d registration_transform;
        registration_transform.matrix() << registration_matrix;
        pcl::transformPointCloud(cloud_xyzrgb, cloud_xyzrgb, registration_transform);
      }

      std::cout << "cloud_xyzrgb size: " << cloud_xyzrgb.size() << std::endl;
      //PointCloudT::ConstPtr::iterator it2 = cloud_->points.begin();
      //pcl::PointCloud<pcl::PointXYZRGB>::iterator it;
      //pcl::PointCloud<pcl::PointXYZ>::iterator it2;
      //for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = cloud_xyzrgb.points.begin(); it != cloud_xyzrgb.points.end(); ++it,++it2)
  
      // debug output -- all inf
      //for (pcl::PointCloud<pcl::PointXYZRGB>::iterator cloud_it(cloud_xyzrgb.begin()); cloud_it != cloud_xyzrgb.end(); ++cloud_it)
      //{
      //  std::cout << "cloud_it->x: " << cloud_it->x << std::endl;
      //  std::cout << "cloud_it->y: " << cloud_it->y << std::endl;
      //  std::cout << "cloud_it->z: " << cloud_it->z << std::endl;
      //}

      //for (pcl::PointCloud<pcl::PointXYZRGB>::iterator cloud_it(cloud_xyzrgb.begin()); cloud_it != cloud_xyzrgb.end();
      //    ++cloud_it)
      //  cloud_it->rgb = colors.at(0).rgb;
      last_received_cloud[frame_id] = cloud_xyzrgb;
      std::map<std::string, pcl::PointCloud<pcl::PointXYZRGB>>::iterator it = last_received_cloud.begin();

      while (it != last_received_cloud.end())
      {
          std::string frame_identifier = it->first;
          pcl::PointCloud<pcl::PointXYZRGB> cloud_xyzrgb = it->second;
          *clouds_stacked += cloud_xyzrgb;
          it++;
      }
      
      std::cout << "clouds stacked size: " << clouds_stacked->size() << std::endl;
      // Publish point clouds
      clouds_stacked->header.frame_id = "world";
      cloud_pub.publish(clouds_stacked);


      // Create XYZ cloud for viz
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_for_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
      //https://answers.ros.org/question/9515/how-to-convert-between-different-point-cloud-types-using-pcl/
      //pcl::PointXYZRGB xyzrgb_point;
      //cloud_for_vis->points.resize(cloud_xyzrgb.width * cloud_xyzrgb.height, xyzrgb_point);
      //cloud_for_vis->width = cloud_xyzrgb.width;
      //cloud_for_vis->height = cloud_xyzrgb.height;
      //cloud_for_vis->is_dense = false;

      // fill xyzrgb
      //for (int i=0;i<cloud_xyzrgb.height;i++)
     // {
      //    for (int j=0;j<cloud_xyzrgb.width;j++)
      //    {
      //    cloud_for_vis->at(j,i).x = cloud_xyzrgb.at(j,i).x;
      //    cloud_for_vis->at(j,i).y = cloud_xyzrgb.at(j,i).y;
      //    cloud_for_vis->at(j,i).z = cloud_xyzrgb.at(j,i).z;
      //    }
      //}
      pcl::copyPointCloud(*clouds_stacked, *cloud_for_vis);

      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_for_vis);
      viewer.addPointCloud<PointT> (cloud_for_vis, rgb, "temp_cloud");
      //viewer.showCloud (clouds_stacked);
      viewer.addCoordinateSystem (0.5, "axis", 0); 
      viewer.setBackgroundColor (0, 0, 0, 0); 
      //viewer.setPosition (800, 400); 
      //viewer.setCameraPosition(0,0,-2,0,-1,0,0);;
      viewer.spinOnce ();
      viewer.removeAllShapes();
      viewer.removeAllPointClouds();  
    }

    void new_callback(const PointCloudT::ConstPtr& cloud_) {
      // Read message header information:
      pcl::GlasbeyLUT colors;
      PointCloudPtr clouds_stacked(new PointCloud);
      std_msgs::Header cloud_header = pcl_conversions::fromPCL(cloud_->header);
      std::string frame_id = cloud_header.frame_id;
      ros::Time frame_time = cloud_header.stamp;

      std::string frame_id_tmp = frame_id;
      int pos = frame_id_tmp.find("_color_optical_frame");
      if (pos != std::string::npos)
        frame_id_tmp.replace(pos, std::string("_color_optical_frame").size(), "");
      pos = frame_id_tmp.find("_depth_optical_frame");
      if (pos != std::string::npos)
      frame_id_tmp.replace(pos, std::string("_depth_optical_frame").size(), "");
      
      frame_id = frame_id_tmp;
      
      // add to pointcloud vis
      //pcl::transformPointCloud(cloud_, PtrPointCloud, inverse_transform.matrix());
      tf::StampedTransform transform;
      tf::StampedTransform inverse_transform;
      Eigen::Affine3d pose_transform;
      Eigen::Affine3d pose_inverse_transform;
      //try {
      //  pose_inverse_transform = frame_transforms[frame_id];
      //} catch(const std::exception& e) {
      //  //Calculate direct and inverse transforms between camera and world frame:
      //  tf_listener.lookupTransform("/world", frame_id, ros::Time(0), transform);
      //  tf_listener.lookupTransform(frame_id, "/world", ros::Time(0), inverse_transform);
      //
      //  tf::transformTFToEigen(transform, pose_transform);
      //  tf::transformTFToEigen(inverse_transform, pose_inverse_transform);
      //  frame_transforms[frame_id] = pose_inverse_transform;
      //}
      
      tf_listener.lookupTransform("/world", frame_id, ros::Time(0), transform);
      tf_listener.lookupTransform(frame_id, "/world", ros::Time(0), inverse_transform);

      tf::transformTFToEigen(transform, pose_transform);
      tf::transformTFToEigen(inverse_transform, pose_inverse_transform);
      //frame_transforms[frame_id] = pose_inverse_transform;
      
      // debug info
      //Eigen::Matrix3d m = pose_inverse_transform.rotation();
      //Eigen::Vector3d v = pose_inverse_transform.translation();
      //std::cout << "Rotation: " << std::endl << m << std::endl;
      //std::cout << "Translation: " << std::endl << v << std::endl;
      //std::cout << "Matrix: " << pose_inverse_transform.matrix() << std::endl;

      std::cout << "cloud_ size: " << cloud_->size() << std::endl;
      pcl::PointCloud < pcl::PointXYZRGB > cloud_xyzrgb;
      pcl::copyPointCloud(*cloud_, cloud_xyzrgb);
      pcl::transformPointCloud(cloud_xyzrgb, cloud_xyzrgb, pose_inverse_transform);


      std::cout << "cloud_xyzrgb size: " << cloud_xyzrgb.size() << std::endl;
      //PointCloudT::ConstPtr::iterator it2 = cloud_->points.begin();
      //pcl::PointCloud<pcl::PointXYZRGB>::iterator it;
      //pcl::PointCloud<pcl::PointXYZ>::iterator it2;
      //for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = cloud_xyzrgb.points.begin(); it != cloud_xyzrgb.points.end(); ++it,++it2)
  
      // debug output -- all inf
      //for (pcl::PointCloud<pcl::PointXYZRGB>::iterator cloud_it(cloud_xyzrgb.begin()); cloud_it != cloud_xyzrgb.end(); ++cloud_it)
      //{
      //  std::cout << "cloud_it->x: " << cloud_it->x << std::endl;
      //  std::cout << "cloud_it->y: " << cloud_it->y << std::endl;
      //  std::cout << "cloud_it->z: " << cloud_it->z << std::endl;
      //}

      //for (pcl::PointCloud<pcl::PointXYZRGB>::iterator cloud_it(cloud_xyzrgb.begin()); cloud_it != cloud_xyzrgb.end();
      //    ++cloud_it)
      //  cloud_it->rgb = colors.at(0).rgb;
      last_received_cloud[frame_id] = cloud_xyzrgb;
      std::map<std::string, pcl::PointCloud<pcl::PointXYZRGB>>::iterator it = last_received_cloud.begin();

      while (it != last_received_cloud.end())
      {
          std::string frame_identifier = it->first;
          pcl::PointCloud<pcl::PointXYZRGB> cloud_xyzrgb = it->second;
          *clouds_stacked += cloud_xyzrgb;
          it++;
      }
      
      std::cout << "clouds stacked size: " << clouds_stacked->size() << std::endl;
      // Publish point clouds
      clouds_stacked->header.frame_id = "world";
      cloud_pub.publish(clouds_stacked);


      // Create XYZ cloud for viz
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_for_vis(new pcl::PointCloud<pcl::PointXYZRGB>);
      //https://answers.ros.org/question/9515/how-to-convert-between-different-point-cloud-types-using-pcl/
      //pcl::PointXYZRGB xyzrgb_point;
      //cloud_for_vis->points.resize(cloud_xyzrgb.width * cloud_xyzrgb.height, xyzrgb_point);
      //cloud_for_vis->width = cloud_xyzrgb.width;
      //cloud_for_vis->height = cloud_xyzrgb.height;
      //cloud_for_vis->is_dense = false;

      // fill xyzrgb
      //for (int i=0;i<cloud_xyzrgb.height;i++)
     // {
      //    for (int j=0;j<cloud_xyzrgb.width;j++)
      //    {
      //    cloud_for_vis->at(j,i).x = cloud_xyzrgb.at(j,i).x;
      //    cloud_for_vis->at(j,i).y = cloud_xyzrgb.at(j,i).y;
      //    cloud_for_vis->at(j,i).z = cloud_xyzrgb.at(j,i).z;
      //    }
      //}
      pcl::copyPointCloud(*clouds_stacked, *cloud_for_vis);

      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_for_vis);
      viewer.addPointCloud<PointT> (cloud_for_vis, rgb, "temp_cloud");
      //viewer.showCloud (clouds_stacked);
      viewer.addCoordinateSystem (0.5, "axis", 0); 
      viewer.setBackgroundColor (0, 0, 0, 0); 
      //viewer.setPosition (800, 400); 
      //viewer.setCameraPosition(0,0,-2,0,-1,0,0);;
      viewer.spinOnce ();
      viewer.removeAllShapes();
      viewer.removeAllPointClouds();  
    }
};

std::string find_frame_id(std::string topic){
  std::regex rgx("^/*[0-9a-zA-Z_]+[/]");
  std::string frame_id_tmp;
  std::match_results<std::string::iterator> match;
  if (std::regex_search(topic.begin(), topic.end(), match, rgx))
    std::cout << "match: " << match[0] << '\n';
  frame_id_tmp = match[0];
  frame_id_tmp.erase(std::remove(frame_id_tmp.begin(), frame_id_tmp.end(), '/'), frame_id_tmp.end());
  return frame_id_tmp;
}

class InputCloud
{
  private:
    //pose ps;
    std::string topic_name;
    std::string frame_id;
    ros::Subscriber sub;
    Eigen::Matrix4f transform;
    pcl::PointCloud<pcl::PointXYZ> inCloud;
  public:
    pcl::PointCloud<pcl::PointXYZ> tfdinCloud;
    CalibrationData calibData;
  public:
    InputCloud(std::string topic,ros::NodeHandle nh)
    {

      std::string topic_frame_id = find_frame_id(topic);
      //if (temp.length()== 0){
      // throw blah blah blah
      //}

      json kalibr_config;
      std::string package_path = ros::package::getPath("recognition");
      std::string hard_coded_path = package_path + "/cfg/kalibr.json";
      std::ifstream kalibr_json_read(hard_coded_path); 
      kalibr_json_read >> kalibr_config;

      int cam_index = 0;
      bool process = true;
      while (process){
        std::string cam = "cam" + std::to_string(cam_index);
        std::string rostopic = kalibr_config[cam]["rostopic"];
        std::string rostopic_frame_id = find_frame_id(rostopic);
        
        try
        {
          if (rostopic_frame_id == topic_frame_id){
            std::vector<double> distortion_coeffs = kalibr_config[cam]["distortion_coeffs"];
            std::vector<double> intrinsics = kalibr_config[cam]["intrinsics"];
            calibData.frame_id = rostopic_frame_id;
            calibData.distortion_coeffs  = distortion_coeffs;
            calibData.intrinsics = intrinsics;
            //calibData.T_cn_cnm1 = Eigen::Matrix<double,4,4>::Identity();
            if (cam_index == 0) {
              calibData.T_cn_cnm1 = Eigen::Matrix<double,4,4>::Identity();
            } else {
              std::vector<std::vector<double>> values = kalibr_config[cam]["T_cn_cnm1"];
              std::cout << "read: values " << std::endl;
              calibData.T_cn_cnm1 << values[0][0], values[0][1], values[0][2], values[0][3],
                    values[1][0], values[1][1], values[1][2], values[1][3],
                    values[2][0], values[2][1], values[2][2], values[2][3],
                    values[3][0], values[3][1], values[3][2], values[3][3];
              Eigen::Matrix<double,3,3> R = calibData.T_cn_cnm1.block<3,3>(0,0);
              Eigen::Matrix<double,3,3> I = Eigen::Matrix<double,3,3>::Identity();
              if (R == I) {
                //ROS_ERROR_STREAM(cam << " cannot have an identity rotation in T_cn_cnm1. Perturb values as needed.");
                Eigen::AngleAxisd a(0.00001,Eigen::Vector3d::UnitX());
                calibData.T_cn_cnm1.block<3,3>(0,0) = a.toRotationMatrix();
              }
            }
            process = false;
          } 
        }
        catch(const std::exception& e)
        {
          std::cerr << e.what() << '\n';
          process = false;
        }
        cam_index++;
      }

      calibData.pose_inverse_transform.matrix() << calibData.T_cn_cnm1;
      
      //initialize InputCloud
      // this->ps = p;
      this->topic_name = topic;
      //this->transform = poseTotfmatrix();
      sub = nh.subscribe(topic_name,1,&InputCloud::cloudCallback,this);
    }
    ~InputCloud() {}
    void cloudCallback(const PointCloudT::ConstPtr& input)
    {
      // not sure if you need this????
      pcl::fromROSMsg(*input,this->inCloud);
      pcl::transformPointCloud(this->inCloud, this->tfdinCloud, calibData.pose_inverse_transform);
    }
};//class InputCloud


class OutputCloud
{
  private:
    std::string topic_name;
    std::string frame_id;
  public:
    pcl::PointCloud<pcl::PointXYZ> outCloud;
    sensor_msgs::PointCloud2 outCloudMsg;
    ros::Publisher pub;
  public:
    OutputCloud(std::string topic, std::string frame, ros::NodeHandle nh)
    {
      this->topic_name = topic;
      this->frame_id = frame;
      pub = nh.advertise<sensor_msgs::PointCloud2>(this->topic_name,1);
      this->outCloud.header.frame_id = this->frame_id;
    }
    ~OutputCloud() {}
};//class OutputCloud


class CloudMerger
{
  private:
    int nsensors;
    InputCloud* inClAry[MAX_NSENSORS];
    OutputCloud* outCl;
  public:
    pcl::visualization::PCLVisualizer viewer = pcl::visualization::PCLVisualizer ("3D Viewer");
    CloudMerger(ros::NodeHandle node, ros::NodeHandle private_nh)
    {
      //use private node handle to get parameters
      std::string s_key("CloudIn0"); //searching key (input)
      std::string so_key("CloudOut"); //searching key (output)
      std::string key;
      this->nsensors = 0;
      
      //get all parameters
      for(int i = 1 ; i <= MAX_NSENSORS ; i++){
        //Searching key must be Cloud[1] ~ Cloud[MAX_NSENSORS]
        s_key[s_key.find((i + '0') - 1)] = i + '0';
        if(private_nh.searchParam(s_key, key)){
          std::string topic_name(s_key); //set to default

          if(!private_nh.getParam(key+"/topic_name",topic_name)){
            std::cout << "not found : "<< key +"/topic_name" << std::endl;
          }

          this->nsensors++;
          inClAry[i-1] = new InputCloud(topic_name, node);
        }
      }
      if(this->nsensors == 0) std::cout << "No input data found" << std::endl;
      //outputCloud object
      if(private_nh.searchParam(so_key,key)){
        std::string topic_name(so_key); //set to default
        std::string frame_id("default_frame_id"); //set to default
        if(!private_nh.getParam(key+"/topic_name",topic_name)){
          std::cout << "not found : "<< key +"/topic_name" << std::endl;
        }
        if(!private_nh.getParam(key+"/frame_id", frame_id)){
          std::cout << "not found : "<< key +"/frame_id" << std::endl;
        }		
        outCl = new OutputCloud(topic_name, frame_id, node);
      }
    }
    void mergeNpub()
    {
      /* synchronization problem
      * Time stamp of 4 sensors are apparently different.
      * velodyne drivers publish messages every 100ms(10hz)
      * Is it ok?
      */
      outCl->outCloud.clear(); //clear before use. Header info is not cleared.
      for(int i = 0 ; i < this->nsensors ; i++){
        outCl->outCloud += inClAry[i]->tfdinCloud;
      }
      //initialize header info with first Cloud info
      outCl->outCloud.header.seq = inClAry[0]->tfdinCloud.header.seq;
      outCl->outCloud.header.stamp = inClAry[0]->tfdinCloud.header.stamp;
      pcl::toROSMsg(outCl->outCloud,outCl->outCloudMsg);
      outCl->pub.publish(outCl->outCloudMsg);


      // Create XYZ cloud for viz
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_for_vis(new pcl::PointCloud<pcl::PointXYZRGB>);

      pcl::copyPointCloud(*outCl->outCloud, *cloud_for_vis);

      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_for_vis);
      viewer.addPointCloud<PointT> (cloud_for_vis, rgb, "temp_cloud");
      //viewer.showCloud (clouds_stacked);
      viewer.addCoordinateSystem (0.5, "axis", 0); 
      viewer.setBackgroundColor (0, 0, 0, 0); 
      //viewer.setPosition (800, 400); 
      //viewer.setCameraPosition(0,0,-2,0,-1,0,0);;
      viewer.spinOnce ();
      viewer.removeAllShapes();
      viewer.removeAllPointClouds();  
    }
    ~CloudMerger() {}
};//class CloudMerger


int main(int argc, char** argv) {
  bool use_kalibr;
  bool use_cloudmerger;

  ros::init(argc, argv, "tvm_detection_node");
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;
  pnh.param("use_kalibr", use_kalibr, false);
  pnh.param("use_cloudmerger", use_cloudmerger, false);
  std::cout << "use_kalibr: " << use_kalibr << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  
  if (use_cloudmerger){
    ros::Rate loop_rate(10);
    CloudMerger::CloudMerger *cm = new CloudMerger::CloudMerger(nh, pnh);
    // Spin
    while(ros::ok())
    {
      cm->mergeNpub();
      ros::spinOnce();
      loop_rate.sleep();
    }
  }
  else {
    VisNode node(nh, use_kalibr);
    std::cout << "VisNode init " << std::endl;
    ros::spin();
  }
  return 0;
}
 