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

    VisNode(ros::NodeHandle& nh):
      node_(nh), it(node_)
      {
      
        //n cameras??
        cloud_pub = node_.advertise<sensor_msgs::PointCloud2>("/world_cloud", 1);
        point_cloud_approximate_sync_ = node_.subscribe("/cleaned_clouds", 1, &VisNode::callback, this);

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
      try {
        pose_inverse_transform = frame_transforms[frame_id];
      } catch(const std::exception& e) {
        //Calculate direct and inverse transforms between camera and world frame:
        tf_listener.lookupTransform("/world", frame_id, ros::Time(0), transform);
        tf_listener.lookupTransform(frame_id, "/world", ros::Time(0), inverse_transform);

        tf::transformTFToEigen(transform, pose_transform);
        tf::transformTFToEigen(inverse_transform, pose_inverse_transform);
        frame_transforms[frame_id] = pose_inverse_transform;
      }
      //std::cout << frame_transforms[frame_id] << std::endl;
      
      Eigen::Matrix3d m = pose_inverse_transform.rotation();
      Eigen::Vector3d v = pose_inverse_transform.translation();
      std::cout << "Rotation: " << std::endl << m << std::endl;
      std::cout << "Translation: " << std::endl << v << std::endl;
      std::cout << "Matrix: " << pose_inverse_transform.matrix() << std::endl;

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
      pcl::copyPointCloud(cloud_xyzrgb, *cloud_for_vis);

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

int main(int argc, char** argv) {
  std::string sensor_name;
  double max_distance;
  // json zone_json;
  bool use_dynamic_reconfigure;
  // std::string area_package_path = ros::package::getPath("recognition");
  // std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
  // std::ifstream area_json_read(area_hard_coded_path);
  // area_json_read >> zone_json;

  std::cout << "--- tvm_detection_node ---" << std::endl;
  ros::init(argc, argv, "tvm_detection_node");
  // something is off here... with the private namespace
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;
  pnh.param("sensor_name", sensor_name, std::string("d435"));
  pnh.param("use_dynamic_reconfigure", use_dynamic_reconfigure, false);
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  VisNode node(nh);
  std::cout << "VisNode init " << std::endl;
  ros::spin();
  return 0;
}
 