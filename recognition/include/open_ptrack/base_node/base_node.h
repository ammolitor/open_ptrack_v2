#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/dnn/dnn.hpp>
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

//#include <open_ptrack/nms/nms.h>

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
//#include <open_ptrack/yolo_tvm.hpp>

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


#ifndef OPEN_PTRACK_BASE_NODE_BASE_NODE_H_
#define OPEN_PTRACK_BASE_NODE_BASE_NODE_H_

namespace open_ptrack
{
  namespace base_node
  {

    /** \brief BaseNode estimates the ground plane equation from a 3D point cloud */
    class BaseNode {
      public:
        ros::NodeHandle node_;
        ros::Subscriber point_cloud_approximate_sync_;
        image_transport::ImageTransport it;
        image_transport::Publisher image_pub;
        ros::ServiceServer camera_info_matrix_server;
        ros::Subscriber camera_info_matrix;
        tf::TransformListener tf_listener;
        tf::Transform worldToCamTransform;

      public:

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
        //###################################
        //## Detection Variables ##
        //###################################
        float thresh = 0.3f;

        //###################################
        //## Transform Listeners ##
        //###################################
        Eigen::Affine3d world2rgb;
        tf::StampedTransform world2rgb_transform;
        tf::StampedTransform world_transform;
        tf::StampedTransform world_inverse_transform;
    
        /** \brief BaseNode Constructor. */
        BaseNode(ros::NodeHandle& nh, std::string sensor_string, json zone);//:
        //    node_(nh), it(node_)
        //  { }

        /** \brief Destructor. */
        virtual ~BaseNode ();

        /**
         * \brief gather info on camera intrinsics.
         *
         * \param[in] msg pointer to the camera info.
         */
        void camera_info_callback(const CameraInfo::ConstPtr & msg);

        /**
         * \brief sets the background of the given camera for background removal.
         *
         * \param[in] pointer to the background cloud.
         */
        void set_background (PointCloudPtr& background_cloud);

        /**
         * \brief extracts the rbg image from the pointcloud.
         *
         * \param[in] pointer to the input_cloud cloud.
         * \param[in] pointer to the output_cloud cloud.
         */
        void extract_RGB_from_pointcloud (PointCloudPtr input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud);

        /**
         * \brief function to compute the background cloud.
         *
         * \param[in] pointer to the point cloud.
         */
        PointCloudT::Ptr compute_background_cloud (PointCloudPtr& cloud);

        /**
         * \brief function to preprocess a cloud.
         *
         * \param[in] pointer to the input_cloud point cloud.
         */
        PointCloudPtr preprocess_cloud (PointCloudPtr& input_cloud);

        /**
         * \brief function to rotate a cloud.
         *
         * \param[in] pointer to the input_cloud point cloud.
         */
        PointCloudPtr rotate_cloud(PointCloudPtr cloud, Eigen::Affine3f transform );

        /**
         * \brief function to rotate the ground.
         *
         * \param[in] vector of ground coefficients.
         * \param[in] the affine transform for rotation.
         */
        Eigen::VectorXf rotate_ground( Eigen::VectorXf ground_coeffs, Eigen::Affine3f transform);

        /**
         * \brief function to rotate the ground.
         *
         * \param[in] vector of ground coefficients.
         * \param[in] the affine transform for rotation.
         */
        Eigen::VectorXf compute_subclustering( Eigen::VectorXf ground_coeffs, Eigen::Affine3f transform);

        /**
         * \brief function to compute blob clusters within the ground-removed point cloud
         *
         * \param[in] no ground pointcloud
         * \param[in] PersonClusers to be filled by this function
         */
        void compute_subclustering(PointCloudPtr no_ground_cloud, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters);

        /**
         * \brief function that creates foreground a cloud and puts blobs found into PersonClusters 
         *
         * \param[in] no ground pointcloud
         * \param[in] PersonClusers to be filled by this function
         */
        void create_foreground_cloud(const PointCloudT::ConstPtr& cloud_, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters);

        /**
         * \brief function that runs all subfunctions necessary to compute the ground plane 
         *
         * \param[in] input cloud from camera
         */
        void set_ground_variables(const PointCloudT::ConstPtr& cloud_);

        /**
         * \brief function to further filter the blob clusters based on how similar to a human's height they may be
         *
         * \param[in] the build person clusters
         * \param[in] empty 2d cluster vector
         * \param[in] empty 3d cluster vector
         */
        void compute_head_subclustering(std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters, std::vector<cv::Point2f> cluster_centroids2d, std::vector<cv::Point3f> cluster_centroids3d);
        /**
         * \brief callback function to perform detection, pose-rec, etc.
         *
         * \param[in]input cloud from camera
         */
        // void callback(const PointCloudT::ConstPtr& cloud_)
        bool check_detection_msg(opt_msgs::Detection detection_msg);


        void draw_skelaton(cv::Mat cv_image_clone, std::vector<cv::Point3f> points);


    };
  } /* namespace base_node */
} /* namespace open_ptrack */
#endif /* OPEN_PTRACK_BASE_NODE_BASE_NODE_H_ */

