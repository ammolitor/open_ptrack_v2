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

#include <open_ptrack/nms/nms.h>

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


//#ifndef OPEN_PTRACK_GROUND_SEGMENTATION_GROUND_SEGMENTATION_H_
//#define OPEN_PTRACK_GROUND_SEGMENTATION_GROUND_SEGMENTATION_H_

namespace open_ptrack
{
  namespace base_node
  {

    /** \brief BaseNode estimates the ground plane equation from a 3D point cloud */
    class BaseNode
    {

      public:

        typedef pcl::PointCloud<PointT> PointCloud;
        typedef boost::shared_ptr<PointCloud> PointCloudPtr;
        typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

        /** \brief BaseNode Constructor. */
        BaseNode(ros::NodeHandle& nh, std::string sensor_string, json zone);

        /** \brief Destructor. */
        virtual ~BaseNode ();

        /**
         * \brief Set the pointer to the input cloud.
         *
         * \param[in] cloud A pointer to the input cloud.
         */
        void camera_info_callback(const CameraInfo::ConstPtr & msg);

void extractRGBFromPointCloud (PointCloudPtr input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud);


void setBackground (PointCloudPtr& background_cloud)


PointCloudT::Ptr computeBackgroundCloud (PointCloudPtr& cloud){

  PointCloudPtr preprocessCloud (PointCloudPtr& input_cloud)

PointCloudPtr rotateCloud(PointCloudPtr cloud, Eigen::Affine3f transform ){

Eigen::VectorXf rotateGround( Eigen::VectorXf ground_coeffs, Eigen::Affine3f transform){

void compute_subclustering(PointCloudPtr no_ground_cloud, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){

void create_foreground_cloud(const PointCloudT::ConstPtr& cloud_, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){

void set_ground_variables(const PointCloudT::ConstPtr& cloud_){

vector<Point3f> clusterPoints(vector<Point3f>& points)

bool filterBboxByArea(int xmin, int ymin, int xmax, int ymax, double range)

void drawCube(Mat& img, Point3d min_xyz, Point3d max_xyz)

void extract_cube(Mat& img, Point3d min_xyz, Point3d max_xyz)

void extract_cube_boundries(vector<Point3f> points_fg, Point3d min_xyz, Point3d max_xyz) {

void compute_head_subclustering(std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters, std::vector<cv::Point2f> cluster_centroids2d, std::vector<cv::Point3f> cluster_centroids3d){

void mode_1_callback_cloud_only(const PointCloudT::ConstPtr& cloud_)

void json_cfg_callback(uint32_t level)

        /**
         * \brief Return true if the cloud has ratio of NaN over total number of points greater than "max_ratio".
         *
         * \param[in] cloud The input cloud.
         * \param[in] max_ratio The ratio of invalid points over which a cloud is considered as not valid.
         *
         * \return true if the cloud has ratio of NaN over total number of points greater than "max_ratio".
         */
        bool
        tooManyNaN (PointCloudConstPtr cloud, float max_ratio);

        /**
         * \brief Return true if the percentage of points with confidence below the confidence_threshold is greater than max_ratio.
         *
         * \param[in] confidence_image Image with confidence values for every pixel.
         * \param[in] confidence_threshold Threshold on the confidence to consider a point as valid.
         * \param[in] max_ratio The ratio of invalid points over which a cloud is considered as not valid.
         *
         * \return true if the cloud has ratio of NaN over total number of points greater than "max_ratio".
         */
        bool
        tooManyLowConfidencePoints (cv::Mat& confidence_image, int confidence_threshold, double max_ratio);

        /**
         * \brief Compute the ground plane coefficients from the transform between two reference frames.
         *
         * \param[in] camera_frame Camera frame id.
         * \param[in] world_frame Ground frame id.
         *
         * \return Vector of ground plane coefficients.
         */
        Eigen::VectorXf
        computeFromTF (std::string camera_frame, std::string ground_frame);

        /**
         * \brief Compute the ground plane coefficients from the transform between two reference frames.
         *
         * \param[in] worldToCamTransform ROS transform between world frame and camera frame.
         *
         * \return Vector of ground plane coefficients.
         */
        Eigen::VectorXf
        computeFromTF (tf::Transform worldToCamTransform);

        /**
         * \brief Read the world to camera transform from file.
         *
         * \param[in] filename Filename listing camera poses for each camera.
         * \param[in] camera_name Name of the camera for which the pose should be read.
         *
         * \return The world to camera transform.
         */
        tf::Transform
        readTFFromFile (std::string filename, std::string camera_name);

        /**
         * \brief Compute the ground plane coefficients.
         *
         * \return Vector of ground plane coefficients.
         */
        Eigen::VectorXf
        compute ();

        /**
         * \brief Compute the ground plane coefficients with the procedure used in multi-camera systems.
         * \param[in] ground_from_extrinsic_calibration If true, exploit extrinsic calibration for estimatin the ground plane equation.
         * \param[in] read_ground_from_file Flag stating if the ground should be read from file, if present.
         * \param[in] pointcloud_topic Topic containing the point cloud.
         * \param[in] sampling_factor Scale factor used to downsample the point cloud.
         *
         * \return Vector of ground plane coefficients.
         */
        Eigen::VectorXf
        computeMulticamera (bool ground_from_extrinsic_calibration, bool read_ground_from_file, std::string pointcloud_topic,
            int sampling_factor, float voxel_size);

        /**
         * \brief Refine ground coefficients by iterating ground plane detection on the input cloud
         *
         * \param[in] cloud Input cloud.
         * \param[in] num_iter Number of iterations.
         * \param[in] inliers_threshold Distance threshold for selecting inliers.
         * \param[in/out] Ground coefficients.
         *
         * return true if ground coefficients have been updated, false otherwise.
         */
        bool
        refineGround (int num_iter, float voxel_size, float inliers_threshold, Eigen::VectorXf& ground_coeffs_calib);

      private:
        // removing all refs to visualization
        /**
         * \brief Callback listening to point clicking on PCL visualizer.
         *
         */
        //static void
        //pp_callback (const pcl::visualization::PointPickingEvent& event, void* args);

        /**
         * \brief Mouse clicking callback on OpenCV images.
         */
        static void
        click_cb (int event, int x, int y, int flags, void* args);

        /**
         * \brief States which planar region is lower.
         *
         * \param[in] region1 First planar region.
         * \param[in] region2 Second planar region.
         *
         * \return true if region2 is lower than region1.
         */
        static bool
        planeHeightComparator (pcl::PlanarRegion<PointT> region1, pcl::PlanarRegion<PointT> region2);

        /**
         * \brief Color the planar regions with different colors or the index-th planar region in red.
         *
         * \param[in] regions Vector of planar regions.
         * \param[in] index If set and > 0, specify the index of the region to be colored in red. If not set, all regions are colored with different colors.
         *
         * \return The colored point cloud.
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        colorRegions (std::vector<pcl::PlanarRegion<PointT>,
            Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions, int index = -1);

      protected:
        /** \brief Flag stating if ground should be estimated manually (0), semi-automatically (1) or automatically with validation (2) or fully automatically (3) */
        int ground_estimation_mode_;

        /** \brief Flag enabling manual ground selection via ssh */
        bool remote_ground_selection_;

        /** \brief pointer to the input cloud */
        PointCloudPtr cloud_;
        
        
        // rid all viz
        /** \brief structure used to pass arguments to the callback function */
        //struct callback_args{
        //    pcl::PointCloud<pcl::PointXYZ>::Ptr clicked_points_3d;
        //    pcl::visualization::PCLVisualizer* viewerPtr;
        //};

        /** \brief structure used to pass arguments to the callback function */
        //struct callback_args_color{
        //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clicked_points_3d;
        //    pcl::visualization::PCLVisualizer* viewerPtr;
        //};

        /** \brief structure used to pass arguments to the callback function associated to an image */
        struct callback_args_image{
            std::vector<cv::Point> clicked_points_2d;
            bool selection_finished;
        };
    };
  } /* namespace ground_segmentation */
} /* namespace open_ptrack */
#include <open_ptrack/ground_segmentation/ground_segmentation.hpp>
#endif /* OPEN_PTRACK_GROUND_SEGMENTATION_GROUND_SEGMENTATION_H_ */

