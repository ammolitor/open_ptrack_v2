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
#include <opt_msgs/TrackArray.h>
#include <opt_msgs/Association.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <dynamic_reconfigure/server.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


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
#include <nlohmann/json.hpp>
using json = nlohmann::json;
typedef sensor_msgs::Image Image;
typedef sensor_msgs::CameraInfo CameraInfo;
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

// THIS IS WHERE THE MAGIC HAPPENS, 
// WE TRANSFORM ALL POINTCLOUD POINTS RESPECTIVE TO THE WORLD VIEW
tf::Transform readTFFromFile(std::string filename, std::string camera_name)
{
  tf::Transform worldToCamTransform;

  ifstream poses_file;
  poses_file.open(filename.c_str());
  std::string line;
  std::string pose_string;
  while(getline(poses_file, line))
  {
    int pos = line.find(camera_name, 0);
    if (pos != std::string::npos)
    {
      pose_string = line.substr(camera_name.size() + 2, line.size() - camera_name.size() - 2);
    }
  }
  poses_file.close();

  // Create transform:
  std::vector<double> pose;
  pose.resize(7, 0.0);
  sscanf(pose_string.c_str(), "%lf %lf %lf %lf %lf %lf %lf", &pose[0], &pose[1], &pose[2], &pose[3], &pose[4], &pose[5], &pose[6]);
  worldToCamTransform.setOrigin(tf::Vector3(pose[0], pose[1], pose[2]));
  worldToCamTransform.setRotation(tf::Quaternion(pose[3], pose[4], pose[5], pose[6]));

  return worldToCamTransform;
}

tf::Transform read_poses_from_json(std::string camera_name)
{
  tf::Transform worldToCamTransform;

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
  double translation_x = pose_config[camera_name]["pose"]["translation"]["x"];
  double translation_y = pose_config[camera_name]["pose"]["translation"]["y"];
  double translation_z = pose_config[camera_name]["pose"]["translation"]["z"];
  double rotation_x = pose_config[camera_name]["pose"]["rotation"]["x"];
  double rotation_y = pose_config[camera_name]["pose"]["rotation"]["y"];
  double rotation_z = pose_config[camera_name]["pose"]["rotation"]["z"];
  double rotation_w = pose_config[camera_name]["pose"]["rotation"]["w"];
  
  worldToCamTransform.setOrigin(tf::Vector3(translation_x, translation_y, translation_z));
  worldToCamTransform.setRotation(tf::Quaternion(rotation_x, rotation_y, rotation_z, rotation_w));

  return worldToCamTransform;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

// positions pointer
struct callback_args_image {
  cv::Point P1;
  cv::Point P2;
  cv::Rect cropRect;
  bool clicked;
  bool selection_finished;
  std::vector<cv::Point> clicked_points_2d;
};

void box_cb(int event, int x, int y, int flags, void* args)
{
  //cv::Mat src,img,ROI;
  //cv::Rect cropRect(0,0,0,0);
  //cv::Point P1(0,0);
  //cv::Point P2(0,0);
  //struct callback_args_image* data = (struct callback_args_image *)args;
  callback_args_image* data = (callback_args_image *)args;

  switch(event)
    {
        case  CV_EVENT_LBUTTONDOWN:
            data->clicked=true;
            data->P1.x=x;
            data->P1.y=y;
            data->P2.x=x;
            data->P2.y=y;
            break;

        case  CV_EVENT_LBUTTONUP:
            data->P2.x=x;
            data->P2.y=y;
            data->clicked=false;
            break;

        case  CV_EVENT_MOUSEMOVE:
            if(data->clicked){
                data->P2.x=x;
                data->P2.y=y;
            }
            break;

        default:
            break;
    }

  if(data->clicked){
        if(data->P1.x>data->P2.x){ data->cropRect.x=data->P2.x;
                      data->cropRect.width=data->P1.x-data->P2.x; }
        else {         data->cropRect.x=data->P1.x;
                      data->cropRect.width=data->P2.x-data->P1.x; }

        if(data->P1.y>data->P2.y){ data->cropRect.y=data->P2.y;
                      data->cropRect.height=data->P1.y-data->P2.y; }
        else {         data->cropRect.y=data->P1.y;
                      data->cropRect.height=data->P2.y-data->P1.y; }
    }

  switch (flags)
  {
  case CV_EVENT_FLAG_SHIFTKEY:
  {
    data->selection_finished = true;
    break;
  }
  case CV_EVENT_FLAG_CTRLKEY:
  {
    data->selection_finished = true;
    break;
  }
  }
}

struct cvcallback_args_image {
  cv::Point P1;
  cv::Point P2;
  cv::Rect cropRect;
  cv::Rect box;
  bool clicked;
  bool selection_finished;
  bool isDrawing;
  std::vector<cv::Point> clicked_points_2d;
};

void opencv_mouse_callback(int event, int x, int y,  int flags, void* args){
  //handlerT * data = (handlerT*)param;
  cvcallback_args_image* data = (cvcallback_args_image *)args;
  switch( event ){
    // update the selected bounding box
    case EVENT_MOUSEMOVE:
      if( data->isDrawing ){
        //if(data->drawFromCenter){
        //  data->box.width = 2*(x-data->center.x)/*data->box.x*/;
        //  data->box.height = 2*(y-data->center.y)/*data->box.y*/;
        //  data->box.x = data->center.x-data->box.width/2.0;
        //  data->box.y = data->center.y-data->box.height/2.0;
        //}else{
          data->box.width = x-data->box.x;
          data->box.height = y-data->box.y;
        //}
      }
    break;

    // start to select the bounding box
    case EVENT_LBUTTONDOWN:
      data->isDrawing = true;
      data->box = cv::Rect( x, y, 0, 0 );
      //data->center = Point2f((float)x,(float)y);
    break;

    // cleaning up the selected bounding box
    case EVENT_LBUTTONUP:
      data->isDrawing = false;
      if( data->box.width < 0 ){
        data->box.x += data->box.width;
        data->box.width *= -1;
      }
      if( data->box.height < 0 ){
        data->box.y += data->box.height;
        data->box.height *= -1;
      }
    break;
  }
  switch (flags){
    case CV_EVENT_FLAG_SHIFTKEY:
    {
      data->selection_finished = true;
      break;
    }
    case CV_EVENT_FLAG_CTRLKEY:
    {
      data->selection_finished = true;
      break;
    }
  }
}

//std::string GetEnv( const std::string & var ) {
//     const char * val = std::getenv( var.c_str() );
//     if ( val == nullptr ) { // invalid to assign nullptr to std::string
//         return "";
//     }
//     else {
//         return val;
//     }
//}

std::string getEnvVar(std::string const& key)
{
    char const* val = getenv(key.c_str()); 
    return val == NULL ? std::string() : std::string(val);
}

/**
 * @brief The AreaDefinitionNode
 */
class AreaDefinitionNode {
  private:
    ros::NodeHandle node_;
    
    //image specific
    tf::TransformListener tf_listener;
    tf::Transform worldToCamTransform;
    //image_transport::ImageTransport image_transport;
    //PointCloudT::Ptr cloud_;//(new PointCloudT);
    //PointCloudT::Ptr cloud_(new PointCloudT);
    bool new_cloud_available_flag = false;
    // Publishers
    ros::Publisher result_pub;
    ros::Publisher point_cloud_publish;
    ros::Publisher image_pub;


    // Subscribers
    ros::Subscriber rgb_sub;
    ros::Subscriber camera_info_sub;
    ros::Subscriber detector_sub;
    ros::Subscriber camera_info_matrix;
    //ros::Subscriber cloud_sub;

      // Message Filters
    message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub;
    message_filters::Subscriber<sensor_msgs::Image> depth_image_sub;
    //pcl cloud?????
    message_filters::Subscriber<opt_msgs::DetectionArray> detections_sub;
    message_filters::Subscriber<opt_msgs::DetectionArray> hand_detections_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info;
    message_filters::Subscriber<opt_msgs::TrackArray> tracks_sub;
    message_filters::Subscriber<opt_msgs::TrackArray> track_correction_sub;
    message_filters::Subscriber<opt_msgs::Association> association_sub;
    message_filters::Subscriber<PointCloudT> cloud_sub;

    // Time sync
    //detections_sub, hand_detections_sub, tracking_sub, tracks_sub, track_correction_sub
    //typedef ApproximateTime<opt_msgs::DetectionArray, opt_msgs::DetectionArray, opt_msgs::TrackArray, opt_msgs::TrackArray, opt_msgs::Association> ApproximatePolicy;
    //typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    //boost::shared_ptr<ApproximateSync> approximate_sync_;
    
    //seconday sync??????
    typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, PointCloudT> ImageApproximatePolicy;
    typedef message_filters::Synchronizer<ImageApproximatePolicy> ImageApproximateSync;
    boost::shared_ptr<ImageApproximateSync> image_approximate_sync_;

    // Image to "world" transforms
    Eigen::Affine3d world2rgb;
    Eigen::Affine3d ir2rgb;
    tf::StampedTransform world2rgb_transform;
    tf::StampedTransform ir2rgb_transform;

    // name of the given sensor
    std::string sensor_name;

    // HARDCODED AREA THRESHOLDS
    std::map<std::string, std::pair<double, double>> area_thres_;

  public:
    // Set camera matrix transforms
    Eigen::Matrix3d cam_intrins_;
    Eigen::Matrix3f intrinsics_matrix;
    bool camera_info_available_flag = false;
    double _cx;
    double _cy;
    double _constant_x;
    double _constant_y;
    vector<tf::Vector3> worldpoints;
    PointCloudT::Ptr cloud_;
    int n_zones;
    int zone_id;
    //PointCloudT::Ptr cloud_(new PointCloudT);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZRGB>);
    /**
     * @brief constructor
     * @param nh node handler
     * @param sensor_string converts to the sensor_name variable
     * @param detections_topic the name of the detections topic to subscribe to
     */
    AreaDefinitionNode(ros::NodeHandle& nh, std::string sensor_string, std::string detections_topic, int N_zones):
        node_(nh)
    {
      //PointCloudT::Ptr cloud_(new PointCloudT);
      // Published Messages
      // TODO - what to publish, and where?????
      result_pub = node_.advertise<opt_msgs::DetectionArray>("/results/results", 3);
      point_cloud_publish = node_.advertise<sensor_msgs::PointCloud2>("detect_result_cloud", 1);
      image_pub = node_.advertise<sensor_msgs::Image>("image_result", 1);

      // Subscribe to Messages
      rgb_image_sub.subscribe(node_, sensor_string +"/color/image_rect_color", 1);
      depth_image_sub.subscribe(node_, sensor_string+"/depth/image_rect_raw", 1);
      detections_sub.subscribe(node_, detections_topic, 10);
      hand_detections_sub.subscribe(node_, "/hand_detector/detections", 10);
      tracks_sub.subscribe(node_, "/tracker/tracks_smoothed", 10);
      track_correction_sub.subscribe(node_, "/face_recognition/people_tracks", 10);
      association_sub.subscribe(node_, "/tracker/association_result", 10);
      cloud_sub.subscribe(node_, sensor_string + "/depth_registered/points", 10);

      // get 
      camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &AreaDefinitionNode::camera_info_callback, this);
      //cloud_sub = node_.subscribe(sensor_string + "/depth_registered/points", 1, &AreaDefinitionNode::cloud_cb, this);

      // camera time sync 
      image_approximate_sync_.reset(new ImageApproximateSync(ImageApproximatePolicy(10), rgb_image_sub, depth_image_sub, cloud_sub));
      image_approximate_sync_->registerCallback(boost::bind(&AreaDefinitionNode::area_callback, this, _1, _2, _3));

      // TODO - manage incoming detections
      //approximate_sync_.reset(new ApproximateSync(
      //    ApproximatePolicy(10), detections_sub, hand_detections_sub, tracks_sub, track_correction_sub, association_sub));
      //approximate_sync_->registerCallback(boost::bind(&AreaDefinitionNode::callback, this, _1, _2, _3, _4, _5));
  
      sensor_name = sensor_string;

      // not sure of the exact structure of this..
      area_thres_["person"] = pair<double, double>(1.8, 0.5);
      //area_thres_["car"] = pair<double, double>(2.7, 1.8);

      // Read worldToCam transform from file:
      //std::string filename = ros::package::getPath("detection") + "/launch/camera_poses.txt";
      //worldToCamTransform = readTFFromFile (filename, sensor_name);
      worldToCamTransform = read_poses_from_json(sensor_name);
      n_zones = N_zones;
      int zone_id = 0;
    }

    void camera_info_callback(const CameraInfo::ConstPtr & msg){
    //printf("running camera_info_callback\n");
        intrinsics_matrix << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
        cam_intrins_ << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1; 
        _cx = msg->K[2];
        _cy = msg->K[5];
        _constant_x =  1.0f / msg->K[0];
        _constant_y = 1.0f /  msg->K[4];
        camera_info_available_flag = true;
        //might have to do it here?
        //cam_intrins_.transposeInPlace();
    }
    void cloud_cb (const PointCloudT::ConstPtr& callback_cloud)
    {
    printf("running cloud_cb\n");
    PointCloudT::Ptr cloud_(new PointCloudT);
    *cloud_ = *callback_cloud;

    // single_camera_tracking_node_azure.launch sets depth_mode to NFOV_UNBINNED = 640x576.
    const int azure_width = 640;
    const int azure_height = 576;
    if (cloud_->height == 1 && cloud_->width == azure_width * azure_height) {
        // The Azure provides a 1D array. Simply fix the width and height to be 2D.
        cloud_->width = azure_width;
        cloud_->height = azure_height;
    }

    new_cloud_available_flag = true;

    // static tf::TransformBroadcaster br;
    // tf::Vector3 tran_input = tf::Vector3(0,0,0);
    // tf::Quaternion quat_input = tf::Quaternion(0,0,0,1);

    // tf::Transform transform;
    // transform.setOrigin(tran_input);
    // transform.setRotation(quat_input);

    // pcl_conversions::toPCL(ros::Time::now(), cloud->header.stamp);

    // cloud->header.frame_id = "new" + callback_cloud->header.frame_id;
    // br.sendTransform(tf::StampedTransform(transform, callback_cloud->header.stamp, cloud->header.frame_id, callback_cloud->header.frame_id));

    }

  private:

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
    bool filterBboxByArea(const cv::Rect rect, double range)
    {
        int xmin = rect.x;
        int ymin = rect.y;
        int xmax = rect.x + rect.width;
        int ymax = rect.y + rect.height;
        
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

  void area_callback(const sensor_msgs::Image::ConstPtr& rgb_image,
                      const sensor_msgs::Image::ConstPtr& depth_image,
                      const PointCloudT::ConstPtr& cloud_) {
    //ros::spinOnce();
    std::cout << "running algorithm callback\n" << std::endl;
    // If extrinsic calibration is not available:
    //if (!extrinsic_calibration)
    //{ // Set fixed transformation from rgb frame and base_link
    //  tf::Vector3 fixed_translation(0, 0, 0);                  // camera_rgb_optical_frame -> world
    //  tf::Quaternion fixed_rotation(-0.5, 0.5, -0.5, -0.5);	// camera_rgb_optical_frame -> world
    //  tf::Vector3 inv_fixed_translation(0.0, 0.0, 0);			// world -> camera_rgb_optical_frame
    //  tf::Quaternion inv_fixed_rotation(-0.5, 0.5, -0.5, 0.5);	// world -> camera_rgb_optical_frame
    //sensor_msg.transform = sensor_msg.transform + [Transform(Vector3(t['x'], t['y'], t['z']), Quaternion(r['x'], r['y'], r['z'], r['w']))]
        
    //  world_to_camera_frame_transform = tf::Transform(fixed_rotation, fixed_translation);
    //  camera_frame_to_world_transform = tf::Transform(inv_fixed_rotation, inv_fixed_translation);
    //}

    //int num_cameras = 0;
    //XmlRpc::XmlRpcValue network;
    //nh.getParam("network", network);
    //std::map<std::string, XmlRpc::XmlRpcValue>::iterator i;
    //for (unsigned i = 0; i < network.size(); i++)
    //{
    //  num_cameras += network[i]["sensors"].size();
    //}

    //tf_listener.waitForTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
    //tf_listener.lookupTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ir2rgb_transform);
    //tf_listener.waitForTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
    //tf_listener.lookupTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), world2rgb_transform);

    //std_msgs::Header cloud_header = pcl_conversions::fromPCL(cloud->header);
    json zone_json;
    sensor_msgs::PointCloud2 msg_pointcloud;
    // transform to eigen
    //tf::transformTFToEigen(world2rgb_transform, world2rgb);
    // open_ptrack::opt_utils::Conversions converter;

    cv::Rect area_of_interest(0, 0, 0, 0);

    // add callback vars
    cv::Mat src, img, ROI;
    // convert rgb_image and depth_image to pointers
    cv_bridge::CvImagePtr cv_ptr_rgb;
    cv_bridge::CvImage::Ptr  cv_ptr_depth;
    cv::Mat cv_image;
    cv::Mat cv_depth_image;
    cv::Mat src_img;

    // set detection variables here
    cv::Size image_size;
    float height;
    float width;

    cv_ptr_rgb = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
    cv_image = cv_ptr_rgb->image;
    src_img  = cv_image.clone();
    cv_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
    cv_depth_image = cv_ptr_depth->image;
    image_size = cv_image.size();
    height =  static_cast<float>(image_size.height);
    width =  static_cast<float>(image_size.width);

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

    //substitute for above??
    //for (int i=0;i<cloud_->height;i++)
    //{
    //    for (int j=0;j<cloud_->width;j++)
    //    {
    //    pcl_cloud->at(j,i).x = cloud_->at(j,i).x;
    //    pcl_cloud->at(j,i).y = cloud_->at(j,i).y;
    //    pcl_cloud->at(j,i).z = cloud_->at(j,i).z;
    //    }
    //}

    // I think this part is failing, bc pcl_cloud hasn't been filled with anything

    // define xyz 3d points in cloud
    Eigen::MatrixXd points_3d_in_cam(3, pcl_cloud->size());
    for(int i = 0; i < pcl_cloud->size(); i++)
    {
        points_3d_in_cam(0, i) = (*pcl_cloud)[i].x;
        points_3d_in_cam(1, i) = (*pcl_cloud)[i].y;
        points_3d_in_cam(2, i) = (*pcl_cloud)[i].z;
    }    

    // alternative method?
    // define xyz 3d points in cloud
    //Eigen::MatrixXd points_3d_in_cam(3, pcl_cloud->size());
    //for(int i = 0; i < pcl_cloud->size(); i++)
    //{
    //    points_3d_in_cam(0, i) = cloud_xyzrgb->points[i].x;
    //    points_3d_in_cam(1, i) = cloud_xyzrgb->points[i].y;
    //    points_3d_in_cam(2, i) = cloud_xyzrgb->points[i].z;
    //}    


    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr clicked_points_3d (new pcl::PointCloud<pcl::PointXYZRGB>);

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
    std::cout << "N Zones: "<< n_zones << std::endl;
    std::string zone_string;

    for (int zone_id = 0; zone_id < n_zones; zone_id++) 
    {
      //zone_zone_idd = zone_id;
      zone_string = std::to_string(zone_id);
      std::cout << "Click and drag for Selection\n" << std::endl;
      std::cout << "\n" << std::endl;
      std::cout << "------> Press 'shift+enter' to save\n" << std::endl;

      // Add point picking callback to viewer:
      cv::Mat curr_image_clone;
      std::vector<cv::Point> clicked_points_2d;
      bool selection_finished = false;
      bool clicked = false;
      cv::Rect cropRect(0, 0, 0, 0);
      cv::Point P1(0, 0);
      cv::Point P2(0, 0);
      //struct callback_args_image cb_args;
      cvcallback_args_image cb_args;
      cb_args.clicked_points_2d = clicked_points_2d;
      cb_args.selection_finished = selection_finished;
      cb_args.isDrawing = false;
      cb_args.box = cv::Rect(0, 0, 0, 0);
      cb_args.P1 = P1;
      cb_args.P2 = P2;
      cb_args.cropRect = cropRect;
      cb_args.clicked = clicked;
      curr_image_clone = curr_image.clone();
      cv::namedWindow("Draw a box around the area of interest");
      cv::setMouseCallback("Draw a box around the area of interest", opencv_mouse_callback, (void*)&cb_args);
      cv::imshow("Draw a box around the area of interest", curr_image_clone);
      cv::waitKey(1);

      // Select the box from the image:
      while(!cb_args.selection_finished)
      {
          //char c=waitKey();
          curr_image_clone = curr_image.clone();
          cv::Rect drect = cb_args.box;        
          cv::rectangle(curr_image_clone, drect, Scalar(0, 255, 0), 1, 8, 0);
          cv::imshow("Draw a box around the area of interest", curr_image_clone);
          cv::waitKey(1);
      }
      std::cout << "DEBUG: box finished" << std::endl;
      //cv::waitKey(1);
      //}

      // Select the corresponding 3D points from the point cloud:
      cv::Point p1 = cb_args.P1;
      cv::Point p2 = cb_args.P2;
      cv::Rect rect = cb_args.box;
      std::cout << "DEBUG: rect x: " << rect.x << std::endl;
      std::cout << "DEBUG: rect y: " << rect.y << std::endl;
      std::cout << "DEBUG: rect width: " << rect.width << std::endl;
      std::cout << "DEBUG: rect height: " << rect.height << std::endl;
      bool points_3d_in_cam_is_empty = points_3d_in_cam.isZero(0);
      std::cout << "DEBUG: points_3d_in_cam_is_empty: " << points_3d_in_cam_is_empty << std::endl;
      // get the bounding box of the area,

      // define this, but maybe do like the camera transform here????
      Eigen::MatrixXd points_2d_homo = cam_intrins_ * points_3d_in_cam;

      // lets assume that points_2d_homo == world transform...

      Eigen::MatrixXd points_2d(2, pcl_cloud->size());
      for(int i = 0; i < pcl_cloud->size(); i++)
      {
          points_2d(0, i) = points_2d_homo(0, i) / points_2d_homo(2, i);
          points_2d(1, i) = points_2d_homo(1, i) / points_2d_homo(2, i);
      }

      std::cout << "DEBUG: points set" << std::endl;
      // define cam_intrins and camera_img
      pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      // cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(camera_img);
      std::cout <<"DEBUG: out_cloud set" << std::endl;
      // get the points w.r.t. 3d cloud
      vector<Point3f> points;
      //vector<Point3f> worldpoints;
      vector<tf::Vector3> worldpoints;
      Point3f tmp;
      Point3f world_to_temp;
      //json zone_json;

      // will this even work????
      //try {
      //  std::string std::string package_path = ros::package::getPath("recognition");
      //  std::string master_hard_coded_path = std::string package_path + "/cfg/area.json";
      //  std::ifstream json_read(master_hard_coded_path);
      //  json_read >> zone_json;
      //  try {
      //    int zoneid = 0;
      //    for(int i = 0; i < zoneid; i++)
      //    {
      //      auto s2 = zone_json[zoneid];
      //    }
      //  } catch {
      //      //     json zone_json;
      //      std::cout << "starting at zone id: " << zoneid << std:endl;
      //  }
      //} catch {
      //  int zoneid = 0;
      //  json zone_json;
      //};


    
      int cube_count = 0;
      for(int i = 0; i < pcl_cloud->size(); i++)
      { 
        if(points_2d(0, i) < rect.x + rect.width && points_2d(0, i) > rect.x 
          && points_2d(1, i) < rect.y + rect.height && points_2d(1, i) > rect.y)
          {
              // transform the point here?
              //current_point = worldToCamTransform(points_3d_in_cam(0, i));
              tmp.x = static_cast<float>(points_3d_in_cam(0, i));
              tmp.y = static_cast<float>(points_3d_in_cam(1, i));
              tmp.z = static_cast<float>(points_3d_in_cam(2, i));

              world_to_temp.x =  static_cast<float>(tmp.x);
              world_to_temp.y =  static_cast<float>(tmp.y);
              world_to_temp.z =  static_cast<float>(tmp.z);

              tf::Vector3 current_world_point(world_to_temp.x, world_to_temp.y, world_to_temp.z);
              current_world_point = worldToCamTransform(current_world_point);

              // get x, y, z of current point

              // transform point here
              points.push_back(tmp);
              worldpoints.push_back(current_world_point);
              // push all points into a json file to define the area.
              // error: invalid use of non-static member function ‘const tfScalar& tf::Vector3::x() const’
              zone_json[zone_string][sensor_name]["world"][cube_count]["x"] = current_world_point.getX();
              zone_json[zone_string][sensor_name][sensor_name][cube_count]["x"] = tmp.x;
              zone_json[zone_string][sensor_name]["world"][cube_count]["y"] = current_world_point.getY();
              zone_json[zone_string][sensor_name][sensor_name][cube_count]["y"] = tmp.y;
              zone_json[zone_string][sensor_name]["world"][cube_count]["z"] = current_world_point.getZ();
              zone_json[zone_string][sensor_name][sensor_name][cube_count]["z"] = tmp.z;            
              cube_count++;
          }
      }

      // TODO
      // transform rect to world rect here
      // save either worldpoints
      // save points
      // https://stackoverflow.com/questions/19074380/how-to-save-stdvectorkeypoint-to-a-text-file-in-c

      std::cout << "DEBUG:  finished - points size: " << points.size() << std::endl;
      vector<Point3f> points_fg = clusterPoints(points);
      // vector<Point3f> points_fg = points;
      Point3d min_xyz(10000, 10000, 10000), max_xyz(-10000, -10000, -10000);
      for(int i = 0; i < points_fg.size(); i++)
      {
          pcl::PointXYZ tmp_pt;
          tmp_pt.x = points_fg[i].x;
          tmp_pt.y = points_fg[i].y;
          tmp_pt.z = points_fg[i].z;
          out_cloud->push_back(tmp_pt);
          
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
      std::cout << "DEBUG: clustering finished" << std::endl;
      if(! filterBboxByArea(rect, (min_xyz.z + max_xyz.z) / 2))
      {
          cv::rectangle(src_img, cv:: Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(30,07,197), 3);
          cv::putText(src_img, "fake", cv::Point(rect.x + 5, rect.y + 25), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 255));
          std::cout << "DEBUG: filterBboxByArea finished" << std::endl;
      }
      else
          {
          double ave_x = (max_xyz.x + min_xyz.x) / 2;
          double ave_y = (max_xyz.y + min_xyz.y) / 2;
          double ave_z = (max_xyz.z + min_xyz.z) / 2;
          
          drawCube(src_img, min_xyz, max_xyz);
          cv::putText(src_img, "area", cv::Point(rect.x + 5, rect.y + 25), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 255, 255));

          char loc_str[30];
          sprintf(loc_str, "%.2f, %.2f, %.2f", ave_x, ave_y, ave_z);
          cv::putText(src_img, loc_str, cv::Point(rect.x + 15, rect.y - 25), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
          std::cout << "DEBUG: else filterBboxByArea finished" << std::endl;
          }
      

      // take max/min and convert to world view
      tf::Vector3 max_xyz_world_point(max_xyz.x, max_xyz.y, max_xyz.z);
      max_xyz_world_point = worldToCamTransform(max_xyz_world_point);
      tf::Vector3 min_xyz_world_point(min_xyz.x, min_xyz.y, min_xyz.z);
      min_xyz_world_point = worldToCamTransform(min_xyz_world_point);

      zone_json[zone_string][sensor_name]["min"]["world"]["x"] = min_xyz_world_point.getX();
      zone_json[zone_string][sensor_name]["min"][sensor_name]["x"] = min_xyz.x;
      zone_json[zone_string][sensor_name]["min"]["world"]["y"] = min_xyz_world_point.getY();
      zone_json[zone_string][sensor_name]["min"][sensor_name]["y"] = min_xyz.y;
      zone_json[zone_string][sensor_name]["min"]["world"]["z"] = min_xyz_world_point.getZ();
      zone_json[zone_string][sensor_name]["min"][sensor_name]["z"] = min_xyz.z;     

      zone_json[zone_string][sensor_name]["max"]["world"]["x"] = max_xyz_world_point.getX();
      zone_json[zone_string][sensor_name]["max"][sensor_name]["x"] = max_xyz.x;
      zone_json[zone_string][sensor_name]["max"]["world"]["y"] = max_xyz_world_point.getY();
      zone_json[zone_string][sensor_name]["max"][sensor_name]["y"] = max_xyz.y;
      zone_json[zone_string][sensor_name]["max"]["world"]["z"] = max_xyz_world_point.getZ();
      zone_json[zone_string][sensor_name]["max"][sensor_name]["z"] = max_xyz.z;  

      // destroy the named window
      cv::destroyAllWindows(); 
    }
    
    cv::destroyAllWindows(); 
    std::cout << "DEBUG: about to show image" << std::endl;
    cv::imshow("disp", src_img);
      
    // saving image
    std::string filename = "/area.jpg";
    std::string home_dir = getEnvVar("HOME");
    //std::string home_dir = std::string(env);
    std::string filepath = home_dir + filename;
    std::cout << "DEBUG: saving image to: " << filepath << std::endl;
    cv::imwrite(filepath, src_img);

    cv::waitKey(1);
    std::cout << "DEBUG: src finished" << std::endl;

    // save area cube to file
    std::string area_path = ros::package::getPath("recognition");
    std::string zone_json_path = area_path + "/cfg/area.json";
    std::ofstream areafile(zone_json_path);
    areafile << std::setw(4) << zone_json << std::endl;

    // output cloud to file
    //sensor_msgs::PointCloud2 out_cloud_ros;
    //pcl::toROSMsg(*out_cloud, out_cloud_ros);
    //out_cloud_ros.header.frame_id = "world";
    //out_cloud_ros.header.stamp = cv_ptr_rgb->header.stamp;
    //point_cloud_publish.publish(out_cloud_ros);

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", src_img).toImageMsg();
    msg->header.stamp = cv_ptr_rgb->header.stamp;
    image_pub.publish(msg);
    //return rect;    
    // shut down ros node
    
    cv::destroyAllWindows();
    ros::shutdown();
    }
};

int main(int argc, char** argv) {
  // read json parameters
  std::string sensor_name;
  std::string detections_topic;
  json master_config;
  int n_zones;
  std::string package_path = ros::package::getPath("recognition");
  std::string master_hard_coded_path = package_path + "/cfg/master.json";
  std::ifstream json_read(master_hard_coded_path);
  json_read >> master_config;
  sensor_name = master_config["sensor_name"]; //the path to the detector model file
  detections_topic = master_config["main_detections_topic"];
  n_zones = master_config["n_zones"];

  std::cout << "--- area_node ---" << std::endl;
  ros::init(argc, argv, "area_node");
  //make sure this call is correct
  ros::NodeHandle nh;
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "detections_topic: " << detections_topic << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  AreaDefinitionNode node(nh, sensor_name, detections_topic, n_zones);
  std::cout << "detection node init " << std::endl;
  ros::spin();
  return 0;
}
