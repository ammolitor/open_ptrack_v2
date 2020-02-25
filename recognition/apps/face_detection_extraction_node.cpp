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

#include <opt_msgs/FeatureVectorArray.h>
#include <opt_msgs/DetectionArray.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>

#include <dynamic_reconfigure/server.h>

#include <open_ptrack/hungarian/Hungarian.h>
#include <open_ptrack/facetracking.hpp>
#include <open_ptrack/FacePreprocess.h>
#include <nlohmann/json.hpp>


using json = nlohmann::json;
typedef sensor_msgs::Image Image;
typedef sensor_msgs::CameraInfo CameraInfo;
using namespace message_filters::sync_policies;


struct ROI {
  int left;
  int top;
  int left_width; //left + width
  int top_height; //top + height
};

struct regions_of_interest{
  ROI* rois;
  int num;
};

struct positionxy {
  int x;
  int y;
  int z;
  int w;
};

struct FaceMat{
  cv::Mat positions;
  int num;
};

struct aligned_face {
  sensor_msgs::ImagePtr image_msg;
};

struct image_messages_struct{
  aligned_face* aligned_faces;
  int num;
};

Mat Zscore(const Mat &fc) {
    /**
     * This is a normalize function before calculating the cosine distance. Experiment has proven it can destory the
     * original distribution in order to make two feature more distinguishable.
     */
    Mat mean, std;
    cv::meanStdDev(fc, mean, std);
    //    cout << mean << std << endl;
    Mat fc_norm = (fc - mean) / std;
    return fc_norm;
}

inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2) {
    /**
     * This module is using to computing the cosine distance between input feature and ground truth feature
     */
    double dot = v1.dot(v2);
    double denom_v1 = cv::norm(v1);
    double denom_v2 = cv::norm(v2);
    return dot / (denom_v1 * denom_v2);
}

int calc_roi(const CameraInfo::ConstPtr& rgb_info_msg, float w, cv::Mat xyz, cv::Point2f uv, ROI &roi){ 
    float half_w;
    int left;
    int top;
    int right;
    int bottom;
    int width;
    int height;

    half_w = w * rgb_info_msg->K[0] / xyz.at<double>(2);
    left = static_cast<int>(uv.x - half_w);
    top = static_cast<int>(uv.y - half_w);
    right = static_cast<int>(uv.x - half_w);
    bottom = static_cast<int>(uv.y - half_w);

    left = std::min(static_cast<int>(rgb_info_msg->width), std::max(0, left));
    top = std::min(static_cast<int>(rgb_info_msg->height), std::max(0, top));
    right = std::min(static_cast<int>(rgb_info_msg->width), std::max(0, right));
    bottom = std::min(static_cast<int>(rgb_info_msg->height), std::max(0, bottom));
    
    width = std::max(0, right - left);
    height = std::max(0, bottom - top);

    roi.left = left;
    roi.top = top;
    roi.left_width = left + width;
    roi.top_height = top + height;
    return 0;
}

cv::Point calculate_centroid_roi(ROI &roi){
  cv::Point tl, br;
  tl = cv::Point(roi.left, roi.top);
  br = cv::Point(roi.left_width, roi.top_height);
  return (tl + br) / 2;
}

cv::Point calculate_centroid_face(int xmin, int ymin, int xmax, int ymax){
  cv::Point tl, br;
  tl = cv::Point(xmin, ymin);
  br = cv::Point(xmax, ymax);
  return (tl + br) / 2;
}

// converts cv mat to f32 ma
std_msgs::Float32MultiArray CVMat2F32MA(cv::Mat& mat){
    std::vector<float> tempVec(mat.begin<float>(), mat.end<float>());
    std_msgs::Float32MultiArray fma;
    fma.data.clear();
    fma.data = tempVec;
    fma.layout.dim.resize(2);
    fma.layout.dim[0].label = "rows";
    fma.layout.dim[0].size = mat.rows;
    fma.layout.dim[0].stride = mat.rows * mat.cols;
    fma.layout.dim[1].label = "cols";
    fma.layout.dim[1].size = mat.rows;
    fma.layout.dim[1].stride = mat.cols;
    return fma;
}

std_msgs::Float32MultiArray init_empty_multiarray(){
    std_msgs::Float32MultiArray fma;
    fma.data.clear();
    fma.layout.dim.resize(2);
    fma.layout.dim[0].label = "rows";
    fma.layout.dim[0].size = 0;
    fma.layout.dim[0].stride = 0;
    fma.layout.dim[1].label = "cols";
    fma.layout.dim[1].size = 0;
    fma.layout.dim[1].stride = 0;
    return fma;
}

/**
 * @brief The FaceDetectionNode
 */
class FaceDetectionNode {
  private:
    ros::NodeHandle node_;
    std::unique_ptr<RetinaFaceDeploy> face_detector;
    std::unique_ptr<FR_MFN_Deploy> face_embedder;
    
    //image specific
    tf::TransformListener tf_listener;
    image_transport::ImageTransport image_transport;
    
    // ROS SERVERS
    dynamic_reconfigure::Server<recognition::FaceDetectionConfig> detector_cfg_server;
    dynamic_reconfigure::Server<recognition::FaceEmbeddingConfig> embedding_cfg_server;

    // Publishers
    ros::Publisher detector_pub;
    ros::Publisher detector_pub_local;
    ros::Publisher embedder_pub;
    ros::Publisher embedder_pub_local;
    //ros::Publisher image_pub;

    // Subscribers
    ros::Subscriber rgb_sub;
    ros::Subscriber camera_info_sub;
    ros::Subscriber detector_sub;

    // Message Filters
    message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub;
    message_filters::Subscriber<opt_msgs::DetectionArray> detections_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info;

    // Time sync
    typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, opt_msgs::DetectionArray> ApproximatePolicy;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    boost::shared_ptr<ApproximateSync> approximate_sync_;

    // Image to "world" transforms
    Eigen::Affine3d world2rgb;
    Eigen::Affine3d ir2rgb;
    tf::StampedTransform world2rgb_transform;
    tf::StampedTransform ir2rgb_transform;

    // Intrinsic Matrix vars
    Eigen::Matrix3f intrinsics_matrix;
    bool camera_info_available_flag = false;
    double _cx;
    double _cy;
    double _constant_x;
    double _constant_y; 
    std::string encoding;

    // name of the given sensor
    std::string sensor_name;

    // Configuration parameters
    std::string detector_model_folder_path; //the path to the face detector model file
    std::string embedder_model_folder_path; //the path to the face embedder model file
    double confidence_thresh; // the threshold for confidence of face detection
    double roi_width; //the width of a face detection ROI in the world space [m]
    bool calc_roi_from_top; // if true, ROIs are calculated from the top positions of detected clusters
    double head_offset_z_top; // the distance between the top position of a human cluster and the center of the face [m]
    double head_offset_z_centroid; // the distance between the centroid of a human cluster and the center of the face [m]
    int upscale_minsize; // the face detection ROI is upscaled so that its width get larger than #upscale_minsize
    bool visualization; // viz or not to viz

  public:
    /**
     * @brief constructor
     * @param nh node handler
     * @param sensor_string converts to the sensor_name variable
     * @param detections_topic the name of the detections topic to subscribe to
     */
    FaceDetectionNode(ros::NodeHandle& nh, std::string sensor_string, std::string detections_topic):
        node_(nh)
    {
      // Published Messages
      detector_pub = node_.advertise<opt_msgs::DetectionArray>("/face_detector/detections", 10);
      detector_pub_local = node_.advertise<opt_msgs::DetectionArray>("/face_detector/detections", 10);
      embedder_pub = node_.advertise<opt_msgs::FeatureVectorArray>("/face_feature_extractor/features", 10);
      embedder_pub_local = node_.advertise<opt_msgs::FeatureVectorArray>("/face_feature_extractor/features", 10);     

      // not sure how to do this...
      //image_transport::ImageTransport it(nh);
      //pub = it.advertise("yolo_object_detector/image", 1);

      // Subscribe to Messages
      rgb_image_sub.subscribe(node_, sensor_string +"/color/image_raw", 10);
      detections_sub.subscribe(node_, "/objects_detector/detections", 10);
      camera_info.subscribe(node_, sensor_string + "/color/camera_info", 10);

      // Camera time sync
      approximate_sync_.reset(new ApproximateSync(
          ApproximatePolicy(10), rgb_image_sub, camera_info, detections_sub));
      approximate_sync_->registerCallback(boost::bind(&FaceDetectionNode::callback, this, _1, _2, _3));

      // face_detector and embedder callbacks
      detector_cfg_server.setCallback(boost::bind(&FaceDetectionNode::detector_cfg_callback, this, _1, _2));
      embedding_cfg_server.setCallback(boost::bind(&FaceEmbeddingNode::embedder_cfg_callback, this, _1, _2));    
      //embedder_model_folder_path = "/home/nvidia/catkin_ws/src/open_ptrak/recognition/data/embedding_folder";
    
      // initialize detector and embedder
      face_detector.reset(new RetinaFaceDeploy(detector_model_folder_path));
      face_embedder.reset(new FR_MFN_Deploy(embedder_model_folder_path));
      sensor_name = sensor_string;
    }

  private:
    /**
     * @brief regions of interest calculator
     * this function mimics the algorithm from openptrack-python that takes each detection found
     * from the object detector (outside this constructor) and picks the part of the image where
     * it's likely a person's head is (the upper top of the given box). In this way, using a separate
     * process (this constructor) to find faces, we can match a region of interest to a given facebox
     * using the hungarian algorithm for centroid/distance matching.  
     * @param rgb_info_msg the rgb image message
     * @param detections_msg the detections message
     */
    regions_of_interest* calc_rois(const CameraInfo::ConstPtr& rgb_info_msg,
                                   const opt_msgs::DetectionArrayConstPtr& detections_msg){

      //ROI rois;
      // allocate memory for results
      regions_of_interest* rois = (regions_of_interest*)calloc(1, sizeof(regions_of_interest));
      rois->num = 100;
      rois->rois = (ROI*)calloc(100, sizeof(ROI));

      //face_positions.num = new_num;
      if (detections_msg->detections.size() == 0) {
        printf("no detections found: %ld\n", detections_msg->detections.size());
      rois->num = 0;
      return rois;
      }
      printf("detections found: %ld\n", detections_msg->detections.size());
      FaceMat face_mat;
      face_mat.positions = cv::Mat::zeros(100, 3, CV_32FC1);
      face_mat.num = 0;

      Eigen::MatrixXf camera_matrix;
      Eigen::MatrixXf distortion;
      cv::Mat rvec = cv::Mat::zeros(cv::Size(1, 3), CV_64FC1);
      cv::Mat tvec = cv::Mat::zeros(cv::Size(1, 3), CV_64FC1);

      int new_num;
      new_num = 0;
      face_mat.num = detections_msg->detections.size();
      face_mat.positions = cv::Mat::zeros(face_mat.num, 3, CV_32FC1);
      for (int i = 0; i < detections_msg->detections.size(); ++i){
          cv::Mat row;
          float top_pt[3];
          const auto& detection = detections_msg->detections[i];
          if (calc_roi_from_top){
            // APPLY THE HEADOFFSET CORRECTION HERE
            top_pt[0] = detection.top.x + world2rgb.translation().x();
            top_pt[1] = detection.top.y + world2rgb.translation().y();;
            top_pt[2] = detection.top.z + world2rgb.translation().z();;
            row = cv::Mat(1, 3, CV_32FC1, &top_pt);
          } else {
            top_pt[0] = detection.centroid.x + world2rgb.translation().x();;
            top_pt[1] = detection.centroid.y + world2rgb.translation().y();;
            top_pt[2] = detection.centroid.z + world2rgb.translation().z();;
            row = cv::Mat(1, 3, CV_32FC1, &top_pt);
          }
      face_mat.positions.push_back(row);
      }

      cv::Mat K;
      cv::Mat D;
      cv::Mat K_mat;
      K = Mat::zeros(3, 3, CV_64FC1);
      int i, j;
      for (i = 0; i < 3; i++){
        for (j = 0; j < 3; j++){
          K.at<double>(i, j) = rgb_info_msg->K[3 * i + j];
        }
      }

      D = Mat::zeros(1,5,CV_64FC1);
      for (i = 0; i < 5; i++){
        D.at<double>(i) = rgb_info_msg->D[i];
      }
    
      std::vector<cv::Point2f> projected;
      cv::projectPoints(face_mat.positions, rvec, tvec, K, D, projected);

      new_num = 0;
      for (int i = 0; i < detections_msg->detections.size(); ++i){
        calc_roi(rgb_info_msg, roi_width, face_mat.positions.row(i), projected[i], rois->rois[i]);
        new_num+=1;
      }
      rois->num = new_num;
      return rois;
    }


    void callback(const sensor_msgs::ImageConstPtr& rgb_image,
                    const CameraInfo::ConstPtr& rgb_info_msg,
                    const opt_msgs::DetectionArrayConstPtr& detections_msg) {
      printf("running algorithm callback");
      tf_listener.waitForTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
      tf_listener.lookupTransform(sensor_name + "_infra1_optical_frame", sensor_name + "_color_optical_frame", ros::Time(0), ir2rgb_transform);
      tf_listener.waitForTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), ros::Duration(3.0), ros::Duration(0.01));
      tf_listener.lookupTransform("/world", sensor_name + "_color_optical_frame", ros::Time(0), world2rgb_transform);

      // transform to eigen
      tf::transformTFToEigen(world2rgb_transform, world2rgb);
      tf::transformTFToEigen(ir2rgb_transform, ir2rgb);
    
      // set hungarian algorithm variables
      int r, c;
      HungarianAlgorithm HungAlgo;
      std::vector<cv::Point> face_centroids;
      std::vector<cv::Point> roi_centroids;
      std::vector<std::vector<double>> cost_matrix;
      std::vector<int> assignment;
      regions_of_interest *rois;
      
      // set image-message specific variables
      cv_bridge::CvImagePtr cv_ptr_rgb;
      cv::Mat cv_image;

      // set detections variables
      int ratio_x;
      int ratio_y;
      
      // set message specific variable
      opt_msgs::DetectionArray detection_array_msg;
      detection_array_msg = *detections_msg;
      opt_msgs::FeatureVectorArray feature_vector_array_msg;
      feature_vector_array_msg.header = detections_msg->header;
      std::vector<std_msgs::Float32MultiArray> features;
      std::vector<sensor_msgs::ImagePtr> images;      


      // convert image to opencv standard -> bgr (rgb swap happens inside network graph)
      cv_ptr_rgb = cv_bridge::toCvCopy(rgb_image,sensor_msgs::image_encodings::BGR8);
      cv_image = cv_ptr_rgb->image;

      // calculate the regions of interest based on the initial ROI algorithm from
      // openptrack.
      rois = calc_rois(rgb_info_msg, detections_msg);
      printf("roi numbers: %ld\n", rois->num);

      // hacky way of doing this???
      //int detection_index[100] = { -1 }; // all elements 0
    
      // run the face detector on the entire image 
      // NOTE: rather than running the detector for every region of
      // interest like openptrack-python, we run once and use the hungarian
      // algorithm to do the matching. 
      RetinaOutput output_ = face_detector->forward(cv_image);
      vector<Anchor> faceInfo = output_.result;
      printf("N faces detected: %ld\n", faceInfo.size());

      if (faceInfo.size() >= 1) {
        ratio_x = output_.ratio.x;
        ratio_y = output_.ratio.y; 

        // Calculate current centroids for each detection.
        for (int i = 0; i < rois->num; i++) {
          cv::Point centroid = calculate_centroid_roi(rois->rois[i]);
          roi_centroids.push_back(centroid);
        }
        printf("N roi_centroids: %ld\n", roi_centroids.size());
            
        // Calculate current centroids for faces.
        for (int i = 0; i < faceInfo.size(); i++) {
          int xmin = (int) faceInfo[i].finalbox.x * ratio_x;
          int ymin = (int) faceInfo[i].finalbox.y * ratio_y;
          // NOTE: WIDTH/HEIGHT are labeled INCORRECTLY 
          // and width == xmax, and height == ymax
          // TODO must fix this in the RetinaFace code
          int w = (int) faceInfo[i].finalbox.width * ratio_x;
          int h = (int) faceInfo[i].finalbox.height * ratio_y;
          int xmax = w;
          int ymax = h;
          cv::Point centroid = calculate_centroid_face(xmin, ymin, xmax, ymax);
          face_centroids.push_back(centroid);
        }
        printf("N face_centroids: %ld\n", face_centroids.size());
        
        // Initialize cost matrix for the hungarian algorithm
        for (int r = 0; r < roi_centroids.size (); r++) {
          std::vector<double> row;
          for (int c = 0; c < face_centroids.size (); c++) {
            float dist;
            dist = cv::norm(cv::Mat(face_centroids[c]), cv::Mat (roi_centroids[r]));
            row.push_back(dist);
          }
          cost_matrix.push_back(row);
        }
        printf("cost_matrix shape: %ld\n", cost_matrix.size());

        // Solve the Hungarian problem to match the distance of the roi centroid
        // to that of the bounding box
        HungAlgo.Solve(cost_matrix, assignment);
        printf("assignment shape: %ld\n", assignment.size());

        // gt face landmark
        float v1[5][2] = {
                {30.2946f, 51.6963f},
                {65.5318f, 51.5014f},
                {48.0252f, 71.7366f},
                {33.5493f, 92.3655f},
                {62.7299f, 92.2041f}};
        cv::Mat src(5, 2, CV_32FC1, v1);
        memcpy(src.data, v1, 2 * 5 * sizeof(float));

        int n_faces = 0;
        for (int i = 0; i < detections_msg->detections.size(); i++) {
          //std::vector<std_msgs::Float32MultiArray> feature_vector;
          std_msgs::Float32MultiArray multiarray;
          const auto& detection = &detection_array_msg.detections[i];
          if (assignment[i] == -1) {
            printf("Hungarian method: current detected face at position %d: %d will be ignored.\n", i, assignment[i]); 
            printf("zeroing detection box\n");
            detection->box_2D.x = 0;
            detection->box_2D.y = 0;
            detection->box_2D.width = 0; 
            detection->box_2D.height = 0;
            multiarray = init_empty_multiarray();
            features.push_back(multiarray);
            continue;
          } else {

            int xmin = (int) faceInfo[assignment[i]].finalbox.x * ratio_x;
            int ymin = (int) faceInfo[assignment[i]].finalbox.y * ratio_y;
            //TODO .width should be .xmax/.ymax ..
            // AS NOTED EARLIER WIDTH == XMAX, HEIGHT == YMAX DUE TO ERROR
            // IN THE WAY RETINAFACE WAS WRITTEN. TODO FIX
            int xmax = (int) faceInfo[assignment[i]].finalbox.width * ratio_x;
            int ymax = (int) faceInfo[assignment[i]].finalbox.height * ratio_y;

            // Perspective Transformation
            float v2[5][2] =
                    {{faceInfo[assignment[i]].pts[0].x, faceInfo[assignment[i]].pts[0].y},
                    {faceInfo[assignment[i]].pts[1].x, faceInfo[assignment[i]].pts[1].y},
                    {faceInfo[assignment[i]].pts[2].x, faceInfo[assignment[i]].pts[2].y},
                    {faceInfo[assignment[i]].pts[3].x, faceInfo[assignment[i]].pts[3].y},
                    {faceInfo[assignment[i]].pts[4].x, faceInfo[assignment[i]].pts[4].y},
                    };
            cv::Mat dst(5, 2, CV_32FC1, v2);
            memcpy(dst.data, v2, 2 * 5 * sizeof(float));
            
            // preprocess the image and transform using the perspective transform
            cv::Mat m = FacePreprocess::similarTransform(dst, src);
            cv::Mat aligned = cv_image.clone();
            cv::warpPerspective(cv_image, aligned, m, cv::Size(96, 112), INTER_LINEAR);
            resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);

            // run the face embedder on the input-aligned image
            Mat fc2 = face_embedder->forward(aligned);
            fc2 = Zscore(fc2);
            printf("face embedding completed!\n");
            // for debugging the output of fc2 only
            //std::cout << "fc2 (python)  = " << std::endl << cv::format(fc2, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
            std_msgs::Float32MultiArray multiarray = CVMat2F32MA(fc2);
            // update the feature array for face recognition
            features.push_back(multiarray);
            n_faces++;

            // update detection box in the same way the detection is updated
            // in python
            detection->box_2D.x = xmin;
            detection->box_2D.y = ymin;
            detection->box_2D.width = xmax - xmin; 
            detection->box_2D.height = ymax - ymin;

            // save an output of the image...
            // for debugging the aligned image
            //cv::Mat aligned_clone = aligned.clone();
            //for (int j = 0; j < faceInfo[assignment[i]].pts.size(); ++j) {
            //    if (j == 0 or j == 3) {
            //        cv::circle(aligned_clone, faceInfo[assignment[i]].pts[j], 3,
            //                   cv::Scalar(0, 255, 0),
            //                   cv::FILLED, cv::LINE_AA);
            //    } else {
            //        cv::circle(aligned_clone, faceInfo[assignment[i]].pts[j], 3,
            //                   cv::Scalar(0, 0, 255),
            //                   cv::FILLED, cv::LINE_AA);
            //    }
            //}
            // sensor_msgs::ImagePtr image_msg_aligned = cv_bridge::CvImage(std_msgs::Header(), "bgr8", aligned_clone).toImageMsg();
            //cv::rectangle(cv_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
            //cv::putText(cv_image, ss.str(), cv::Point(boxes->boxes[i].x+10,boxes->boxes[i].y+20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
            // cv::imwrite("/home/nvidia/OUTPUTIMAGE.JPG", cv_image);
          }
        }
        // update all the feature vectors from the detected faces
        feature_vector_array_msg.vectors = features;
      // this is for *if* there were no faces detected
      } else {
        // in the same way the python version zeros all detections
        // we'll zero all detections that don't contain a face...
        printf("zeroing 2D boxes for all non-face detections\n");
        for (int i = 0; i < detections_msg->detections.size(); i++) {
          std_msgs::Float32MultiArray multiarray;
          const auto& detection = &detection_array_msg.detections[i];
          detection->box_2D.x = 0;
          detection->box_2D.y = 0;
          detection->box_2D.width = 0; 
          detection->box_2D.height = 0;
          multiarray = init_empty_multiarray();
          features.push_back(multiarray);
        }
      // send empty features to our feature fector pub
      feature_vector_array_msg.vectors = features;
    }
    //if(pub.getNumSubscribers() > 0){
    //    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    //    pub.publish(msg);
    //}
    embedder_pub.publish(feature_vector_array_msg);
    embedder_pub_local.publish(feature_vector_array_msg);
    detector_pub.publish(detection_array_msg);
    detector_pub_local.publish(detection_array_msg);
    }

    /**
     * @brief callback for dynamic reconfigure
     * @param config  configure parameters
     * @param level   configure level
     */
    void detector_cfg_callback(recognition::FaceDetectionConfig& config, uint32_t level) {
        std::cout << "--- cfg_callback ---" << std::endl;
        std::string package_path = ros::package::getPath("recognition");
        std::cout << package_path << std::endl;
        detector_model_folder_path = package_path + config.face_detector_path; //the path to the face detector model file
        std::cout << detector_model_folder_path << std::endl;

        confidence_thresh = config.confidence_thresh; // the threshold for confidence of face detection
        roi_width = config.roi_width_; //the width of a face detection ROI in the world space [m]
        calc_roi_from_top = config.calc_roi_from_top; // if true, ROIs are calculated from the top positions of detected clusters
        head_offset_z_top = config.head_offset_z_top; // the distance between the top position of a human cluster and the center of the face [m]
        head_offset_z_centroid = config.head_offset_z_centroid; // the distance between the centroid of a human cluster and the center of the face [m]
        upscale_minsize = config.upscale_minsize; // the face detection ROI is upscaled so that its width get larger than #upscale_minsize
        visualization = config.visualization; // viz or not to viz
    }

    /**
     * @brief callback for dynamic reconfigure
     * @param config  configure parameters
     * @param level   configure level
     */
    void embedder_cfg_callback(recognition::FaceEmbeddingConfig& config, uint32_t level) {
        std::cout << "--- cfg_callback ---" << std::endl;
        std::string package_path = ros::package::getPath("recognition");
        std::cout << package_path << std::endl;
        embedder_model_folder_path = package_path + config.embedder_path; //the path to the face detector model file
        std::cout << model_folder_path << std::endl;
    }

    // BOTH JSON CALLBACKS HAVEN'T BEEN USED YET
    /**
     * @brief callback for dynamic reconfigure of detector params
     * @param level   configure level
     */
    void json_cfg_callback_detector(uint32_t level) {
        json model_config;
        std::string hard_coded_path = "/cfg/master.json";
        std::cout << "--- detector cfg_callback ---" << std::endl;
        std::string package_path = ros::package::getPath("recognition");
        std::string full_path = package_path + hard_coded_path;
        std::ifstream json_read(full_path);
        json_read >> model_config;

        detector_model_folder_path = model_config["face_detector_path"]; //the path to the face detector model file
        confidence_thresh = model_config["confidence_thresh"]; // the threshold for confidence of face detection
        roi_width = model_config["roi_width_"]; //the width of a face detection ROI in the world space [m]
        calc_roi_from_top = model_config["calc_roi_from_top"]; // if true, ROIs are calculated from the top positions of detected clusters
        head_offset_z_top = model_config["head_offset_z_top"]; // the distance between the top position of a human cluster and the center of the face [m]
        head_offset_z_centroid = model_config["head_offset_z_centroid"]; // the distance between the centroid of a human cluster and the center of the face [m]
        upscale_minsize = model_config["upscale_minsize"]; // the face detection ROI is upscaled so that its width get larger than #upscale_minsize
        visualization = model_config["visualization"]; // viz or not to viz
    }

    /**
     * @brief callback for dynamic reconfigure 0f embedder params
     * @param level   configure level
     */
    void json_cfg_callback_embedder(uint32_t level) {
        json model_config;
        std::string hard_coded_path = "/cfg/master.json";
        std::cout << "--- embedder cfg_callback ---" << std::endl;
        std::string package_path = ros::package::getPath("recognition");
        std::string full_path = package_path + hard_coded_path;
        std::ifstream json_read(full_path);
        json_read >> model_config;
        // embbeder only needs the parameter folder
        embedder_model_folder_path = model_config["embedder_path"]; //the path to the face detector model file
    }
};

int main(int argc, char** argv) {
  // read json parameters
  std::string sensor_name;
  std::string detections_topic;
  json master_config;
  std::string package_path = ros::package::getPath("recognition");
  std::string master_hard_coded_path = package_path + "/cfg/master.json";
  std::ifstream json_read(master_hard_coded_path);
  json_read >> master_config;
  sensor_name = master_config["sensor_name"]; //the path to the detector model file
  detections_topic = master_config["main_detections_topic"];


  std::cout << "--- face_detection_extraction_recognition_node ---" << std::endl;
  ros::init(argc, argv, "face_detection_extraction_recognition_node");
  //make sure this call is correct
  ros::NodeHandle nh;
  std::cout << "sensor_name: " << sensor_name << std::endl;
  std::cout << "detections_topic: " << detections_topic << std::endl;
  std::cout << "nodehandle init " << std::endl; 
  FaceDetectionNode node(nh, sensor_name, detections_topic);
  std::cout << "detection node init " << std::endl;
  ros::spin();
  return 0;
}
