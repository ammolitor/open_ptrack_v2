#include <open_ptrack/base_node/base_node.h>
#include<iostream>
#include<fstream>
using namespace std;


namespace open_ptrack
{
  namespace base_node
  {
    /**
     * @brief The TVMPoseNode
     */
    BaseNode::BaseNode(ros::NodeHandle& nh, std::string sensor_string, json zone):
          node_(nh), it(node_)
          {
            try
            {
              //json zone_json;
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
            
            image_pub = it.advertise(sensor_string + "/image", 1);

            // Camera callback for intrinsics matrix update
            camera_info_matrix = node_.subscribe(sensor_string + "/color/camera_info", 10, &BaseNode::camera_info_callback, this);

            //point_cloud_approximate_sync_ = node_.subscribe(sensor_string + "/depth_registered/points", 10, &BaseNode::callback, this);

            sensor_name = sensor_string;
            transform = transform.Identity();
            anti_transform = transform.inverse();
            zone_json = zone;
            rgb_image_ = pcl::PointCloud<pcl::RGB>::Ptr(new pcl::PointCloud<pcl::RGB>);
          }
  
    BaseNode::~BaseNode()
    {
      
    }

    void BaseNode::camera_info_callback(const CameraInfo::ConstPtr & msg){
      intrinsics_matrix << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
      cam_intrins_ << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1;
      _cx = msg->K[2];
      _cy = msg->K[5];
      _constant_x =  1.0f / msg->K[0];
      _constant_y = 1.0f /  msg->K[4];
      camera_info_available_flag = true;
    }

    void BaseNode::extract_RGB_from_pointcloud (PointCloudPtr input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud){
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

    void BaseNode::set_background (PointCloudPtr& background_cloud){
      // Voxel grid filtering:
      std::cout << "starting voxel grid filtering: " << std::endl;
      PointCloudT::Ptr cloud_filtered(new PointCloudT);
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

    PointCloudT::Ptr BaseNode::compute_background_cloud (PointCloudPtr& cloud){
      std::cout << "Background acquisition..." << std::flush;
      // Initialization for background subtraction
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
        //compute_background_cloud (max_background_frames, voxel_size, frame_id, rate, background_cloud);
        std::cout << "could not find background file, begining generation..." << std::endl;
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

    PointCloudPtr BaseNode::preprocess_cloud (PointCloudPtr& input_cloud){
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
      std::cout << "mean_luminance: " << mean_luminance << std::endl;

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
      std::cout << "preprocess_cloud downsampled size: " << cloud_downsampled->size() << std::endl;

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
      std::cout << "preprocess_cloud cloud_denoised size: " << cloud_denoised->size() << std::endl;

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
      std::cout << "preprocess_cloud cloud_filtered: " << cloud_filtered->size() << std::endl;

      return cloud_filtered;
    }

    PointCloudPtr BaseNode::rotate_cloud(PointCloudPtr cloud, Eigen::Affine3f transform ){
      std::cout << "rotating cloud." << std::endl;
      PointCloudPtr rotated_cloud (new PointCloud);
      pcl::transformPointCloud(*cloud, *rotated_cloud, transform);
      rotated_cloud->header.frame_id = cloud->header.frame_id;
      return rotated_cloud;
      }

    Eigen::VectorXf BaseNode::rotate_ground( Eigen::VectorXf ground_coeffs, Eigen::Affine3f transform){
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

    void BaseNode::compute_subclustering(PointCloudPtr no_ground_cloud, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){
      //PointCloudT::Ptr cloud(new PointCloudT);
      //*cloud = *cloud_;      
      std::cout << "creating people clusters from compute_subclustering" << std::endl;
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

    void BaseNode::create_foreground_cloud(const PointCloudT::ConstPtr& cloud_, std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters){
      int min_points = 30;
      int max_points = 5000;
      PointCloudT::Ptr cloud(new PointCloudT);
      *cloud = *cloud_;
      std::cout << "create_foreground_cloud cloud: " << cloud->size() << std::endl;
      // Point cloud pre-processing (downsampling and filtering):
      PointCloudPtr cloud_filtered(new PointCloud);
      cloud_filtered = preprocess_cloud(cloud);
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
        no_ground_cloud_rotated = rotate_cloud(no_ground_cloud_, transform_);
        ground_coeffs_new.resize(4);
        ground_coeffs_new = rotate_ground(ground_coeffs, transform_);
      }
      else
      {
        transform_ = transform_.Identity();
        anti_transform_ = transform_.inverse();
        no_ground_cloud_rotated = no_ground_cloud_;
        ground_coeffs_new = ground_coeffs;
      }
    }

    void BaseNode::set_ground_variables(const PointCloudT::ConstPtr& cloud_){
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

    void BaseNode::compute_head_subclustering(std::vector<open_ptrack::person_clustering::PersonCluster<PointT> >& clusters, std::vector<cv::Point2f> cluster_centroids2d, std::vector<cv::Point3f> cluster_centroids3d){

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
    bool BaseNode::check_detection_msg(opt_msgs::Detection detection_msg){
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


void BaseNode::draw_skelaton(cv::Mat cv_image_clone, std::vector<cv::Point3f> points){
  int num_parts = points.size();
  int gluon_to_rtpose_map[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
  cv::Point3f nose_head = points[0];
  cv::Point3f left_shoulder = points[5];
  cv::Point3f right_shoulder = points[6];
  cv::Point3f left_elbow = points[7];
  cv::Point3f right_elbow = points[8];
  cv::Point3f left_wrist = points[9];
  cv::Point3f right_wrist = points[10];
  cv::Point3f left_hip = points[11];
  cv::Point3f right_hip = points[12];
  cv::Point3f left_knee = points[13];
  cv::Point3f right_knee = points[14];
  cv::Point3f left_ankle = points[15];
  cv::Point3f right_ankle = points[16];
  for (size_t i = 0; i < num_parts; i++){
    int rtpose_part_index = gluon_to_rtpose_map[i];
    /* code */
    // IGNORE eyes/ears
    if (rtpose_part_index == -1){
      continue;
    } else {
      cv::Point3f point = points[i];
      float confidence = point.z;
      int cast_x = static_cast<int>(point.x);
      int cast_y = static_cast<int>(point.y);
      // debug this 
      cv::circle(cv_image_clone, cv::Point(cast_x, cast_y), 3, (0,0,0));
    }
  }
  // ******* NECK == joint location 1\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  // center of each shoulder == chest
  float x = (left_shoulder.x + right_shoulder.x) / 2;
  float y = (left_shoulder.y + right_shoulder.y) / 2;
  int cast_point_x = static_cast<int>(x);
  int cast_point_y = static_cast<int>(y);
  cv::circle(cv_image_clone, cv::Point(cast_point_x, cast_point_y), 3, (0,0,0));
  
  // ******** CHEST
  // weighted mean from rtpose
  // TODO if this looks ugly, we'll just use the neck
  float cx = (left_hip.x + right_hip.x) * 0.4 + (left_shoulder.x + right_shoulder.x) * 0.1;
  float cy = (left_hip.y + right_hip.y) * 0.4 + (left_shoulder.y + right_shoulder.y) * 0.1;
  int cast_cx = static_cast<int>(cx);
  int cast_cy = static_cast<int>(cy);
  cv::circle(cv_image_clone, cv::Point(cast_cx, cast_cy), 3, (0,0,0));

  //joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
  //                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  //                [5, 11], [6, 12], [11, 12],
  //                [11, 13], [12, 14], [13, 15], [14, 16]]
  //[0.        , 0.06666667, 0.13333333, 0.2       , 0.26666667,
  // 0.33333333, 0.4       , 0.46666667, 0.53333333, 0.6       ,
  // 0.66666667, 0.73333333, 0.8       , 0.86666667, 0.93333333,
  // 1.        ])
  //cv::line(legend_image, cv::Point(0,y_coord), cv::Point(100,y_coord),
  //       cv::Scalar(255*color(2), 255*color(1), 255*color(0)), 8)
  float color_map[16] = {0., 0.06666667, 0.13333333, 0.2, 0.26666667, 0.33333333, 0.4, 0.46666667, 0.53333333, 0.6, 0.66666667, 0.73333333, 0.8, 0.86666667, 0.93333333, 1.};
  cv::line(cv_image_clone, cv::Point(static_cast<int>(nose_head.x), static_cast<int>(nose_head.y)), cv::Point(static_cast<int>(x), static_cast<int>(y)), cv::Scalar(static_cast<int>(255.0*color_map[12]), static_cast<int>(255.0*color_map[12]), static_cast<int>(255.0*color_map[12])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_shoulder.x), static_cast<int>(left_shoulder.y)), cv::Point(static_cast<int>(right_shoulder.x), static_cast<int>(right_shoulder.y)), cv::Scalar(static_cast<int>(255.0*color_map[0]), static_cast<int>(255.0*color_map[0]), static_cast<int>(255.0*color_map[0])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_shoulder.x), static_cast<int>(left_shoulder.y)), cv::Point(static_cast<int>(left_elbow.x), static_cast<int>(left_elbow.y)), cv::Scalar(static_cast<int>(255.0*color_map[1]), static_cast<int>(255.0*color_map[1]), static_cast<int>(255.0*color_map[1])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_elbow.x), static_cast<int>(left_elbow.y)), cv::Point(static_cast<int>(left_wrist.x), static_cast<int>(left_wrist.y)), cv::Scalar(static_cast<int>(255.0*color_map[2]), static_cast<int>(255.0*color_map[2]), static_cast<int>(255.0*color_map[2])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_shoulder.x), static_cast<int>(right_shoulder.y)), cv::Point(static_cast<int>(right_elbow.x), static_cast<int>(right_elbow.y)), cv::Scalar(static_cast<int>(255.0*color_map[3]), static_cast<int>(255.0*color_map[3]), static_cast<int>(255.0*color_map[3])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_elbow.x), static_cast<int>(right_elbow.y)), cv::Point(static_cast<int>(right_wrist.x), static_cast<int>(right_wrist.y)), cv::Scalar(static_cast<int>(255.0*color_map[4]), static_cast<int>(255.0*color_map[4]), static_cast<int>(255.0*color_map[4])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_shoulder.x), static_cast<int>(left_shoulder.y)), cv::Point(static_cast<int>(left_hip.x), static_cast<int>(left_hip.y)), cv::Scalar(static_cast<int>(255.0*color_map[5]), static_cast<int>(255.0*color_map[5]), static_cast<int>(255.0*color_map[5])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_shoulder.x), static_cast<int>(right_shoulder.y)), cv::Point(static_cast<int>(right_hip.x), static_cast<int>(right_hip.y)), cv::Scalar(static_cast<int>(255.0*color_map[6]), static_cast<int>(255.0*color_map[6]), static_cast<int>(255.0*color_map[6])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_hip.x), static_cast<int>(left_hip.y)), cv::Point(static_cast<int>(right_hip.x), static_cast<int>(right_hip.y)), cv::Scalar(static_cast<int>(255.0*color_map[7]), static_cast<int>(255.0*color_map[7]), static_cast<int>(255.0*color_map[7])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_hip.x), static_cast<int>(left_hip.y)), cv::Point(static_cast<int>(left_knee.x), static_cast<int>(left_knee.y)), cv::Scalar(static_cast<int>(255.0*color_map[8]), static_cast<int>(255.0*color_map[8]), static_cast<int>(255.0*color_map[8])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_hip.x), static_cast<int>(right_hip.y)), cv::Point(static_cast<int>(right_knee.x), static_cast<int>(right_knee.y)), cv::Scalar(static_cast<int>(255.0*color_map[9]), static_cast<int>(255.0*color_map[9]), static_cast<int>(255.0*color_map[9])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(left_knee.x), static_cast<int>(left_knee.y)), cv::Point(static_cast<int>(left_ankle.x), static_cast<int>(left_ankle.y)), cv::Scalar(static_cast<int>(255.0*color_map[10]), static_cast<int>(255.0*color_map[10]), static_cast<int>(255.0*color_map[10])), 8);
  cv::line(cv_image_clone, cv::Point(static_cast<int>(right_knee.x), static_cast<int>(right_knee.y)), cv::Point(static_cast<int>(right_ankle.x), static_cast<int>(right_ankle.y)), cv::Scalar(static_cast<int>(255.0*color_map[11]), static_cast<int>(255.0*color_map[11]), static_cast<int>(255.0*color_map[11])), 8);
  
}


  } /* namespace base_node */
} /* namespace open_ptrack */

