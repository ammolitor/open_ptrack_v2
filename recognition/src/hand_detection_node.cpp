
#include <open_ptrack/hand_detection_node/hand_detection_node.h>

namespace open_ptrack
{
  namespace hand_detection_node
  {

    /**
     * @brief The DetectionNode constructor
     */
    HandDetectionNode::HandDetectionNode(ros::NodeHandle& nh, std::string sensor_string, json zone):
      //open_ptrack::base_node::BaseNode(nh, sensor_string, json_zone)
      open_ptrack::base_node::BaseNode(nh, sensor_string, zone)
      {
        // Publish Messages
        detections_pub = node_.advertise<opt_msgs::DetectionArray>("/hand_detector/detections", 3);
        image_pub = it.advertise(sensor_string + "/objects_detector/image", 1);
        tvm_object_detector.reset(new NoNMSYoloFromConfig("/cfg/hand_detector.json", "recognition"));
        point_cloud_approximate_sync_ = node_.subscribe(sensor_string + "/depth_registered/points", 10, &HandDetectionNode::callback, this);
      }


    void HandDetectionNode::basic_callback(const PointCloudT::ConstPtr& cloud_) {

      //Calculate direct and inverse transforms between camera and world frame:
      tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
                                  world_transform);
      tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
                                  world_inverse_transform);

      std::cout << "running algorithm callback" << std::endl;

      if (setbackground){
        std::cout << "background frame n: " << n_frame << std::endl;
        PointCloudT::Ptr newcloud(new PointCloudT);
        *newcloud = *cloud_;
        background_cloud = compute_background_cloud(newcloud);
        if (n_frame >= n_frames){
          set_background(background_cloud);
          setbackground = false;
        }
      } else { 
        // background is set
        // estimate ground plane and continue with detection process
        if (estimate_ground_plane) {
          set_ground_variables(cloud_);
          estimate_ground_plane = false;
        }

        // set message vars here
        open_ptrack::opt_utils::Conversions converter; 
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

        // set detection vars here
        int r, c;
        HungarianAlgorithm HungAlgo;
        std::vector<std::vector<double>> cost_matrix;
        cv::Point2f output_centroid;
        cv::Point3f output_centroid3d;
        std::vector<int> assignment;
        yolo_centroids2d.clear();
        yolo_centroids3d.clear();
        cluster_centroids2d.clear();
        cluster_centroids3d.clear();
        std::vector<int> valid;
        std::vector<open_ptrack::person_clustering::PersonCluster<PointT> > clusters;   

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
        // Fill rgb image:
        //rgb_image_->points.clear();                            // clear RGB pointcloud
        //extractRGBFromPointCloud(cloud_, rgb_image_);          // fill RGB pointcloud

        cv_image_clone = cv_image.clone();
        image_size = cv_image.size();
        height = static_cast<float>(image_size.height);
        width = static_cast<float>(image_size.width);

        std::cout << "running inference" << std::endl;
        // forward inference of object detector
        begin = ros::Time::now();
        output = tvm_object_detector->forward_full(cv_image);
        duration = ros::Time::now().toSec() - begin.toSec();
        std::cout << "inference detection time: " << duration << std::endl;
        std::cout << "inference detections: " << output->num << std::endl;
          
        // build cost matrix
        if (output->num >= 1) {
          float xmin;
          float ymin;
          float xmax;
          float ymax;
          int cast_xmin;
          int cast_ymin;
          int cast_xmax;
          int cast_ymax;
          float median_x;
          float median_y;
          float median_depth;
          float mx;
          float my;
          for (int i = 0; i < output->num; i++) {
            std::cout << "building inference centroid: " << i+1 << std::endl;
            // there's a rare case when all values == 0...
            xmin = output->boxes[i].xmin;
            ymin = output->boxes[i].ymin;
            xmax = output->boxes[i].xmax;
            ymax = output->boxes[i].ymax;

            if ((xmin == 0) && (ymin == 0) && (xmax == 0) && (ymax == 0)){
              std::cout << "xmin: " << xmin << std::endl;
              std::cout << "ymin: " << ymin << std::endl;
              std::cout << "xmax: " << xmax << std::endl;
              std::cout << "ymax: " << ymax << std::endl;
              std::cout << "all values zero. passing" << std::endl;
              continue;
            }

            cast_xmin = static_cast<int>(xmin);
            cast_ymin = static_cast<int>(ymin);
            cast_xmax = static_cast<int>(xmax);
            cast_ymax = static_cast<int>(ymax);
            // set the median of the bounding box
            median_x = xmin + ((xmax - xmin) / 2.0);
            median_y = ymin + ((ymax - ymin) / 2.0);
            
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

            // get x, y, z points
            mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
            my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
            median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

            std::cout << "yolo centroid - x:" << mx << ", y: " << my << ", z: " << median_depth << std::endl;
            if(std::isfinite(median_depth) && std::isfinite(mx) && std::isfinite(my)){
              output_centroid = cv::Point2f(mx, my); // or median_x, median_y
              output_centroid3d = cv::Point3f(mx, my, median_depth);
              yolo_centroids2d.push_back(output_centroid);
              yolo_centroids3d.push_back(output_centroid3d);
              std::cout << "centroid added" << std::endl; 
              valid.push_back(i);
            }
          }

          std::cout << "yolo centroids size: " << yolo_centroids2d.size() << std::endl;

          if (yolo_centroids2d.size() > 0){
            // filter the background and create a filtered cloud
            std::cout << "creating foreground cloud" << std::endl;
            create_foreground_cloud(cloud_, clusters);

            // compute_head_subclustering(clusters, cluster_centroids, cluster_centroids3d);
            std::cout << "cluster_centroids2d size: " << cluster_centroids2d.size() << std::endl;
            // use 3 dimensions
            if (cluster_centroids3d.size() > 0) {
              // Initialize cost matrix for the hungarian algorithm
              std::cout << "initialize cost matrix for the hungarian algorithm" << std::endl;
              for (int r = 0; r < cluster_centroids3d.size (); r++) {
                std::vector<double> row;
                for (int c = 0; c < yolo_centroids3d.size (); c++) {
                  float dist;
                  dist = cv::norm(cv::Mat(yolo_centroids3d[c]), cv::Mat (cluster_centroids3d[r]));
                  row.push_back(dist);
                }
                cost_matrix.push_back(row);
              }
              
              // Solve the Hungarian problem to match the distance of the roi centroid
              // to that of the bounding box
              std::cout << "solving Hungarian problem" << std::endl;
              HungAlgo.Solve(cost_matrix, assignment);
              // rows == pcl centroids index
              // values ==  yolo index
              // assignment size == cluster_centroids2d size:
              // value at each == yolo
              std::cout << "assignment shape: " <<  assignment.size() << std::endl;
              int negs = 0;
              int poss = 0;
              for (int i = 0; i < assignment.size(); i++){
                if (assignment[i] == -1){
                  negs+=1;
                } else {
                  poss+=1;
                  std::cout << "assignment i: " << i << " value: " << assignment[i] << std::endl;
                }
              }
              std::cout << "assignment positives: " <<  poss << std::endl;
              std::cout << "assignment negatives: " <<  negs << std::endl;

              for (int x = 0; x < assignment.size(); x++) {
                if (assignment[x] == -1){
                  continue;
                }
                else
                {
                  // cluster_centroids3d 36 maps to yolo_centroids3d 0.
                  // yolo_centroids3d 0 maps to valid 2
                  // valid 2 == output->boxes[2]
                  // 
                  // output->boxes 0 and 1 are both bad detections...
                  int i = valid[assignment[x]];
                  std::cout << "cluster: " << x << " to yolo number: " << i << std::endl;
                  open_ptrack::person_clustering::PersonCluster<PointT> person_cluster = clusters[x];
                  float xmin = output->boxes[i].xmin;
                  float ymin = output->boxes[i].ymin;
                  float xmax = output->boxes[i].xmax;
                  float ymax = output->boxes[i].ymax;
                  float score = output->boxes[i].score;
                  std::cout << "xmin: " << xmin << std::endl;
                  std::cout << "ymin: " << ymin << std::endl;
                  std::cout << "xmax: " << xmax << std::endl;
                  std::cout << "ymax: " << ymax << std::endl;
                  std::cout << "score: " << score << std::endl;

                  std::cout << "yolo xmin check " << xmin << std::endl;
                  // make sure nothing == 0 or MAX so no display errors happen
                  if (xmin <= 1.0f){
                    xmin = 1.0f;
                  }   
                  if (ymin <= 1.0f){
                    ymin = 1.0f;
                  }
                  if (xmax >= width){
                    xmax = height-1.0f;
                  }
                  if (ymax >= height){
                    ymax = height-1.0f;
                  }                     
                  std::cout << "cleaned xmin: " << xmin << std::endl;
                  std::cout << "cleaned ymin: " << ymin << std::endl;
                  std::cout << "cleaned xmax: " << xmax << std::endl;
                  std::cout << "cleaned ymax: " << ymax << std::endl;                  

                  float label = static_cast<float>(output->boxes[i].id);
                  std::string object_name = "hands";
                  std::cout << "object_name: " << object_name << std::endl;
                  // get the coordinate information
                  int cast_xmin = static_cast<int>(xmin);
                  int cast_ymin = static_cast<int>(ymin);
                  int cast_xmax = static_cast<int>(xmax);
                  int cast_ymax = static_cast<int>(ymax);

                  std::cout << "cast_xmin: " << cast_xmin << std::endl;
                  std::cout << "cast_ymin: " << cast_ymin << std::endl;
                  std::cout << "cast_xmax: " << cast_xmax << std::endl;
                  std::cout << "cast_ymax: " << cast_ymax << std::endl; 

                  //std::vector<cv::Point3f> points = output->boxes[i].points;
                  //int num_parts = points.size();
                  //std::cout << "num_parts: " << num_parts << std::endl;

                  // set the median of the bounding box
                  float median_x = xmin + ((xmax - xmin) / 2.0);
                  float median_y = ymin + ((ymax - ymin) / 2.0);
                
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
                  std::cout << "cleaned median_x: " << median_x << std::endl;
                  std::cout << "cleaned median_y: " << median_y << std::endl;
                  // set the new coordinates of the image so that the boxes are set
                  int new_x = static_cast<int>(median_x - (median_factor * (median_x - xmin)));
                  int new_y = static_cast<int>(median_y - (median_factor * (median_y - ymin)));
                  int new_width = static_cast<int>(2 * (median_factor * (median_x - xmin)));
                  int new_height = static_cast<int>(2 * (median_factor * (median_y - ymin)));
                  
                  std::cout << "new_x: " << new_x << std::endl;
                  std::cout << "new_y: " << new_y << std::endl;
                  std::cout << "new_width: " << new_width << std::endl;
                  std::cout << "new_height: " << new_height << std::endl;

                  // get x, y, z points
                  float mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
                  float my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
                  float median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

                  std::cout << "mx: " << mx << std::endl;
                  std::cout << "my: " << my << std::endl;
                  std::cout << "median_depth: " << median_depth << std::endl;

                  // Create detection message: -- literally tatken ground_based_people_detector_node
                  opt_msgs::Detection detection_msg;
                  converter.Vector3fToVector3(anti_transform * person_cluster.getMin(), detection_msg.box_3D.p1);
                  converter.Vector3fToVector3(anti_transform * person_cluster.getMax(), detection_msg.box_3D.p2);
                      
                  float head_centroid_compensation = 0.05;
                  // theoretical person centroid:
                  Eigen::Vector3f centroid3d = anti_transform * person_cluster.getTCenter();
                  Eigen::Vector3f centroid2d = converter.world2cam(centroid3d, intrinsics_matrix);
                  // theoretical person top point:
                  Eigen::Vector3f top3d = anti_transform * person_cluster.getTTop();
                  Eigen::Vector3f top2d = converter.world2cam(top3d, intrinsics_matrix);
                  // theoretical person bottom point:
                  Eigen::Vector3f bottom3d = anti_transform * person_cluster.getTBottom();
                  Eigen::Vector3f bottom2d = converter.world2cam(bottom3d, intrinsics_matrix);
                  float enlarge_factor = 1.1;
                  float pixel_xc = centroid2d(0);
                  float pixel_yc = centroid2d(1);
                  float pixel_height = (bottom2d(1) - top2d(1)) * enlarge_factor;
                  float pixel_width = pixel_height / 2;
                  detection_msg.box_2D.x = int(centroid2d(0) - pixel_width/2.0);
                  detection_msg.box_2D.y = int(centroid2d(1) - pixel_height/2.0);
                  detection_msg.box_2D.width = int(pixel_width);
                  detection_msg.box_2D.height = int(pixel_height);
                  detection_msg.height = person_cluster.getHeight();
                  detection_msg.confidence = score;//use yolo score, not pcl //person_cluster.getPersonConfidence();
                  detection_msg.distance = person_cluster.getDistance();
                  converter.Vector3fToVector3((1+head_centroid_compensation/centroid3d.norm())*centroid3d, detection_msg.centroid);
                  converter.Vector3fToVector3((1+head_centroid_compensation/top3d.norm())*top3d, detection_msg.top);
                  converter.Vector3fToVector3((1+head_centroid_compensation/bottom3d.norm())*bottom3d, detection_msg.bottom);


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
                      std::cout << "zone_string: " << zone_string << std::endl;
                      // type must be number but is null...
                      //https://github.com/nlohmann/json/issues/1593

                      // translate between world and frame
                      world_x_min = zone_json[zone_string]["min"]["world"]["x"];
                      world_y_min = zone_json[zone_string]["min"]["world"]["y"];
                      world_z_min = zone_json[zone_string]["min"]["world"]["z"];
                      world_x_max = zone_json[zone_string]["max"]["world"]["x"];
                      world_y_max = zone_json[zone_string]["max"]["world"]["y"];
                      world_z_max = zone_json[zone_string]["max"]["world"]["z"];

                      std::cout << "world_x_min: " << world_x_min << std::endl;
                      std::cout << "world_y_min: " << world_y_min << std::endl;
                      std::cout << "world_z_min: " << world_z_min << std::endl;
                      std::cout << "world_x_max: " << world_x_max << std::endl;
                      std::cout << "world_y_max: " << world_y_max << std::endl;
                      std::cout << "world_z_max: " << world_z_max << std::endl;

                      Eigen::Vector3d min_vec;
                      Eigen::Vector3d max_vec;
                      tf::Vector3 min_point(world_x_min, world_y_min, world_z_min);
                      tf::Vector3 max_point(world_x_max, world_y_max, world_z_max);
                      
                      min_point = world_transform(min_point);
                      max_point = world_transform(max_point);

                      x_min = min_point.getX();
                      y_min = min_point.getY();
                      z_min = min_point.getZ();
                      x_max = min_point.getX();
                      y_max = min_point.getY();
                      z_max = min_point.getZ();

                      std::cout << "x_min: " << x_min << std::endl;
                      std::cout << "y_min: " << y_min << std::endl;
                      std::cout << "z_min: " << z_min << std::endl;
                      std::cout << "x_max: " << x_max << std::endl;
                      std::cout << "y_max: " << y_max << std::endl;
                      std::cout << "z_max: " << z_max << std::endl;
                      std::cout << "mx: " << mx << std::endl;
                      std::cout << "my: " << my << std::endl;
                      std::cout << "median_depth: " << median_depth << std::endl;

                      inside_area_cube = (mx <= x_max && mx >= x_min) && (my <= y_max && my >= y_min) && (median_depth <= z_max && median_depth >= z_min);
                      std::cout << "inside_cube: " << inside_area_cube << std::endl;
                      if (inside_area_cube) {
                        break;
                      }
                    }

                    if (inside_area_cube) {
                      detection_msg.zone_id = zone_id;
                      std::cout << "DEBUG -- INSIDE ZONE: " << zone_id << std::endl;
                    } else {
                      detection_msg.zone_id = 1000;
                    } 
                  }
                  // final check here 
                  // only add to message if no nans exist
                  if (check_detection_msg(detection_msg)){
                    std::cout << "valid detection!" << std::endl;
                    detection_msg.object_name=object_name;            
                    detection_array_msg->detections.push_back(detection_msg);
                
                  cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
                  cv::putText(cv_image_clone, object_name, cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
                }
              }
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
    }


    void HandDetectionNode::callback(const PointCloudT::ConstPtr& cloud_) {

      //Calculate direct and inverse transforms between camera and world frame:
      tf_listener.lookupTransform("/world", sensor_name, ros::Time(0),
                                  world_transform);
      tf_listener.lookupTransform(sensor_name, "/world", ros::Time(0),
                                  world_inverse_transform);

      std::cout << "running algorithm callback" << std::endl;


      // set message vars here
      open_ptrack::opt_utils::Conversions converter; 
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
      // Fill rgb image:
      //rgb_image_->points.clear();                            // clear RGB pointcloud
      //extractRGBFromPointCloud(cloud_, rgb_image_);          // fill RGB pointcloud

      cv_image_clone = cv_image.clone();
      image_size = cv_image.size();
      height = static_cast<float>(image_size.height);
      width = static_cast<float>(image_size.width);

      std::cout << "running inference" << std::endl;
      // forward inference of object detector
      begin = ros::Time::now();
      output = tvm_object_detector->forward_full(cv_image);
      duration = ros::Time::now().toSec() - begin.toSec();
      std::cout << "inference detection time: " << duration << std::endl;
      std::cout << "inference detections: " << output->num << std::endl;
        
      // build cost matrix
      if (output->num >= 1) {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        int cast_xmin;
        int cast_ymin;
        int cast_xmax;
        int cast_ymax;
        float median_x;
        float median_y;
        float median_depth;
        float mx;
        float my;
        float score;
        for (int i = 0; i < output->num; i++) {
          // there's a rare case when all values == 0...
          xmin = output->boxes[i].xmin;
          ymin = output->boxes[i].ymin;
          xmax = output->boxes[i].xmax;
          ymax = output->boxes[i].ymax;
          score = output->boxes[i].score;

          if ((xmin == 0) && (ymin == 0) && (xmax == 0) && (ymax == 0)){
            std::cout << "xmin: " << xmin << std::endl;
            std::cout << "ymin: " << ymin << std::endl;
            std::cout << "xmax: " << xmax << std::endl;
            std::cout << "ymax: " << ymax << std::endl;
            std::cout << "all values zero. passing" << std::endl;
            continue;
          }

          cast_xmin = static_cast<int>(xmin);
          cast_ymin = static_cast<int>(ymin);
          cast_xmax = static_cast<int>(xmax);
          cast_ymax = static_cast<int>(ymax);
          // set the median of the bounding box
          median_x = xmin + ((xmax - xmin) / 2.0);
          median_y = ymin + ((ymax - ymin) / 2.0);
          
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

          // get x, y, z points
          mx = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).x;
          my = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).y;
          median_depth = cloud_->at(static_cast<int>(median_x), static_cast<int>(median_y)).z;

          // set the new coordinates of the image so that the boxes are set
          int new_x = static_cast<int>(median_x - (median_factor * (median_x - xmin)));
          int new_y = static_cast<int>(median_y - (median_factor * (median_y - ymin)));
          int new_width = static_cast<int>(2 * (median_factor * (median_x - xmin)));
          int new_height = static_cast<int>(2 * (median_factor * (median_y - ymin)));

          float cpoint1x = cloud_->at(static_cast<int>(new_x), static_cast<int>(new_y)).x;
          float cpoint1y = cloud_->at(static_cast<int>(new_x), static_cast<int>(new_y)).y;
          float cpoint1z = cloud_->at(static_cast<int>(new_x), static_cast<int>(new_y)).z;
          float cpoint2x = cloud_->at(static_cast<int>(new_x+new_width), static_cast<int>(new_y+new_height)).x;
          float cpoint2y = cloud_->at(static_cast<int>(new_x+new_width), static_cast<int>(new_y+new_height)).y;
          float cpoint2z = cloud_->at(static_cast<int>(new_x+new_width), static_cast<int>(new_y+new_height)).z;

          // Create detection message: -- literally tatken ground_based_people_detector_node
          opt_msgs::Detection detection_msg;
          geometry_msgs::Vector3 p1;
          p1.x = mx;
          p1.y = my;
          p1.z = median_depth;
          detection_msg.box_3D.p1 = p1;
          geometry_msgs::Vector3 p2;
          p2.x = mx;
          p2.y = my;
          p2.z = median_depth;
          detection_msg.box_3D.p2 = p2;
          geometry_msgs::Vector3 centroid;
          centroid.x = mx;
          centroid.y = my;
          centroid.z = median_depth;
          detection_msg.centroid = centroid;
          geometry_msgs::Vector3 top;
          top.x = mx;
          top.y = my;
          top.z = median_depth;
          detection_msg.top = top;
          detection_msg.box_2D.x = mx;//int(centroid2d(0) - pixel_width/2.0);
          detection_msg.box_2D.y = my;//int(centroid2d(1) - pixel_height/2.0);
          detection_msg.box_2D.width = cpoint2x - cpoint1x;//int(pixel_width);
          detection_msg.box_2D.height = cpoint2y - cpoint1y;;//int(pixel_height);
          detection_msg.height = 0.0;//person_cluster.getHeight();
          detection_msg.confidence = score;//person_cluster.getPersonConfidence();
          detection_msg.distance = median_depth;//person_cluster.getDistance();

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
            for (zone_id = 0; zone_id < n_zones; zone_id++){
              // need a world view here bc each detection was transformed
              // this will work for a singular cam, but would mean each cam would have to tune
              // to the specific area; which I think would be fine. // but will need
              // to test to be sure
              // a given detection can be in only one place at one time, thus it can't be in
              // multiple zones
              zone_string = std::to_string(zone_id);
              std::cout << "zone_string: " << zone_string << std::endl;
              // type must be number but is null...
              //https://github.com/nlohmann/json/issues/1593

              // translate between world and frame
              world_x_min = zone_json[zone_string]["min"]["world"]["x"];
              world_y_min = zone_json[zone_string]["min"]["world"]["y"];
              world_z_min = zone_json[zone_string]["min"]["world"]["z"];
              world_x_max = zone_json[zone_string]["max"]["world"]["x"];
              world_y_max = zone_json[zone_string]["max"]["world"]["y"];
              world_z_max = zone_json[zone_string]["max"]["world"]["z"];

              std::cout << "world_x_min: " << world_x_min << std::endl;
              std::cout << "world_y_min: " << world_y_min << std::endl;
              std::cout << "world_z_min: " << world_z_min << std::endl;
              std::cout << "world_x_max: " << world_x_max << std::endl;
              std::cout << "world_y_max: " << world_y_max << std::endl;
              std::cout << "world_z_max: " << world_z_max << std::endl;

              Eigen::Vector3d min_vec;
              Eigen::Vector3d max_vec;
              tf::Vector3 min_point(world_x_min, world_y_min, world_z_min);
              tf::Vector3 max_point(world_x_max, world_y_max, world_z_max);
              
              min_point = world_transform(min_point);
              max_point = world_transform(max_point);

              x_min = min_point.getX();
              y_min = min_point.getY();
              z_min = min_point.getZ();
              x_max = min_point.getX();
              y_max = min_point.getY();
              z_max = min_point.getZ();

              std::cout << "x_min: " << x_min << std::endl;
              std::cout << "y_min: " << y_min << std::endl;
              std::cout << "z_min: " << z_min << std::endl;
              std::cout << "x_max: " << x_max << std::endl;
              std::cout << "y_max: " << y_max << std::endl;
              std::cout << "z_max: " << z_max << std::endl;
              std::cout << "mx: " << mx << std::endl;
              std::cout << "my: " << my << std::endl;
              std::cout << "median_depth: " << median_depth << std::endl;

              inside_area_cube = (mx <= x_max && mx >= x_min) && (my <= y_max && my >= y_min) && (median_depth <= z_max && median_depth >= z_min);
              std::cout << "inside_cube: " << inside_area_cube << std::endl;
              if (inside_area_cube) {
                break;
              }
            }

            if (inside_area_cube) {
              detection_msg.zone_id = zone_id;
              std::cout << "DEBUG -- INSIDE ZONE: " << zone_id << std::endl;
            } else {
              detection_msg.zone_id = 1000;
            } 
          }
          // final check here 
          // only add to message if no nans exist
          if (check_detection_msg(detection_msg)){
            std::cout << "valid detection!" << std::endl;
            detection_msg.object_name="hands";            
            detection_array_msg->detections.push_back(detection_msg);
        
          cv::rectangle(cv_image_clone, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar( 255, 0, 255 ), 10);
          cv::putText(cv_image_clone, "hands", cv::Point(xmin + 10, ymin + 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(200,200,250), 1, CV_AA);
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
  } //hand_detection_node
} //open_ptrack