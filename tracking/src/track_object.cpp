/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2011-2012, Matteo Munaro [matteo.munaro@dei.unipd.it], Filippo Basso [filippo.basso@dei.unipd.it]
 * Copyright (c) 2013-, Open Perception, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Matteo Munaro [matteo.munaro@dei.unipd.it], Filippo Basso [filippo.basso@dei.unipd.it]
 *
 */

#include <ros/ros.h>

#include <open_ptrack/tracking/track_object.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace open_ptrack
{
namespace tracking
{

TrackObject::TrackObject(
    int id,
    std::string frame_id,
    double position_variance,
    double acceleration_variance,
    double period,
    bool velocity_in_motion_term) :
  id_(id),
  frame_id_(frame_id),
  period_(period),
  velocity_in_motion_term_(velocity_in_motion_term)
{
  color_ = Eigen::Vector3f(
        float(rand() % 256) / 255,
        float(rand() % 256) / 255,
        float(rand() % 256) / 255);

  MAX_SIZE = 90; //XXX create a parameter!!!
  if (velocity_in_motion_term_)
  {
    filter_ = new open_ptrack::tracking::KalmanFilter(period, position_variance, acceleration_variance, 4);
    tmp_filter_ = new open_ptrack::tracking::KalmanFilter(period, position_variance, acceleration_variance, 4);
    mahalanobis_map4d_.resize(MAX_SIZE, MahalanobisParameters4d());
  }
  else
  {
    filter_ = new open_ptrack::tracking::KalmanFilter(period, position_variance, acceleration_variance, 2);
    tmp_filter_ = new open_ptrack::tracking::KalmanFilter(period, position_variance, acceleration_variance, 2);
    mahalanobis_map2d_.resize(MAX_SIZE, MahalanobisParameters2d());
  }

}

TrackObject::~TrackObject()
{
  delete filter_;
  delete tmp_filter_;
}

void
TrackObject::init(const TrackObject& old_track)
{
  double x, y;
  old_track.filter_->getState(x, y);

  filter_->init(x, y, 10, old_track.velocity_in_motion_term_);

  *tmp_filter_ = *filter_;
  visibility_ = old_track.visibility_;

  ROS_INFO("%d -> %d", old_track.id_, id_);

  z_ = old_track.z_;
  height_ = old_track.height_;
  distance_ = old_track.distance_;
  if(old_track.object_name_!="default")
    object_name_=old_track.object_name_;
  age_ = old_track.age_;

  detection_source_ = old_track.detection_source_;
  velocity_in_motion_term_ = old_track.velocity_in_motion_term_;
  validated_ = validated_ || old_track.validated_;
  low_confidence_consecutive_frames_ = old_track.low_confidence_consecutive_frames_;

  first_time_detected_ = old_track.first_time_detected_;
  last_time_detected_ = old_track.last_time_detected_;
  last_time_detected_with_high_confidence_ = old_track.last_time_detected_with_high_confidence_;
  last_time_predicted_ = old_track.last_time_predicted_;
  last_time_predicted_index_ = old_track.last_time_predicted_index_;

  data_association_score_ = old_track.data_association_score_;
}



void
TrackObject::init(double x, double y, double z, double height, double distance,std::string object_name,
                   open_ptrack::detection::DetectionSource* detection_source)
{
  //Init Kalman filter
  filter_->init(x, y, distance, velocity_in_motion_term_);
  z_ = z;
  height_ = height;
  distance_ = distance;
  if(object_name!="default")
    object_name_=object_name;
  status_ = NEW;
  visibility_ = VISIBLE;
  validated_ = false;
  updates_with_enough_confidence_ = low_confidence_consecutive_frames_ = 0;
  detection_source_ = detection_source;
  first_time_detected_ = detection_source->getTime();
  last_time_predicted_ = last_time_detected_ = last_time_detected_with_high_confidence_ = detection_source->getTime();
  last_time_predicted_index_ = 0;
  age_ = 0.0;
}

void
TrackObject::update(
    double x,
    double y,
    double z,
    double height,
    double distance,
    std::string object_name,
    double data_assocation_score,
    double confidence,
    double min_confidence,
    double min_confidence_detections,
    open_ptrack::detection::DetectionSource* detection_source,
    bool first_update)
{
  //Update Kalman filter
  int difference;
  double vx, vy;
  if (velocity_in_motion_term_)
  {
    ros::Duration d(1.0);
    ros::Duration d2(2.0);

    double t = std::max(first_time_detected_.toSec(), (detection_source->getTime() - d).toSec());
    t = std::min(t, last_time_detected_.toSec());
    t = std::max(t, (detection_source->getTime() - d2).toSec());
    double dt = t - last_time_predicted_.toSec();

    difference = int(round(dt / period_));
    int vIndex = (MAX_SIZE + last_time_predicted_index_ + difference) % MAX_SIZE;

    if(difference != 0)
    {
      vx = - (x - mahalanobis_map4d_[vIndex].x) / dt;
      vy = - (y - mahalanobis_map4d_[vIndex].y) / dt;
    }
    else
    {
      vx = mahalanobis_map4d_[vIndex].x;
      vy = mahalanobis_map4d_[vIndex].x;
    }
  }

  // Update Kalman filter from the last time the track was visible:
  int framesLost = int(round((detection_source->getTime() - last_time_detected_).toSec() / period_)) - 1;

  for(int i = 0; i < framesLost; i++)
  {
    filter_->predict();
    filter_->update();
  }

  filter_->predict();
  if (velocity_in_motion_term_)
  {
    filter_->update(x, y, vx, vy, distance);
  }
  else
  {
    filter_->update(x, y, distance);
  }

  *tmp_filter_ = *filter_;
  difference = int(round((detection_source->getTime() - last_time_predicted_).toSec() / period_));
  last_time_predicted_index_ = (MAX_SIZE + last_time_predicted_index_ + difference) % MAX_SIZE;
  last_time_predicted_ = last_time_detected_ = detection_source->getTime();
  if (velocity_in_motion_term_)
    filter_->getMahalanobisParameters(mahalanobis_map4d_[last_time_predicted_index_]);
  else
    filter_->getMahalanobisParameters(mahalanobis_map2d_[last_time_predicted_index_]);

  // Update z_ and height_ with a weighted combination of current and new values:
  z_ = z_ * 0.9 + z * 0.1;
  height_ = height_ * 0.9 + height * 0.1;
  distance_ = distance;
  if(object_name!="default")
    object_name_=object_name;
  if(confidence > min_confidence)
  {
    updates_with_enough_confidence_++;
    last_time_detected_with_high_confidence_ = last_time_detected_;
  }

  //      if (((confidence - 0.5) < min_confidence_detections) && ((last_detector_confidence_ - 0.5) < min_confidence_detections))
  if ((confidence < (min_confidence + min_confidence_detections)/2) && (last_detector_confidence_ < (min_confidence + min_confidence_detections)/2))
  {
    low_confidence_consecutive_frames_++;
  }
  else
  {
    low_confidence_consecutive_frames_ = 0;
  }
  last_detector_confidence_ = confidence;

  data_association_score_ = data_assocation_score;

  // Compute track age:
  age_ = (detection_source->getTime() - first_time_detected_).toSec();

  detection_source_ = detection_source;
}

void
TrackObject::validate()
{
  validated_ = true;
}

bool
TrackObject::isValidated()
{
  return validated_;
}

int
TrackObject::getId()
{
  return id_;
}

void
TrackObject::setStatus(TrackObject::Status s)
{
  status_ = s;
}

TrackObject::Status
TrackObject::getStatus()
{
  return status_;
}

void
TrackObject::setVisibility(TrackObject::Visibility v)
{
  visibility_ = v;
}

TrackObject::Visibility
TrackObject::getVisibility()
{
  return visibility_;
}

float
TrackObject::getSecFromFirstDetection(ros::Time current_time)
{
  return (current_time - first_time_detected_).toSec();
}

float
TrackObject::getSecFromLastDetection(ros::Time current_time)
{
  return (current_time - last_time_detected_).toSec();
}

float
TrackObject::getSecFromLastHighConfidenceDetection(ros::Time current_time)
{
  return (current_time - last_time_detected_with_high_confidence_).toSec();
}

float
TrackObject::getLowConfidenceConsecutiveFrames()
{
  return low_confidence_consecutive_frames_;
}

int
TrackObject::getUpdatesWithEnoughConfidence()
{
  return updates_with_enough_confidence_;
}

double
TrackObject::getMahalanobisDistance(double x, double y, const ros::Time& when)
{
  int difference = int(round((when - last_time_predicted_).toSec() / period_));
  //      std::cout << "time difference from last detection: " << difference << std::endl;
  int index;
  if(difference <= 0)
  {
    index = (MAX_SIZE + last_time_predicted_index_ + difference) % MAX_SIZE;
  }
  else
  {
    for(int i = 0; i < difference; i++)
    {
      tmp_filter_->predict();
      last_time_predicted_index_ = (last_time_predicted_index_ + 1) % MAX_SIZE;
      if (velocity_in_motion_term_)
        tmp_filter_->getMahalanobisParameters(mahalanobis_map4d_[last_time_predicted_index_]);
      else
        tmp_filter_->getMahalanobisParameters(mahalanobis_map2d_[last_time_predicted_index_]);
      tmp_filter_->update();
    }
    last_time_predicted_ = when;
    index = last_time_predicted_index_;
  }

  if (velocity_in_motion_term_)
  {
    ros::Duration d(1.0);
    ros::Duration d2(2.0);

    double t = std::max(first_time_detected_.toSec(), (when - d).toSec());
    t = std::min(t, last_time_detected_.toSec());
    t = std::max(t, (last_time_predicted_ - d2).toSec());
    double dt = t - last_time_predicted_.toSec();

    difference = int(round(dt / period_));
    int vIndex = (MAX_SIZE + last_time_predicted_index_ + difference) % MAX_SIZE;

    //        std::cout << "dt: " << dt << std::endl;
    //        std::cout << "vIndex: " << vIndex << std::endl;

    double vx, vy;
    if(difference != 0)
    {
      vx = - (x - mahalanobis_map4d_[vIndex].x) / dt;
      vy = - (y - mahalanobis_map4d_[vIndex].y) / dt;
    }
    else
    {
      vx = mahalanobis_map4d_[vIndex].x;
      vy = mahalanobis_map4d_[vIndex].y;
    }

    //        std::cout << "vx: " << vx << ", vy: " << vy<< std::endl;

    return open_ptrack::tracking::KalmanFilter::performMahalanobisDistance(x, y, vx, vy, mahalanobis_map4d_[index]);
  }
  else
  {
    return open_ptrack::tracking::KalmanFilter::performMahalanobisDistance(x, y, mahalanobis_map2d_[index]);
  }

}

void
TrackObject::draw(bool vertical)
{
  cv::Scalar color(int(255.0 * color_(0)), int(255.0 * color_(1)), int(255.0 * color_(2)));

  double _x2, _y2;
  tmp_filter_->getState(_x2, _y2);
  Eigen::Vector3d centroid2(_x2, _y2, z_);
  centroid2 = detection_source_->transformToCam(centroid2);

  if(visibility_ == TrackObject::NOT_VISIBLE)
    return;

  double _x, _y;
  filter_->getState(_x, _y);

  cv::Scalar darkBlue(130,0,0);
  cv::Scalar white(255,255,255);
  Eigen::Vector3d centroid(_x, _y, z_ );
  Eigen::Vector3d top(_x, _y, z_ + (height_/2));
  Eigen::Vector3d bottom(_x, _y, z_ - (height_/2));

  std::vector<Eigen::Vector3d> points;
  double delta = height_ / 5.0;
  points.push_back(Eigen::Vector3d(_x - delta, _y - delta, z_ - (height_/2)));
  points.push_back(Eigen::Vector3d(_x + delta, _y - delta, z_ - (height_/2)));
  points.push_back(Eigen::Vector3d(_x + delta, _y + delta, z_ - (height_/2)));
  points.push_back(Eigen::Vector3d(_x - delta, _y + delta, z_ - (height_/2)));
  points.push_back(Eigen::Vector3d(_x - delta, _y - delta, z_ + (height_/2)));
  points.push_back(Eigen::Vector3d(_x + delta, _y - delta, z_ + (height_/2)));
  points.push_back(Eigen::Vector3d(_x + delta, _y + delta, z_ + (height_/2)));
  points.push_back(Eigen::Vector3d(_x - delta, _y + delta, z_ + (height_/2)));

  //TODO loop for each detection source

  centroid = detection_source_->transformToCam(centroid);
  cv::circle(detection_source_->getImage(), cv::Point(centroid(0), centroid(1)), 5, color, 1);
  top = detection_source_->transformToCam(top);
  bottom = detection_source_->transformToCam(bottom);
  for(std::vector<Eigen::Vector3d>::iterator it = points.begin(); it != points.end(); it++)
    *it = detection_source_->transformToCam(*it);

  // Draw a paralellepiped around the person:
  for(size_t i = 0; i < 4; i++)
  {
    cv::line(detection_source_->getImage(), cv::Point(points[i](0), points[i](1)),
                                                      cv::Point(points[(i + 1) % 4](0), points[(i + 1) % 4](1)), color,
                                                      visibility_ == VISIBLE ? 2 : 1, CV_AA);
             cv::line(detection_source_->getImage(), cv::Point(points[i + 4](0), points[i + 4](1)),
        cv::Point(points[(i + 1) % 4 + 4](0), points[(i + 1) % 4 + 4](1)), color,
        visibility_ == VISIBLE ? 2 : 1, CV_AA);
    cv::line(detection_source_->getImage(), cv::Point(points[i](0), points[i](1)),
                                                      cv::Point(points[i + 4](0), points[i + 4](1)), color,
                                                      visibility_ == VISIBLE ? 2 : 1, CV_AA);
  }

  std::stringstream ss;
  float distance_to_display = float(int(distance_*100))/100;
  ss << id_ << ": " << distance_to_display;

  float id_half_number_of_digits = (float)(ss.str().size())/2;

  // Draw white id_ on a blue background:
  if (not vertical)
  {
    cv::rectangle(detection_source_->getImage(), cv::Point(top(0)-8*id_half_number_of_digits, top(1)-12),
                  cv::Point(top(0)+12*id_half_number_of_digits, top(1) +2), darkBlue, CV_FILLED,
                  visibility_ == VISIBLE ? 8 : 1);//, 0);
    cv::putText(detection_source_->getImage(), ss.str(), cv::Point(top(0)-8*id_half_number_of_digits,
                                                                   top(1)), cv::FONT_HERSHEY_SIMPLEX, 0.5, white, 1.7, CV_AA); 	// white id_
  }
  else
  {
    cv::Mat rotated_image = detection_source_->getImage();
    cv::flip(rotated_image.t(), rotated_image, -1);
    cv::flip(rotated_image, rotated_image, 1);
    cv::rectangle(rotated_image, cv::Point(top(1)-8*id_half_number_of_digits, (rotated_image.rows - top(0)+1)-12),
                  cv::Point(top(1) +12*id_half_number_of_digits,  (rotated_image.rows - top(0)+1)+2), darkBlue, CV_FILLED,
                  visibility_ == VISIBLE ? 8 : 1);//, 0);
    cv::putText(rotated_image, ss.str(), cv::Point(top(1)-8*id_half_number_of_digits, rotated_image.rows - top(0)+1),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, white, 1.7, CV_AA); 	// white id_
    cv::flip(rotated_image, rotated_image, -1);
    cv::flip(rotated_image, rotated_image, 1);
    rotated_image = rotated_image.t();
    detection_source_->setImage(rotated_image);
  }
  //TODO end loop
}

void
TrackObject::createMarker(visualization_msgs::MarkerArray::Ptr& msg)
{
  if(visibility_ == TrackObject::NOT_VISIBLE)
    return;

  double _x, _y;
  filter_->getState(_x, _y);

  visualization_msgs::Marker marker;

  marker.header.frame_id = frame_id_;
  marker.header.stamp = ros::Time::now();

  marker.ns = "objects";
  marker.id = id_;

  
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;

  marker.pose.position.x = _x;
  marker.pose.position.y = _y;
  marker.pose.position.z = z_;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  marker.scale.x = 0.1;
  marker.scale.y = 0.1;
  marker.scale.z = 0.1;

  marker.color.r = color_(2);
  marker.color.g = color_(1);
  marker.color.b = color_(0);
  marker.color.a = 1.0;

  marker.lifetime = ros::Duration(0.2);

  msg->markers.push_back(marker);

  ///////////////////////////////////

  visualization_msgs::Marker text_marker;

  text_marker.header.frame_id = frame_id_;
  text_marker.header.stamp = ros::Time::now();

  text_marker.ns = "names";
  text_marker.id = id_;

  text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text_marker.action = visualization_msgs::Marker::ADD;

   std::stringstream ss;
   ss << id_ << ":" << object_name_;
  text_marker.text = ss.str();
 // text_marker.text = object_name_;

  text_marker.pose.position.x = _x;
  text_marker.pose.position.y = _y;
  //      text_marker.pose.position.z = z_ + (height_/2) + 0.1;
  text_marker.pose.position.z = z_+ 0.1; //for object markers
  text_marker.pose.orientation.x = 0.0;
  text_marker.pose.orientation.y = 0.0;
  text_marker.pose.orientation.z = 0.0;
  text_marker.pose.orientation.w = 1.0;

  text_marker.scale.x = 0.2;
  text_marker.scale.y = 0.2;
  text_marker.scale.z = 0.2;

  text_marker.color.r = color_(2);
  text_marker.color.g = color_(1);
  text_marker.color.b = color_(0);
  text_marker.color.a = 1.0;

  text_marker.lifetime = ros::Duration(0.2);

  msg->markers.push_back(text_marker);

}

bool
TrackObject::getPointXYZRGB(pcl::PointXYZRGB& p)
{
  if(visibility_ == TrackObject::NOT_VISIBLE)
    return false;

  double _x, _y;
  filter_->getState(_x, _y);

  p.x = float(_x);
  p.y = float(_y);
  p.z = float(z_);
  uchar* rgb_ptr = (uchar*)&p.rgb;
  *rgb_ptr++ = uchar(color_(0) * 255.0f);
  *rgb_ptr++ = uchar(color_(1) * 255.0f);
  *rgb_ptr++ = uchar(color_(2) * 255.0f);
  return true;
}

void
TrackObject::toMsg(opt_msgs::Track &track_msg, bool vertical)
{

  double _x, _y;
  filter_->getState(_x, _y);

  track_msg.id = id_;
  track_msg.x = _x;
  track_msg.y = _y;
  track_msg.height = height_;
  track_msg.distance = distance_;
  track_msg.object_name=object_name_;
  track_msg.age = age_;
  track_msg.confidence = - data_association_score_;   // minus for transforming distance into a sort of confidence
  track_msg.visibility = visibility_;

  Eigen::Vector3d top(_x, _y, z_ + (height_/2));
  Eigen::Vector3d bottom(_x, _y, z_ - (height_/2));
  top = detection_source_->transformToCam(top);
  bottom = detection_source_->transformToCam(bottom);
  if (not vertical)
  {
    track_msg.box_2D.height = int(std::abs((top - bottom)(1)));
    track_msg.box_2D.width = track_msg.box_2D.height / 2;
    track_msg.box_2D.x = int(top(0)) - track_msg.box_2D.height / 4;
    track_msg.box_2D.y = int(top(1));
  }
  else
  {
    track_msg.box_2D.width = int(std::abs((top - bottom)(0)));
    track_msg.box_2D.height = track_msg.box_2D.width / 2;
    track_msg.box_2D.x = int(top(0)) - track_msg.box_2D.width;
    track_msg.box_2D.y = int(top(1)) - track_msg.box_2D.width / 4;
  }
}

void
TrackObject::zone_msg(json zone_json, int n_zones, opt_msgs::Track &track_msg, bool vertical)
{
  double _x, _y;
  filter_->getState(_x, _y);
  bool inside_area_cube = false;
  int zone_id;
  std::string zone_string;
  
  double x_min;
  double y_min;
  double z_min;
  double x_max;
  double y_max;
  double z_max;
  for (zone_id = 0; zone_id < n_zones; zone_id++)
  {
    zone_string = std::to_string(zone_id);
    /// zone: 0, 1, 2
    for (auto& zone : zone_json.items()){
      if (zone.key() == zone_string) {
        //sensor: name0, name1, etc. 
        for (auto& reference : zone.value().items()) {
          if (reference.key() == frame_id_) {
            //boundary: min/max
            for (auto& boundary : reference.value().items()){
              if (boundary.key() == "min"){
                // min_boundary: sensor_name, world
                for (auto& min_boundary : boundary.value().items()){
                  //x, y, z
                  if (min_boundary.key() == "world"){
                    for (auto& min_value : min_boundary.value().items()){
                      if (min_value.key() == "x"){
                        x_min = min_value.value();
                      }
                      if (min_value.key() == "y"){
                        y_min = min_value.value();
                      }
                      if (min_value.key() == "z"){
                        z_min = min_value.value();
                      }
                    }
                  }
                }
              }
              if (boundary.key() == "max"){
                // max_boundary: sensor_name, world
                for (auto& max_boundary : boundary.value().items()){
                  // sensor_name, world
                  if (max_boundary.key() == "world"){
                    //x, y, z
                    for (auto& max_value : max_boundary.value().items()){
                      if (max_value.key() == "x"){
                        x_max = max_value.value();
                      }
                      if (max_value.key() == "y"){
                        y_max = max_value.value();
                      }
                      if (max_value.key() == "z"){
                        z_max = max_value.value();
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // python representation here
    // need a world view here bc each detection was transformed
    //double x_min = zone_json[zone_string][frame_id_]["min"]["world"]["x"];
    //double y_min = zone_json[zone_string][frame_id_]["min"]["world"]["y"];
    //double z_min = zone_json[zone_string][frame_id_]["min"]["world"]["z"];
    //double x_max = zone_json[zone_string][frame_id_]["max"]["world"]["x"];
    //double y_max = zone_json[zone_string][frame_id_]["max"]["world"]["y"];
    //double z_max = zone_json[zone_string][frame_id_]["max"]["world"]["z"];
    inside_area_cube = (_x <= x_max && _x >= x_min) && (_y <= y_max && _y >= y_min) && (z_ <= z_max && z_ >= z_min);
    // I think this works. 
    if (inside_area_cube) {
      break;
    }
  }

  track_msg.id = id_;
  track_msg.x = _x;
  track_msg.y = _y;
  track_msg.height = height_;
  track_msg.distance = distance_;
  track_msg.object_name=object_name_;
  track_msg.age = age_;
  track_msg.confidence = - data_association_score_;   // minus for transforming distance into a sort of confidence
  track_msg.visibility = visibility_;

  Eigen::Vector3d top(_x, _y, z_ + (height_/2));
  Eigen::Vector3d bottom(_x, _y, z_ - (height_/2));
  top = detection_source_->transformToCam(top);
  bottom = detection_source_->transformToCam(bottom);
  if (not vertical)
  {
    track_msg.box_2D.height = int(std::abs((top - bottom)(1)));
    track_msg.box_2D.width = track_msg.box_2D.height / 2;
    track_msg.box_2D.x = int(top(0)) - track_msg.box_2D.height / 4;
    track_msg.box_2D.y = int(top(1));
  }
  else
  {
    track_msg.box_2D.width = int(std::abs((top - bottom)(0)));
    track_msg.box_2D.height = track_msg.box_2D.width / 2;
    track_msg.box_2D.x = int(top(0)) - track_msg.box_2D.width;
    track_msg.box_2D.y = int(top(1)) - track_msg.box_2D.width / 4;
  }
  
  if (inside_area_cube) {
    track_msg.zone_id = zone_id;
  } else {
    // they're in transit
    track_msg.zone_id = 1000;
  }

}


open_ptrack::detection::DetectionSource*
TrackObject::getDetectionSource()
{
  return detection_source_;
}

void
TrackObject::setVelocityInMotionTerm (bool velocity_in_motion_term, double acceleration_variance, double position_variance)
{
  velocity_in_motion_term_ = velocity_in_motion_term;

  // Re-initialize Kalman filter
  filter_->setPredictModel (acceleration_variance);
  filter_->setObserveModel (position_variance);
  double x, y;
  filter_->getState(x, y);
  filter_->init(x, y, distance_, velocity_in_motion_term_);

  *tmp_filter_ = *filter_;
}

void
TrackObject::setAccelerationVariance (double acceleration_variance)
{
  filter_->setPredictModel (acceleration_variance);
  tmp_filter_->setPredictModel (acceleration_variance);
}

void
TrackObject::setPositionVariance (double position_variance)
{
  filter_->setObserveModel (position_variance);
  tmp_filter_->setObserveModel (position_variance);
}
} /* namespace tracking */
} /* namespace open_ptrack */
