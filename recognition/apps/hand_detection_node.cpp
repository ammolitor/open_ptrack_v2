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
#include <open_ptrack/hand_detection_node/hand_detection_node.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char** argv) {

  std::string sensor_name;
  json zone_json;
  std::string area_package_path = ros::package::getPath("recognition");
  std::string area_hard_coded_path = area_package_path + "/cfg/area.json";
  std::ifstream area_json_read(area_hard_coded_path);
  area_json_read >> zone_json;

  ros::init(argc, argv, "hand_detection_node");
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;
  pnh.param("sensor_name", sensor_name, std::string("d435"));
  std::cout << "sensor_name: " << sensor_name << std::endl;
  HandDetectionNode node(nh, sensor_name, zone_json);
  std::cout << "pose node initialized " << std::endl;
  ros::spin();
  return 0;
}

