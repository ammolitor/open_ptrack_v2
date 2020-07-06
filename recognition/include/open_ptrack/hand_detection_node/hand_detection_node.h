#include <open_ptrack/base_node/base_node.h>
#include <open_ptrack/yolo_tvm.hpp>
#ifndef OPEN_PTRACK_HAND_DETECTION_NODE_HAND_DETECTION_NODE_H_
#define OPEN_PTRACK_HAND_DETECTION_NODE_HAND_DETECTION_NODE_H_

namespace open_ptrack
{
  namespace hand_detection_node
  {

    /** \brief DetectionNode estimates the ground plane equation from a 3D point cloud */
    class HandDetectionNode: public open_ptrack::base_node::BaseNode
    {
      private:
        std::unique_ptr<NoNMSYoloFromConfig> tvm_object_detector;
        // Publishers
        ros::Publisher detections_pub;
        image_transport::Publisher image_pub;

      public:
        int gluon_to_rtpose[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
        /** \brief HandDetectionNode Constructor. */
        HandDetectionNode(ros::NodeHandle& nh, std::string sensor_string, json zone);
        //  open_ptrack::base_node::BaseNode(nh, sensor_string, zone)
        //{}
        
        /** \brief Destructor. */
        virtual ~HandDetectionNode ();

        void callback(const PointCloudT::ConstPtr& cloud_);

        void basic_callback(const PointCloudT::ConstPtr& cloud_);

    };
  } /* namespace base_node */
} /* namespace open_ptrack */
#endif /* OPEN_PTRACK_HAND_DETECTION_NODE_HAND_DETECTION_NODE_H_ */

