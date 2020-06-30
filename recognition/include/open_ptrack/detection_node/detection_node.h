#include <open_ptrack/base_node/base_node.h>


#ifndef OPEN_PTRACK_DETECTION_NODE_DETECTION_NODE_H_
#define OPEN_PTRACK_DETECTION_NODE_DETECTION_NODE_H_

namespace open_ptrack
{
  namespace detection_node
  {

    /** \brief DetectionNode estimates the ground plane equation from a 3D point cloud */
    class DetectionNode: public open_ptrack::base_node::BaseNode
    {
      private:
        std::unique_ptr<NoNMSYoloFromConfig> tvm_object_detector;
        // Publishers
        ros::Publisher detections_pub;
        image_transport::Publisher image_pub;

        // Subscribers
        ros::Subscriber rgb_sub;
        ros::Subscriber camera_info_matrix;
        ros::Subscriber detector_sub;
      public:
        int gluon_to_rtpose[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
        /** \brief DetectionNode Constructor. */
        DetectionNode(ros::NodeHandle& nh, std::string sensor_string, json zone):
            open_ptrack::base_node::BaseNode(ros::NodeHandle& nh, std::string sensor_string, json zone)
          {  }
        
        /** \brief Destructor. */
        virtual ~DetectionNode ();

        void callback(const PointCloudT::ConstPtr& cloud_);

    };
  } /* namespace base_node */
} /* namespace open_ptrack */
#include <open_ptrack/detection_node/detection_node.hpp>
#endif /* OPEN_PTRACK_DETECTION_NODE_DETECTION_NODE_H_ */

