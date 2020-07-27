#include <open_ptrack/nodes/base_node.h>

#ifndef OPEN_PTRACK_NODES_POSE_NODE_H_
#define OPEN_PTRACK_NODES_POSE_NODE_H_

namespace open_ptrack
{
  namespace nodes
  {

    /** \brief DetectionNode estimates the ground plane equation from a 3D point cloud */
    class PoseNode: public BaseNode
    {
      private:
        std::unique_ptr<NoNMSPoseFromConfig> tvm_pose_detector;
        // Publishers
        ros::Publisher detections_pub;
        ros::Publisher skeleton_pub;
        image_transport::Publisher image_pub;

      public:
        int gluon_to_rtpose[17] = {0, -1, -1, -1, -1, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};
        /** \brief PoseNode Constructor. */
        PoseNode(ros::NodeHandle& nh, std::string sensor_string, json zone);//:
        //  open_ptrack::nodes::BaseNode(nh, sensor_string, zone)
        //{ }
        
        /** \brief Destructor. */
        virtual ~PoseNode ();

        void callback(const PointCloudT::ConstPtr& cloud_);

    };
  } /* namespace nodes */
} /* namespace open_ptrack */
#include <open_ptrack/nodes/pose_node.hpp>
#endif /* OPEN_PTRACK_NODES_POSE_NODE_H_ */

