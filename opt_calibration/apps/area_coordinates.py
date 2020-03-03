#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import sys, math
import dlib
import datetime
import numpy
import cProfile
import multiprocessing

import tf
import rospy
import rospkg
import cv_bridge
import message_filters
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from opt_msgs.msg import *
import os
import mxnet as mx
import mxnet
import gluoncv
from mxnet.gluon.data.vision import transforms
from dynamic_reconfigure.server import Server
# TODO add hand detection config
#from recognition.cfg import HandDetectionConfig
import recognition_utils as recutils

#os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
#os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
#os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"

""" script for basic image transforming utilities"""
import random

import cv2
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms


TRANSFORM_FN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

def _get_interp_method(interp, sizes=()):
    """Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
    """
    # pylint: disable=C0103
    if interp == 9:
        # pylint: disable=R1705
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            # pylint: disable=R1705
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp

# pylint: disable=C0111, C0103, C0301, R0913
def resize_short_within(img, short=512, max_size=1024, mult_base=32, interp=2, debug=False):
    """
    similar function to gluoncv's resize short within code that resizes
    the short side of the image and scales the long side to that new short size

    Args:
    -----
    img: np.array
        the image you want to resize
    short: int
        the desired short size
    max_size: int
        the maximum size the short side can take
    mult_base: int
        how to scale the resizing
    interp: int
        the interpretation method used by _get_interp_method
    debug: Bool
        prints the output of the new-size
    Returns:
    --------
    img: np.array
        the resized image
    """
    h, w, _ = img.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base), int(np.round(h * scale / mult_base) * mult_base))
    if debug:
        print(new_w, new_h)
    img = cv2.resize(img, (new_w, new_h), interpolation=_get_interp_method(interp, (h, w, new_h, new_w)))
    return img


def load_image_for_yolo(img, short=512, max_size=1024, stride=32, stack=False, ctx=None):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 image or list of images.
    Parameters
    ----------
    img : np.array or list of np arrays
        Image  to be loaded.
    short : int, default=416
        Resize image short side to this `short` and keep aspect ratio. Note that yolo network
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our YOLO implementation.
    stride : int, optinal, default is 32
        The stride constraint due to precised alignment of bounding box prediction module.
        Image's width and height must be multiples of `stride`. Use `stride = 1` to
        relax this constraint.
    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.
    """

    img = resize_short_within(img, short, max_size, mult_base=stride)
    img = mx.nd.array(img)
    orig_img = img.asnumpy().astype('uint8')

    out = TRANSFORM_FN(img)
    if not stack:
        out = out.expand_dims(0)
    if ctx:
        out = out.as_in_context(ctx)
    return out, orig_img

def detections_to_original(bboxes, from_image, to_image, max_size=1024, mult_base=32, absolute=False):
    #if isinstance(bboxes, mx.ndarray.ndarray.NDArray):
    #    bboxes = bboxes.asnumpy()
    short = min(to_image.shape[:-1])
    h, w, _ = from_image.shape    
    #if bboxes.max() > 1.0:
    if absolute: # to normalized 0-1
        height = from_image.shape[0]
        width = from_image.shape[1]
        bboxes[:, (0, 2)] /= width
        bboxes[:, (1, 3)] /= height

    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)

    bboxes[:, (0, 2)] *= scale
    bboxes[:, (1, 3)] *= scale

    if absolute: # to normalized 0-1
        height = to_image.shape[0]
        width = to_image.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    return bboxes

class HandDetectionNode:
    """
    this class is reponsible for doing the primary detecions in a specific area
    
    the first in the "tree"

    image -> this node -> fine-grained/specialized node

    image -> detect people - pub to detections ->

    or

    image -> detect hands -> publish
    
    """
    def __init__(self, sensor_name):
	    import time
        print("sleeping 1 minute")
	    time.sleep(60*1)
        #self.cfg_server = Server(HandDetectionConfig, self.cfg_callback)
        self.params = '/opt/catkin_ws/src/open_ptrack/recognition/data/yolo3_mobilenet1.0_hands.params'
        self.confidence_thresh = .4
        self.visualization = False
        self.cv_bridge = cv_bridge.CvBridge()
        self.classes = ["hands"]
        # do this in config
        self.ctx = mx.gpu(0)
        model_name = 'yolo3_mobilenet1.0'
        self.net = gluoncv.model_zoo.get_model("%s_custom" % model_name, classes=self.classes)
        self.net.load_parameters(self.params)
        self.net.collect_params().reset_ctx(self.ctx)
        self.net.hybridize(static_alloc=True)

        self.sensor_name = "{}".format(sensor_name)
        self.hand_detections_topic = '/hand/detections'
        self.bounding_box_topic = self.sensor_name + self.hand_detections_topic + '/bounding_boxes'

        # get transformation between world, color, and depth images
        now = rospy.Time(0)
        tf_listener = tf.TransformListener()
        print(self.sensor_name))
        self.ir2rgb = recutils.lookupTransform(tf_listener, self.sensor_name + '_infra1_optical_frame', self.sensor_name + '_color_optical_frame', 5.0, now)
        print('--- ir2rgb ---\n', self.ir2rgb)

    
        # set publishers
        self.pub = rospy.Publisher(self.hand_detections_topic, DetectionArray, queue_size=10)
        self.pub_local = rospy.Publisher(self.sensor_name + self.hand_detections_topic, DetectionArray, queue_size=10)
        self.bounding_box_pub = rospy.Publisher(self.bounding_box_topic, Image, queue_size=10)
        
        # set transforms
        self.ir2rgb = recutils.lookupTransform(tf_listener, self.sensor_name + '_infra1_optical_frame', self.sensor_name + '_color_optical_frame', 5.0, now)
        # set subscribers

        rospy.client.wait_for_message(self.sensor_name + '/color/image_raw', Image, 20.0)
        img_subscriber = message_filters.Subscriber(self.sensor_name + '/color/image_raw', Image)
        # TODO add depth subscriber name
        depth_img_subscriber = message_filters.Subscriber(self.sensor_name + '/depth/image_rect_raw', Image)

        self.subscribers = [
            img_subscriber,
            depth_img_subscriber,
            message_filters.Subscriber(self.sensor_name + '/color/camera_info', CameraInfo)
        ]
        self.ts = recutils.TimeSynchronizer(self.subscribers, 60, 1000)
        self.ts.registerCallback(self.callback)
        # figure out the proper channel here
        self.reset_time_sub = rospy.Subscriber('/reset_time', Empty, self.reset_time)
        print("init complete")

    # callback for dynamic configure
    def cfg_callback(self, config, level):
        package_path = rospkg.RosPack().get_path('recognition')
        self.params = package_path + config.params_path      # the path to the face detector model file
        #self.confidence_thresh = config.confidence_thresh               # the threshold for confidence of face detection
        self.confidence_thresh = .4
        self.visualization = config.visualization                       # if true, the visualization of the detection will be shown
        # I would do this here, but instead, we'll force gpu for testing
        #if config.gpu:
        #    self.ctx = mx.gpu(self.gpu)
        #else:
        #    self.ctx = mx.cpu()
        print('--- cfg_callback ---')
        print('confidence_thresh', config.confidence_thresh)
        print('visualization', config.visualization)
        return config

    def reset_time(self, msg):
        print('reset time')
	    self.ts = message_filters.ApproximateTimeSynchronizer(self.subscribers, 200, 0.00001)
	    self.ts.registerCallback(self.callback)

    # callback
    def callback(self, rgb_image_msg, depth_img_msg, rgb_info_msg):
        print('running callback!')
        detection_array = DetectionArray()
        t1 = rospy.Time.now()

        # read rgb image
        #if type(rgb_image_msg) is CompressedImage:
        #    rgb_image = recutils.decompress(rgb_image_msg)
        #else:
        #    #
        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_image_msg)
        # this should work, no?
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_img_msg)

        # taken from yolo based object detector cpp
        cy = rgb_info_msg.K[2]
        cx = rgb_info_msg.K[5]
        _constant_x =  1.0 / rgb_info_msg.K[0]
        _constant_y = 1.0 /  rgb_info_msg.K[4]

        #rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        #         
        # TODO write IMAGE_TRANSFORM()
        img_for_network, transformed_image = load_image_for_yolo(rgb_image, ctx=self.ctx)
        class_ids, scores, bounding_boxes = self.net(img_for_network)
        #if isinstance(bounding_boxes, mxnet.ndarray.ndarray.NDArray):
        bounding_boxes = bounding_boxes.asnumpy()[0]
        #if isinstance(scores, mx.ndarray.ndarray.NDArray):
        scores = scores.asnumpy()[0]
        #if isinstance(class_ids, mx.ndarray.ndarray.NDArray):
        class_ids = class_ids.asnumpy()[0]
        bboxes = detections_to_original(bounding_boxes, transformed_image, rgb_image, absolute=True)
        
        # TODO write transform()
        # detections = transform(class_ids, scores, bounding_boxes)
        # publish the face detection result           
        detections_list = []
	    print(bboxes)
        for i, bbox in enumerate(bboxes):
            #print(bbox)
            score = scores.flat[i]
            if scores is not None and score < self.confidence_thresh:
                break
            cls_id = int(class_ids.flat[i])# if labels is not None else -1
            if cls_id is None or cls_id == -1:
                break
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]

            object_mask = depth_image[ymin:ymax, xmin:xmax]

            object_name = self.classes[cls_id]
            detection = Detection()
            # detection.header = Header()
            # detection.header.stamp = rospy.get_rostime()

            object_mask = depth_image[xmin:xmax, ymin:ymax]
            median_depth = np.median(object_mask)

            median_x = ((xmax - xmin) / 2) + xmin
            median_y = ((ymax - ymin) / 2) + ymin
            mx =  (median_x - cx) * median_depth * _constant_x
            my = (median_y - cy) * median_depth * _constant_y

            detection.box_3D.p1.x = mx
            detection.box_3D.p1.y = my
            detection.box_3D.p1.z = median_depth
        
            detection.box_3D.p2.x = mx
            detection.box_3D.p2.y = my
            detection.box_3D.p2.z = median_depth
        
            #detection.box_2D.x = medianX
            #detection.box_2D.y = medianY
            #detection.box_2D.width = 0
            #detection.box_2D.height = 0
            detection.height = 0
            detection.confidence = score
            detection.distance = median_depth
            detection.box_2D = BoundingBox2D(x=xmin, y=ymin, width=xmax - xmin, height=ymax - ymin)

            detection.centroid.x = mx
            detection.centroid.y = my
            detection.centroid.z = median_depth
        
            detection.top.x = 0
            detection.top.y = 0
            detection.top.z = 0
        
            detection.bottom.x = 0
            detection.bottom.y = 0
            detection.bottom.z = 0
        
            detection.object_name = object_name
            detections_list.append(detection)

            rgb_image = cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 225, 0), 1)
        
        # do 3D stuff here
        detection_array.header = Header()
        detection_array.header.stamp = rospy.get_rostime()
        detection_array.detections = detections_list
        detection_array.confidence_type = 'yolo'
        detection_array.image_type = 'rgb'

        self.pub.publish(detection_array)
        self.pub_local.publish(detection_array)
        self.bounding_box_pub.publish(self.cv_bridge.cv2_to_imgmsg(rgb_image[..., ::-1]))

        t2 = rospy.Time.now()

        #if self.visualization:
        #   self.visualize(rgb_image, rois, faces, (t2 - t1).to_sec())
        print("callback-time: {} seconds".format((t2 - t1).to_sec()))
        cv2.imshow('rgb_image', rgb_image[..., ::-1])
        cv2.waitKey(30)

def main():
	sensor_name = '/kinect2_head' if len(sys.argv) < 2 else '/' + sys.argv[1]
	print 'sensor_name', sensor_name

	rospy.init_node('hand_detection_node_' + sensor_name[1:])
	node = HandDetectionNode(sensor_name)
	rospy.spin()

if __name__ == '__main__':
	main()

