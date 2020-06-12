cmake_minimum_required(VERSION 2.8.3)
project(recognition)

add_definitions(-std=c++11)

set(TVM_ROOT "/opt/src/tvm")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fPIC -pthread -L${TVM_ROOT}/build -ldl -lpthread -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu/ -lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_viz -lopencv_core")
# -lcublas  -lcuda -lcudart -lcudnn

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fPIC -pthread -L${TVM_ROOT}/build -ldl -lpthread -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -L/usr/lib/aarch64-linux-gnu/ -lopencv_shape -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_viz -lopencv_core -L/usr/local/cuda/lib64 -lcublas  -lcuda -lcudart -lcudnn -ltvm_runtime")

set(Torch_DIR "/opt/src/pytorch/torch/share/cmake/Torch")

# list(APPEND CMAKE_PREFIX_PATH "/opt/src/pytorch/torch")
find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#set(Torch_DIR "$ENV{TORCH_ROOT}/torch/share/cmake/Torch/")

include_directories(
    ${TVM_ROOT}/include
    ${TVM_ROOT}/build
    ${TVM_ROOT}/3rdparty/dmlc-core/include
    ${TVM_ROOT}/3rdparty/dlpack/include
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/cuda/include
    /usr/local/cuda/lib64)

# set(CMAKE_BUILD_TYPE RelWithDebInfo)
find_package(catkin REQUIRED COMPONENTS
    rospy
    roscpp
    tf
    tf_conversions
    std_msgs
    sensor_msgs
    pcl_ros
    pcl_conversions
    opt_msgs
    cv_bridge
    dynamic_reconfigure
    image_transport
    # detection
    # adding from detection
    cmake_modules
    calibration_common
    message_filters 
    nodelet
    rosconsole
    opt_utils
    std_msgs 
    rostime
    compressed_image_transport 
    compressed_depth_image_transport 
    kinect2_bridge
)

# list(APPEND LINKER_LIBS ${CMAKE_DL_LIBS})
# adding from detection
include_directories(${catkin_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
include_directories( ${BOOST_INCLUDE_DIRS} )



find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})



# add eigen
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS} include ${catkin_INCLUDE_DIRS})

include_directories(include ${catkin_INCLUDE_DIRS})

add_service_files(FILES OPTSetPredefinedFaces.srv OPTSaveRegisteredFaces.srv OPTLoadRegisteredFaces.srv)
generate_messages(DEPENDENCIES std_msgs sensor_msgs opt_msgs)

generate_dynamic_reconfigure_options(
    cfg/FaceDetection.cfg
    cfg/FaceEmbedding.cfg
    cfg/FaceFeatureExtraction.cfg
    cfg/FaceRecognition.cfg
    cfg/GenDetection.cfg
    cfg/HandDetection.cfg
    cfg/RecognitionVisualization.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES face_reconstruction
  CATKIN_DEPENDS roscpp std_msgs opt_msgs message_filters sensor_msgs image_transport pcl_ros pcl_conversions opt_utils
#  DEPENDS system_lib
)
message(STATUS CURRENT_DIR: ${CMAKE_CURRENT_LIST_DIR})
find_package(CUDA REQUIRED)

# initial variables\
list(APPEND LINKER_LIBS "${CMAKE_DL_LIBS}")
# list(APPEND LINKER_LIBS opencv_dnn)
#list(APPEND LINKER_LIBS "${TORCH_LIBRARIES}")
list(APPEND LINKER_LIBS "${CMAKE_DL_LIBS}")
list(APPEND LINKER_LIBS "${CUDA_CUDA_LIBRARY}")
list(APPEND LINKER_LIBS "${CUDA_CURAND_LIBRARY}")
list(APPEND LINKER_LIBS "${CUDA_CUDNN_LIBRARY}")
list(APPEND LINKER_LIBS "${CUDA_CUDART_LIBRARY}")
list(APPEND LINKER_LIBS "${CUDA_CUBLAS_LIBRARY}")
list(APPEND LINKER_LIBS "${CUDA_NVRTC_LIBRARY}")
list(APPEND LINKER_LIBS "${CUDA_TOOLKIT_ROOT_DIR}")
list(APPEND LINKER_LIBS "${TVM_RUNTIME_LINKER_LIBS}")
list(APPEND LINKER_LIBS "${RUNTIME_SRCS}")
list(APPEND LINKER_LIBS "${OpenCV_LIBS}")
list(APPEND LINKER_LIBS "tvm_runtime")
#list(APPEND LINKER_LIBS "${TORCH_LIBRARIES}")

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})

# Hungarian variables
list(APPEND HUNGARIAN_LIBS include/open_ptrack/hungarian/Hungarian.cpp)
list(APPEND HUNGARIAN_LIBS include/open_ptrack/hungarian/Hungarian.h)

# face-detecion variables
list(APPEND DETECTION_LIBS include/open_ptrack/MTCNN/mtcnn_opencv.cpp)
list(APPEND DETECTION_LIBS include/open_ptrack/FacePreprocess.h)
list(APPEND DETECTION_LIBS include/open_ptrack/MTCNN/mtcnn_opencv.hpp)
list(APPEND DETECTION_LIBS include/open_ptrack/RetinaFace/anchor_generator.cpp)
list(APPEND DETECTION_LIBS include/open_ptrack/RetinaFace/anchor_generator.h)
list(APPEND DETECTION_LIBS include/open_ptrack/RetinaFace/config.cpp)
list(APPEND DETECTION_LIBS include/open_ptrack/RetinaFace/config.h)
list(APPEND DETECTION_LIBS include/open_ptrack/RetinaFace/tools.cpp)
list(APPEND DETECTION_LIBS include/open_ptrack/RetinaFace/tools.h)
list(APPEND DETECTION_LIBS include/open_ptrack/RetinaFace/ulsMatF.h)
list(APPEND DETECTION_LIBS include/FacePreprocess.h)
list(APPEND DETECTION_LIBS include/facetracking.hpp)

#tvm_runtime_pack.cc
ADD_LIBRARY(facedetection
        include/open_ptrack/hungarian/Hungarian.cpp
        include/open_ptrack/hungarian/Hungarian.h
        include/open_ptrack/FacePreprocess.h
        include/open_ptrack/RetinaFace/anchor_generator.cpp
        include/open_ptrack/RetinaFace/anchor_generator.h
        include/open_ptrack/RetinaFace/config.cpp
        include/open_ptrack/RetinaFace/config.h
        include/open_ptrack/RetinaFace/tools.cpp
        include/open_ptrack/RetinaFace/tools.h
        include/open_ptrack/RetinaFace/ulsMatF.h
        include/open_ptrack/facetracking.hpp)

#ADD_LIBRARY(segmentation_clustering
#        include/open_ptrack/person_clustering/person_cluster.cpp
#        include/open_ptrack/person_clustering/person_cluster.h
#        include/open_ptrack/person_clustering/head_based_subclustering.cpp
#        include/open_ptrack/person_clustering/head_based_subclustering.h
#        )
#        #        #include/open_ptrack/ground_segmentation/ground_segmentation.cpp
#        #include/open_ptrack/ground_segmentation/ground_segmentation.h

#ADD_LIBRARY(object_detection
#        include/open_ptrack/yolo_tvm.hpp)

message(STATUS LINKER_LIBS: "${LINKER_LIBS}")
message(STATUS OPENCV_LIBS: "${OpenCV_LIBS}")

###############################################################################
add_executable(face_recognition_node apps/face_recognition_node.cpp)
target_link_libraries(face_recognition_node ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_dependencies(face_recognition_node ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)

###############################################################################
add_executable(map_generation_node apps/map_generation_node.cpp)
target_link_libraries(map_generation_node ${OpenCV_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES})
add_dependencies(map_generation_node ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)

###############################################################################
add_executable(face_detection_extraction_node apps/face_detection_extraction_node.cpp include/open_ptrack/hungarian/Hungarian.cpp)
target_link_libraries(face_detection_extraction_node PUBLIC Threads::Threads ${OpenCV_LIBS} 
                                          ${catkin_LIBRARIES}
                                          ${LINKER_LIBS}
                                          facedetection
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS})

add_dependencies(face_detection_extraction_node ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(face_detection_extraction_node PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(face_detection_extraction_node PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(face_detection_extraction_node PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###############################################################################
add_executable(tvm_detection_node apps/tvm_detection_node.cpp include/open_ptrack/yolo_tvm.hpp)
target_link_libraries(tvm_detection_node PUBLIC Threads::Threads ${OpenCV_LIBS} 
                                          ${catkin_LIBRARIES} 
                                          ${LINKER_LIBS}
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS}
                                          ${TORCH_LIBRARIES})

add_dependencies(tvm_detection_node ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(tvm_detection_node PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(tvm_detection_node PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(tvm_detection_node PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###############################################################################
add_executable(tvm_hand_detection_node apps/tvm_hand_detection_node.cpp include/open_ptrack/yolo_tvm.hpp)
target_link_libraries(tvm_hand_detection_node PUBLIC Threads::Threads ${OpenCV_LIBS} 
                                          ${catkin_LIBRARIES} 
                                          ${LINKER_LIBS}
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS}
                                          ${TORCH_LIBRARIES})

add_dependencies(tvm_hand_detection_node ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(tvm_hand_detection_node PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(tvm_hand_detection_node PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(tvm_hand_detection_node PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###############################################################################
add_executable(area_node apps/area_node.cpp)
target_link_libraries(area_node PUBLIC Threads::Threads ${OpenCV_LIBS}
                                          ${catkin_LIBRARIES}
                                          ${LINKER_LIBS}
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS}
                                          ${TORCH_LIBRARIES})

add_dependencies(area_node ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(area_node PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(area_node PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(area_node PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

###############################################################################
add_executable(debug_tvm_detector apps/debug_tvm_detector.cpp include/open_ptrack/yolo_tvm.hpp)
target_link_libraries(debug_tvm_detector PUBLIC Threads::Threads ${OpenCV_LIBS} 
                                          ${catkin_LIBRARIES} 
                                          ${LINKER_LIBS}
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS}
                                          ${TORCH_LIBRARIES})

add_dependencies(debug_tvm_detector ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(debug_tvm_detector PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(debug_tvm_detector PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(debug_tvm_detector PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###############################################################################
add_executable(debug_pose_model apps/debug_pose_model.cpp include/open_ptrack/yolo_tvm.hpp 
                                include/open_ptrack/hungarian/Hungarian.cpp 
                                include/open_ptrack/ground_segmentation/ground_segmentation.h 
                                include/open_ptrack/ground_segmentation/ground_segmentation.hpp
                                include/open_ptrack/person_clustering/person_cluster.hpp
                                include/open_ptrack/person_clustering/person_cluster.h
                                include/open_ptrack/person_clustering/head_based_subclustering.hpp
                                include/open_ptrack/person_clustering/head_based_subclustering.h
                                )
SET_TARGET_PROPERTIES(debug_pose_model PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(debug_pose_model PUBLIC Threads::Threads ${OpenCV_LIBS} 
                                          ${catkin_LIBRARIES} 
                                          ${LINKER_LIBS}
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS}
                                          ${TORCH_LIBRARIES}
                                          ${PCL_LIBRARY_DIRS})

add_dependencies(debug_pose_model ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(debug_pose_model PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(debug_pose_model PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(debug_pose_model PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###############################################################################
add_executable(debug_test apps/debug_test.cpp include/open_ptrack/yolo_tvm.hpp)
target_link_libraries(debug_test PUBLIC Threads::Threads ${OpenCV_LIBS} 
                                          ${catkin_LIBRARIES} 
                                          ${LINKER_LIBS}
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS}
                                          ${TORCH_LIBRARIES})

add_dependencies(debug_test ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(debug_test PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(debug_test PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(debug_test PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###############################################################################
add_executable(tvm_pose_recognition_node apps/tvm_pose_recognition_node.cpp include/open_ptrack/yolo_tvm.hpp)
target_link_libraries(tvm_pose_recognition_node PUBLIC Threads::Threads ${OpenCV_LIBS} 
                                          ${catkin_LIBRARIES} 
                                          ${LINKER_LIBS}
                                          ${CUDA_CUDA_LIBRARY}
                                          ${CUDA_CUDART_LIBRARY}
                                          ${CUDA_CURAND_LIBRARY}
                                          ${CUDA_CUDNN_LIBRARY}
                                          ${TVM_RUNTIME_LINKER_LIBS}
                                          ${TORCH_LIBRARIES})

add_dependencies(tvm_pose_recognition_node ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_gencpp)
target_include_directories(tvm_pose_recognition_node PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_definitions(tvm_pose_recognition_node PUBLIC TVM_CUDA_RUNTIME=1)
target_compile_definitions(tvm_pose_recognition_node PUBLIC TVM_USE_GRAPH_RUNTIME_DEBUG=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")