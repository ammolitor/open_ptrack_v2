#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/dnn/dnn.hpp>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <random>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <open_ptrack/nms/nms.h>
#include "tvm_detection_helpers.hpp"
using json = nlohmann::json;

//#ifndef OPEN_PTRACK_MODELS_BASED_SUBCLUSTER_H_
//#define OPEN_PTRACK_MODELS_BASED_SUBCLUSTER_H_

namespace open_ptrack
{
  namespace models
  {

    class NoNMSYoloFromConfig{
        private:
            //working: void * handle;
            std::unique_ptr<tvm::runtime::Module> detector_handle;
            std::unique_ptr<tvm::runtime::Module> pose_handle;

        public:
            // confident in these
            // set these as the default
            std::string deploy_lib_path;
            std::string deploy_graph_path;
            std::string deploy_param_path;
            bool gpu = true;
            int device_id = 0;// = 0;
            int dtype_code = kDLFloat;
            int dtype_bits = 32;
            int dtype_lanes = 1;
            int device_type = kDLGPU;
            // set default here???
            int detector_width;// = 512;
            int detector_height;// = 512;
            int detector_total_input;// = 3 * width * height;
            int in_ndim = 4;
            int detector_out_ndim = 3;
            int max_yolo_boxes = 100;
            // maybe we can dynamically set all of these
            int64_t no_nms_output_size[3] = {1, 322560, 6};
            float thresh = 0.3f;
            int n_dets;

            /**
             * function that reads both the yolo detector and the pose detector
             * 
            */
            NoNMSYoloFromConfig(std::string config_path, std::string ros_package_string) {
                // read config with nlohmann-json
                std::cout << "start model_config reading" << std::endl;
                json model_config;
                std::string package_path = "/opt/catkin_ws/src/open_ptrack/recognition";
                std::string full_path = package_path + config_path;
                std::ifstream json_read(full_path);
                json_read >> model_config;
                std::cout << "model_config read into memory" << std::endl;
                // read detector variables
                std::string detector_lib_path = model_config["detector_deploy_lib_path"];
                std::string detector_graph_path = model_config["detector_deploy_graph_path"];
                std::string detector_param_path = model_config["detector_deploy_param_path"];

                device_id = model_config["device_id"];
                detector_width = model_config["detector_width"]; //(512,512)
                detector_height = model_config["detector_height"]; //(512,512)
                gpu = model_config["gpu"];
                n_dets = model_config["n_dets"];
                no_nms_output_size[1] = n_dets;
                thresh = model_config["thresh"];
                detector_total_input = 1 * 3 * detector_width * detector_height;

                std::string detector_deploy_lib_path = package_path + detector_lib_path;
                std::string detector_deploy_graph_path = package_path + detector_graph_path;
                std::string detector_deploy_param_path = package_path + detector_param_path;

                // set device type -- I think this has to be set here...
                if (gpu){
                    device_type = kDLGPU;
                } else {
                    device_type = kDLCPU;
                }
                // DETECTOR READ
                // read deploy lib
                tvm::runtime::Module detector_mod_syslib = tvm::runtime::Module::LoadFromFile(detector_deploy_lib_path);
                // read deplpy json
                std::ifstream detector_json_in(detector_deploy_graph_path, std::ios::in);
                std::string detector_json_data((std::istreambuf_iterator<char>(detector_json_in)), std::istreambuf_iterator<char>());
                detector_json_in.close();
                // get global function module for graph runtime
                tvm::runtime::Module detector_mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(detector_json_data, detector_mod_syslib,
                                                                                                    device_type, device_id);
                this->detector_handle.reset(new tvm::runtime::Module(detector_mod));
                // parameters in binary
                std::ifstream detector_params_in(detector_deploy_param_path, std::ios::binary);
                std::string detector_params_data((std::istreambuf_iterator<char>(detector_params_in)), std::istreambuf_iterator<char>());
                detector_params_in.close();
                // parameters need to be TVMByteArray type to indicate the binary data
                TVMByteArray detector_params_arr;
                detector_params_arr.data = detector_params_data.c_str();
                detector_params_arr.size = detector_params_data.length();
                tvm::runtime::PackedFunc detector_load_params = detector_mod.GetFunction("load_params");
                detector_load_params(detector_params_arr);
            }
        
            /**
             * \brief function to normalize an image before it's processed by the network
             * \param[in] the raw cv::mat image
             * \return the normalized version of the iamge.
             */  
            cv::Mat preprocess_image(cv::Mat frame, int width, int height, bool convert){
                cv::Size new_size = cv::Size(width, height); // or is it height width????
                cv::Mat resized_image;
                if (convert){
                cv::Mat rgb;
                // bgr to rgb
                cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
                cv::resize(rgb, resized_image, new_size);
                } else {
                cv::resize(frame, resized_image, new_size);
                }
                // resize to 512x512
                cv::Mat resized_image_floats(new_size, CV_32FC3);
                // convert resized image to floats and normalize
                resized_image.convertTo(resized_image_floats, CV_32FC3, 1.0f/255.0f);
                //mimic mxnets 'to_tensor' function
                cv::Mat normalized_image(new_size, CV_32FC3);
                // mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
                cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
                cv::Mat temp;
                temp = resized_image_floats - mean;
                normalized_image = temp / theta;
                return normalized_image;
            }

            // we can set it externally with dynamic reconfigure
            yoloresults* forward_full(cv::Mat frame, float override_threshold)
            {
                //std::cout << "starting function" << std::endl;
                // get height/width dynamically
                cv::Size image_size = frame.size();
                float img_height = static_cast<float>(image_size.height);
                float img_width = static_cast<float>(image_size.width);

                //Set constants and variables -- this is a prob,
                // how to do this at runtime?
                //constexpr int const_dtype_code = dtype_code;
                //constexpr int const_dtype_bits = dtype_bits;
                //constexpr int const_dtype_lanes = dtype_lanes;
                //constexpr int const_device_type = device_type;
                //constexpr int const_device_id = device_id;
                //int64_t in_shape[4] = {1, in_c, in_h, in_w};
                int64_t in_shape[4] = {1, 3, detector_height, detector_width};
                int total_input = 3 * detector_width * detector_height;
                //std::cout << "width: " << detector_width << std::endl;
                //std::cout << "height: " << detector_height << std::endl;
                //std::cout << "total_input: " << total_input << std::endl;
                //std::cout << "device_id: " << device_id << std::endl;
                //std::cout << "dtype_code: " << dtype_code << std::endl;
                //std::cout << "dtype_bits: " << dtype_bits << std::endl;
                //std::cout << "dtype_lanes: " << dtype_lanes << std::endl;
                //std::cout << "device_type: " << device_type << std::endl;

                DLTensor *output_for_nms;
                DLTensor *input;
                float *data_x = (float *) malloc(total_input * sizeof(float));
            
                // allocate memory for results
                yoloresults* results = (yoloresults*)calloc(1, sizeof(yoloresults));
                results->num = 100;
                results->boxes = (bbox_result*)calloc(100, sizeof(bbox_result));

                //std::cout << "about to allocate info" << std::endl;
                // allocate DLTensor memory on device for all the vars needed
                TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
                TVMArrayAlloc(no_nms_output_size, detector_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_for_nms);
                //std::cout << "allocate info finished" << std::endl;

                //copy processed image to DLTensor
                //std::cout << "about to preprocess" << std::endl;
                cv::Mat processed_image = preprocess_image(frame, detector_width, detector_height, true);
                //std::cout << "preprocess finished" << std::endl;
                cv::Mat split_mat[3];
                cv::split(processed_image, split_mat);
                memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
                memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                    processed_image.cols * processed_image.rows * sizeof(float));
                memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                    processed_image.cols * processed_image.rows * sizeof(float));
                TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
                //std::cout << "TVMArrayCopyFromBytes finished" << std::endl;           

                // standard tvm module run
                // get the module, set the module-input, and run the function
                // this is symbolic it ISNT run until TVMSync is performed
                tvm::runtime::Module *mod = (tvm::runtime::Module *) detector_handle.get();
                tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
                set_input("data", input);
                tvm::runtime::PackedFunc run = mod->GetFunction("run");
                run();
                tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
                //std::cout << "run/getoutput/setinput finished" << std::endl;

                // https://github.com/apache/incubator-tvm/issues/979?from=timeline
                //"This may give you some ideas to start with.
                //In general you want to use pinned memory and you want
                //to interleave computation with copying; so you want to
                // be upload the next thing while you are computing the
                //current thing while you are downloading the last thing."
                TVMSynchronize(device_type, device_id, nullptr);
                get_output(0, output_for_nms);
                //std::cout << "TVMSynchronize finished" << std::endl;  

                // copy to output
                //ulsMatF(int cols, int rows, int channels)
                //at(int channel, int row, int c
                MatF yolo_output(6, n_dets, 1); //ulsMatF yolo_output(6, n_dets, 1);
                TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
                //TVMArrayCopyToBytes(output_for_nms, yolo_output.m_data, 1* n_dets * 6 * sizeof(float));
                //std::cout << "TVMSynchronize finished" << std::endl;  
                
                //std::cout << "starting nms" << std::endl;
                //auto tick = Clock::now();
                std::vector<sortable_result> tvm_results;
                std::vector<sortable_result> proposals;
                proposals.clear();
                tvm_nms_cpu(proposals, yolo_output, override_threshold, override_threshold, tvm_results);
                //std::cout << "ending nms" << std::endl;

                TVMArrayFree(input);
                TVMArrayFree(output_for_nms); // may have to move this to the bottom.
                input = nullptr;
                free(data_x);
                data_x = nullptr;

                float fheight = static_cast<float>(img_height);
                float fwidth = static_cast<float>(img_width);
                int new_num = 0;
                for (int i = 0; i < tvm_results.size(); ++i) {

                    float xmin;
                    float ymin;
                    float xmax;
                    float ymax;

                    float score = tvm_results[i].probs;
                    float label = tvm_results[i].cls;
                    if (score < thresh) continue;
                    if (label < 0) continue;
                    // people only
                    if (label > 0) continue;

                    int cls_id = static_cast<int>(label);
                    xmin = tvm_results[i].xmin;
                    ymin = tvm_results[i].ymin;
                    xmax = tvm_results[i].xmax;
                    ymax = tvm_results[i].ymax;
                    //SCALE to frame height
                    xmin = xmin * (img_width/detector_height); // move down to 480 space  ()
                    ymin = ymin / (detector_width/img_height); // move up to 640
                    xmax = xmax * (img_width/detector_height);
                    ymax = ymax / (detector_width/img_height);
                    
                    results->boxes[i].xmin = xmin;
                    results->boxes[i].ymin = ymin;
                    results->boxes[i].xmax = xmax;
                    results->boxes[i].ymax = ymax;
                    results->boxes[i].id = cls_id;
                    results->boxes[i].score = score;
                    new_num+=1;
                };
                results->num = new_num;
                //std::cout << "torch array iter finished" << std::endl;
                tvm_results = std::vector<sortable_result>();
                proposals = std::vector<sortable_result>();           
                return results;
            }
    };
  } /* namespace models */
} /* namespace open_ptrack */
//#include <open_ptrack/person_clustering/height_map_2d.hpp>
