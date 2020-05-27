//
// Created by markson zhang on 2019-03-20.
//
#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "stdlib.h"
#include <iostream>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "RetinaFace/anchor_generator.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "RetinaFace/config.h"
#include "RetinaFace/tools.h"
#include "RetinaFace/ulsMatF.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/dnn/dnn.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/dnn/dnn.hpp>
using namespace std;
using namespace cv;

// Tunable Parameters
const int avg_face = 1;
const int minSize = 60;
const int stage = 4;
//const cv::Size frame_size = Size(1280,760);
const cv::Size frame_size = Size(640, 480);
const string prefix = "/home/mkrzus/github/open_ptrack_lite/Face-Recognition-Cpp/models/linux";
const char arcface_model[30] = "y1-arcface-emore_109";

struct _FaceInfo {
    /**
     * Structure _FaceInfo
     * face_count: the count of total face
     * face_details: the [confidence, x, y, w, h, eyes, nose, cheek] coordinators
     */
    int face_count;
    std::vector<std::array<double, 15>> face_details;
//    double face_details[][15];
};

struct RetinaOutput {
    std::vector<Anchor> result;
    cv::Point ratio;
};

cv::Mat non_graph_preprocess_image(cv::Mat frame){
    cv::Size new_size = cv::Size(512, 512);
    cv::Mat resized_image;
    cv::Mat rgb;
    // bgr to rgb
    cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
    // resize to 512x512
    cv::resize(rgb, resized_image, new_size);
    cv::Mat resized_image_floats(new_size, CV_32FC3);
    // convert resized image to floats and normalize
    resized_image.convertTo(resized_image_floats, CV_32FC3, 1.0f/255.0f);
    //mimic mxnets 'to_tensor' function 
    cv::Mat normalized_image(new_size, CV_32FC3);
    // mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

    // these values can also be set from somewhere if need be
    // but are static for now
    cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    cv::Mat temp;
    temp = resized_image_floats - mean;
    normalized_image = temp / theta;
    return normalized_image;
}

cv::Mat preprocess_image(cv::Mat frame){
    cv::Size size = cv::Size(112, 112);
    //cv::Mat resized_image;
    cv::Mat rgb;
    // bgr to rgb
    cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
    //cv::resize(rgb, resized_image, size);
    cv::Mat resized_image_floats(size, CV_32FC3);
    // convert resized image to floats and normalize
    rgb.convertTo(resized_image_floats, CV_32FC3);
    return resized_image_floats;
}

class MTCNN;

/**
 * Class of TVM model implementation, it contains the model definition module and the inference function.
 * the inference function is the forward
 */
class FR_MFN_Deploy {
private:
    std::unique_ptr<tvm::runtime::Module> handle;

public:
    FR_MFN_Deploy(std::string modelFolder) {
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(
                modelFolder + format("/mnet.facerec.aarch64.cpu.so"));
        //load graph
        std::ifstream json_in(modelFolder + format("/mnet.facerec.aarch64.cpu.json"));
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        int device_type = kDLCPU;
        int device_id = 0;
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                              device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        //load param
        std::ifstream params_in(modelFolder + format("/mnet.facerec.aarch64.cpu.params"), std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }

    cv::Mat forward(cv::Mat processed_image) { //inputImageAligned
        //mobilefacnet preprocess has been written in graph.
        //   cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned, 1.0, cv::Size(112, 112), cv::Scalar(0, 0, 0), true);
        //convert uint8 to float32 and convert to RGB via opencv dnn function


        DLTensor *input;
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;
        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLCPU;
        constexpr int device_id = 0;
        constexpr int in_ndim = 4;
        const int64_t in_shape[in_ndim] = {1, 3, 112, 112};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
        

        // way to not have to use cv::dnn
        int total_input = 3 * 112 * 112;
        float *data_x = (float *) malloc(total_input * sizeof(float));
        cv::Mat split_mat[3];
        cv::split(processed_image, split_mat);
        memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
        memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
               processed_image.cols * processed_image.rows * sizeof(float));
        memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
               processed_image.cols * processed_image.rows * sizeof(float));
        TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
        // TVMArrayCopyFromBytes(input, tensor.data, 112 * 3 * 112 * 4);

        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", input);
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        tvm::runtime::NDArray res = get_output(0);
        cv::Mat vector(128, 1, CV_32F);
        memcpy(vector.data, res->data, 128 * 4);
        cv::Mat _l2;
        cv::multiply(vector, vector, _l2);
        float l2 = cv::sqrt(cv::sum(_l2).val[0]);
        vector = vector / l2;
        TVMArrayFree(input);
        return vector;
    }

};

class FaceEmbedderGPU {
private:
    std::unique_ptr<tvm::runtime::Module> handle;

public:
    FaceEmbedderGPU(std::string modelFolder) {
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(
                modelFolder + format("/mnet.facerec.aarch64.cuda.so"));
        //load graph
        std::ifstream json_in(modelFolder + format("/mnet.facerec.aarch64.cuda.json"));
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        int device_type = kDLGPU;
        int device_id = 0;
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                              device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        //load param
        std::ifstream params_in(modelFolder + format("/mnet.facerec.aarch64.cuda.params"), std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }

    cv::Mat forward(cv::Mat processed_image) { //inputImageAligned
        //mobilefacnet preprocess has been written in graph.
        //   cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned, 1.0, cv::Size(112, 112), cv::Scalar(0, 0, 0), true);
        //convert uint8 to float32 and convert to RGB via opencv dnn function

        DLTensor *output;
        DLTensor *input;
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;
        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLGPU;
        constexpr int device_id = 0;
        constexpr int in_ndim = 4;
        const int64_t in_shape[in_ndim] = {1, 3, 112, 112};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
        

        // way to not have to use cv::dnn
        int total_input = 3 * 112 * 112;
        float *data_x = (float *) malloc(total_input * sizeof(float));
        cv::Mat split_mat[3];
        cv::split(processed_image, split_mat);
        memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
        memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
               processed_image.cols * processed_image.rows * sizeof(float));
        memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
               processed_image.cols * processed_image.rows * sizeof(float));
        TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
        // TVMArrayCopyFromBytes(input, tensor.data, 112 * 3 * 112 * 4);

        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", input);
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        
        // get output
        //tvm::runtime::NDArray res = get_output(0);
        get_output(0, output);
        // create cv output
        cv::Mat vector(128, 1, CV_32F);

        // copy to data
        TVMArrayCopyToBytes(output, vector.data,  128 * sizeof(float));
        //TVMArrayCopyToBytes(res->data, vector.data,  128 * sizeof(float));
        //memcpy(vector.data, res->data, 128 * 4);
        cv::Mat _l2;
        cv::multiply(vector, vector, _l2);
        float l2 = cv::sqrt(cv::sum(_l2).val[0]);
        vector = vector / l2;
        TVMArrayFree(input);
        return vector;
    }
};

class FaceEmbedderGPUFromConfig {
private:
    std::unique_ptr<tvm::runtime::Module> handle;

public:
    // confident in these
    // set these as the default
    std::string deploy_lib_path;
    std::string deploy_graph_path;
    std::string deploy_param_path;
    bool gpu = true;
    int device_id;// = 0;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLGPU;
    // set default here???
    int width;// = 512;
    int height;// = 512;
    //int64_t in_shape;// = {1, 3, height, width};
    //int64_t tvm_id_and_score_size[3] = {1, 100, 1};
    //int64_t tvm_box_size[3];// = {1, 100, 4};
    int total_input;// = 3 * width * height;
    int in_ndim = 4;
    int out_ndim = 2;
    int embedding_dim = 128;
    int max_yolo_boxes = 100;
    int64_t tvm_id_and_score_size[3] = {1, 100, 1};
    int64_t tvm_box_size[3] = {1, 100, 4};

    FaceEmbedderGPUFromConfig(std::string config_path, std::string ros_package_string) {
        // read config with nlohmann-json
        std::cout << "start model_config reading" << std::endl;
        json model_config;
        std::string package_path = ros::package::getPath(ros_package_string);
        std::string full_path = package_path + config_path;
        std::ifstream json_read(full_path);
        json_read >> model_config;
        std::cout << "model_config read into memory" << std::endl;
        // read variables
        std::string lib_path = model_config["deploy_lib_path"];
        std::string graph_path = model_config["deploy_graph_path"];
        std::string param_path = model_config["deploy_param_path"];

        deploy_lib_path = package_path + lib_path;
        deploy_graph_path = package_path + graph_path;
        deploy_param_path = package_path + param_path;

        device_id = model_config["device_id"];
        width = model_config["width"];
        height = model_config["height"];
        gpu = model_config["gpu"];
        total_input = 3 * width * height;
        embedding_dim = model_config['embedding_dim'];

        //int64_t in_shape[4] = {1, 3, height, width};
        //int64_t in_shape[4] = {1, 3, height, width};
        // set device type -- I think this has to be set here...
        if (gpu){
            device_type = kDLGPU;
        } else {
            device_type = kDLCPU;
        }

        // read deploy lib
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(deploy_lib_path);
        // read deplpy json
        std::ifstream json_in(deploy_graph_path, std::ios::in);
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        // parameters in binary
        std::ifstream params_in(deploy_param_path, std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        // parameters need to be TVMByteArray type to indicate the binary data
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }

    cv::Mat forward(cv::Mat processed_image) { //inputImageAligned
        //mobilefacnet preprocess has been written in graph.
        //   cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned, 1.0, cv::Size(112, 112), cv::Scalar(0, 0, 0), true);
        //convert uint8 to float32 and convert to RGB via opencv dnn function

        DLTensor *output;
        DLTensor *input;
        //constexpr int dtype_code = kDLFloat;
        //constexpr int dtype_bits = 32;
        //constexpr int dtype_lanes = 1;
        //constexpr int device_type = kDLGPU;
        //constexpr int device_id = 0;
        //constexpr int in_ndim = 4;
        //const int64_t in_shape[in_ndim] = {1, 3, 112, 112};
        int64_t in_shape[4] = {1, 3, height, width};
        int64_t out_shape[2] = {1, embedding_dim};
        total_input = 3 * width * height;
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
        // is this necessary
        TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output);

        // way to not have to use cv::dnn
        //int total_input = 3 * 112 * 112;
        float *data_x = (float *) malloc(total_input * sizeof(float));
        cv::Mat split_mat[3];
        cv::split(processed_image, split_mat);
        memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
        memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
               processed_image.cols * processed_image.rows * sizeof(float));
        memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
               processed_image.cols * processed_image.rows * sizeof(float));
        TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));
        // TVMArrayCopyFromBytes(input, tensor.data, 112 * 3 * 112 * 4);

        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", input);
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        
        // get output
        //tvm::runtime::NDArray res = get_output(0);
        get_output(0, output);
        // create cv output
        cv::Mat vector(embedding_dim, 1, CV_32F);

        // copy to data
        TVMArrayCopyToBytes(output, vector.data,  embedding_dim * sizeof(float));
        //TVMArrayCopyToBytes(res->data, vector.data,  128 * sizeof(float));
        //memcpy(vector.data, res->data, 128 * 4);
        cv::Mat _l2;
        cv::multiply(vector, vector, _l2);
        float l2 = cv::sqrt(cv::sum(_l2).val[0]);
        vector = vector / l2;
        TVMArrayFree(input);
        return vector;
    }

};


class RetinaFaceDeploy {
private:
    std::unique_ptr<tvm::runtime::Module> handle;

public:
    RetinaFaceDeploy(std::string modelFolder) {
        // tvm module for compiled functions
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(modelFolder + "/mnet.25.aarch64.cpu.so");
        // json graph
        std::ifstream json_in(modelFolder + "/mnet.25.aarch64.cpu.json", std::ios::in);
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();

        int dtype_code = kDLFloat;
        int dtype_bits = 32;

        int dtype_lanes = 1;
        int device_type = kDLCPU;//kDLGPU
        int device_id = 0;
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                              device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        // parameters in binary
        std::ifstream params_in(modelFolder + "/mnet.25.aarch64.cpu.params", std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        // parameters need to be TVMByteArray type to indicate the binary data
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }

    RetinaOutput forward(cv::Mat image) {
        constexpr int dtype_code = kDLFloat;
        constexpr int dtype_bits = 32;

        constexpr int dtype_lanes = 1;
        constexpr int device_type = kDLCPU;//kDLGPU
        constexpr int device_id = 0;
        DLTensor *x;
        int in_ndim = 4;
        int in_c = 3, in_h = 480, in_w = 640;
        int ratio_x = 1, ratio_y = 1;
        int64_t in_shape[4] = {1, in_c, in_h, in_w};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

        int64_t w1 = ceil(in_w / 32.0), w2 = ceil(in_w / 16.0), w3 = ceil(in_w / 8.0), h1 = ceil(
                in_h / 32.0), h2 = ceil(in_h / 16.0), h3 = ceil(in_h / 8.0);
        int out_num = (w1 * h1 + w2 * h2 + w3 * h3) * (4 + 8 + 20);

        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();

        int total_input = 3 * in_w * in_h;
        float *data_x = (float *) malloc(total_input * sizeof(float));

        //float* y_iter = (float*)malloc(out_num*4);

        if (!image.data)
            printf("load error");

        //input data
        cv::Mat resizeImage;
        cv::resize(image, resizeImage, cv::Size(in_w, in_h), cv::INTER_AREA);
        cv::Mat input_mat;

        resizeImage.convertTo(input_mat, CV_32FC3);
        //cv::cvtColor(input_mat, input_mat, cv::COLOR_BGR2RGB);
        cv::Mat split_mat[3];
        cv::split(input_mat, split_mat);
        memcpy(data_x, split_mat[2].ptr<float>(), input_mat.cols * input_mat.rows * sizeof(float));
        memcpy(data_x + input_mat.cols * input_mat.rows, split_mat[1].ptr<float>(),
               input_mat.cols * input_mat.rows * sizeof(float));
        memcpy(data_x + input_mat.cols * input_mat.rows * 2, split_mat[0].ptr<float>(),
               input_mat.cols * input_mat.rows * sizeof(float));
        TVMArrayCopyFromBytes(x, data_x, total_input * sizeof(float));

        // get the function from the module(set input data)
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", x);
        // get the function from the module(run it)
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            int stride = _feat_stride_fpn[i];
            ac[i].Init(stride, anchor_cfg[stride], false);
        }
        std::vector<Anchor> proposals;
        proposals.clear();

        int64_t w[3] = {w1, w2, w3};
        int64_t h[3] = {h1, h2, h3};
        int64_t out_size[9] = {w1 * h1 * 4, w1 * h1 * 8, w1 * h1 * 20, w2 * h2 * 4, w2 * h2 * 8, w2 * h2 * 20,
                               w3 * h3 * 4, w3 * h3 * 8, w3 * h3 * 20};

        int out_ndim = 4;
        int64_t out_shape[9][4] = {{1, 4,  h1, w1},
                                   {1, 8,  h1, w1},
                                   {1, 20, h1, w1},
                                   {1, 4,  h2, w2},
                                   {1, 8,  h2, w2},
                                   {1, 20, h2, w2},
                                   {1, 4,  h3, w3},
                                   {1, 8,  h3, w3},
                                   {1, 20, h3, w3}};
        DLTensor *y[9];
        for (int i = 0; i < 9; i++)
            TVMArrayAlloc(out_shape[i], out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y[i]);
        for (int i = 0; i < 9; i += 3) {
            get_output(i, y[i]);
            get_output(i + 1, y[i + 1]);
            get_output(i + 2, y[i + 2]);

            ulsMatF clsMat(w[i / 3], h[i / 3], 4);
            ulsMatF regMat(w[i / 3], h[i / 3], 8);
            ulsMatF ptsMat(w[i / 3], h[i / 3], 20);


            TVMArrayCopyToBytes(y[i], clsMat.m_data, out_size[i] * sizeof(float));
            TVMArrayCopyToBytes(y[i + 1], regMat.m_data, out_size[i + 1] * sizeof(float));
            TVMArrayCopyToBytes(y[i + 2], ptsMat.m_data, out_size[i + 2] * sizeof(float));


            ac[i / 3].FilterAnchor(clsMat, regMat, ptsMat, proposals);
//            std::cout << "proposals:" << proposals.size() << std::endl;

        }

        // nms
        std::vector<Anchor> result;
        nms_cpu(proposals, nms_threshold, result);
//        printf("final proposals: %ld\n", result.size());

        // free buffer
        free(data_x);
        data_x = nullptr;
        TVMArrayFree(x);
        for (int i = 0; i < 9; i++)
            TVMArrayFree(y[i]);

        RetinaOutput output_;
        output_.result = result;
        output_.ratio.x = ratio_x;
        output_.ratio.y = ratio_y;
        return output_;
    }
};

class RetinaFaceDeployFromConfig {
private:
    std::unique_ptr<tvm::runtime::Module> handle;

public:
    // confident in these
    // set these as the default
    std::string deploy_lib_path;
    std::string deploy_graph_path;
    std::string deploy_param_path;
    bool gpu = true;
    int device_id;// = 0;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLGPU;
    // set default here???
    int width;// = 512;
    int height;// = 512;
    //int64_t in_shape;// = {1, 3, height, width};
    //int64_t tvm_id_and_score_size[3] = {1, 100, 1};
    //int64_t tvm_box_size[3];// = {1, 100, 4};
    int total_input;// = 3 * width * height;
    int in_ndim = 4;
    int out_ndim = 3;
    int max_yolo_boxes = 100;
    int64_t tvm_id_and_score_size[3] = {1, 100, 1};
    int64_t tvm_box_size[3] = {1, 100, 4};

    RetinaFaceDeployFromConfig(std::string config_path, std::string ros_package_string) {
        // read config with nlohmann-json
        std::cout << "start model_config reading" << std::endl;
        json model_config;
        std::string package_path = ros::package::getPath(ros_package_string);
        std::string full_path = package_path + config_path;
        std::ifstream json_read(full_path);
        json_read >> model_config;
        std::cout << "model_config read into memory" << std::endl;
        // read variables
        std::string lib_path = model_config["deploy_lib_path"];
        std::string graph_path = model_config["deploy_graph_path"];
        std::string param_path = model_config["deploy_param_path"];

        deploy_lib_path = package_path + lib_path;
        deploy_graph_path = package_path + graph_path;
        deploy_param_path = package_path + param_path;

        device_id = model_config["device_id"];
        width = model_config["width"];
        height = model_config["height"];
        gpu = model_config["gpu"];
        total_input = 3 * width * height;

        //int64_t in_shape[4] = {1, 3, height, width};
        //int64_t in_shape[4] = {1, 3, height, width};
        // set device type -- I think this has to be set here...
        if (gpu){
            device_type = kDLGPU;
        } else {
            device_type = kDLCPU;
        }

        // read deploy lib
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(deploy_lib_path);
        // read deplpy json
        std::ifstream json_in(deploy_graph_path, std::ios::in);
        std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
        json_in.close();
        // get global function module for graph runtime
        tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                device_type, device_id);
        this->handle.reset(new tvm::runtime::Module(mod));
        // parameters in binary
        std::ifstream params_in(deploy_param_path, std::ios::binary);
        std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
        params_in.close();
        // parameters need to be TVMByteArray type to indicate the binary data
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
    }
    cv::Mat preprocess_image(cv::Mat frame){
        cv::Size new_size = cv::Size(width, height); // or is it height width????
        cv::Mat resized_image;
        cv::Mat rgb;
        // bgr to rgb
        cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
        // resize to 512x512
        cv::resize(rgb, resized_image, new_size);
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

    RetinaOutput forward(cv::Mat image) {
        // maybe it's faster this way????
        //constexpr int dtype_code = kDLFloat;
        //constexpr int dtype_bits = 32;
        //constexpr int dtype_lanes = 1;
        //constexpr int device_type = kDLCPU;//kDLGPU
        //constexpr int device_id = 0;
        
        DLTensor *x;
        int in_ndim = 4;
        int in_c = 3, in_h = 480, in_w = 640;
        int ratio_x = 1, ratio_y = 1;
        int64_t in_shape[4] = {1, in_c, in_h, in_w};
        TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

        int64_t w1 = ceil(in_w / 32.0), w2 = ceil(in_w / 16.0), w3 = ceil(in_w / 8.0), h1 = ceil(
                in_h / 32.0), h2 = ceil(in_h / 16.0), h3 = ceil(in_h / 8.0);
        int out_num = (w1 * h1 + w2 * h2 + w3 * h3) * (4 + 8 + 20);

        tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();

        int total_input = 3 * in_w * in_h;
        float *data_x = (float *) malloc(total_input * sizeof(float));

        //float* y_iter = (float*)malloc(out_num*4);

        if (!image.data)
            printf("load error");

        //input data
        cv::Mat resizeImage;
        cv::resize(image, resizeImage, cv::Size(in_w, in_h), cv::INTER_AREA);
        cv::Mat input_mat;

        resizeImage.convertTo(input_mat, CV_32FC3);
        //cv::cvtColor(input_mat, input_mat, cv::COLOR_BGR2RGB);
        cv::Mat split_mat[3];
        cv::split(input_mat, split_mat);
        memcpy(data_x, split_mat[2].ptr<float>(), input_mat.cols * input_mat.rows * sizeof(float));
        memcpy(data_x + input_mat.cols * input_mat.rows, split_mat[1].ptr<float>(),
               input_mat.cols * input_mat.rows * sizeof(float));
        memcpy(data_x + input_mat.cols * input_mat.rows * 2, split_mat[0].ptr<float>(),
               input_mat.cols * input_mat.rows * sizeof(float));
        TVMArrayCopyFromBytes(x, data_x, total_input * sizeof(float));

        // get the function from the module(set input data)
        tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
        set_input("data", x);
        // get the function from the module(run it)
        tvm::runtime::PackedFunc run = mod->GetFunction("run");
        run();
        tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
        std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            int stride = _feat_stride_fpn[i];
            ac[i].Init(stride, anchor_cfg[stride], false);
        }
        std::vector<Anchor> proposals;
        proposals.clear();

        int64_t w[3] = {w1, w2, w3};
        int64_t h[3] = {h1, h2, h3};
        int64_t out_size[9] = {w1 * h1 * 4, w1 * h1 * 8, w1 * h1 * 20, w2 * h2 * 4, w2 * h2 * 8, w2 * h2 * 20,
                               w3 * h3 * 4, w3 * h3 * 8, w3 * h3 * 20};

        int out_ndim = 4;
        int64_t out_shape[9][4] = {{1, 4,  h1, w1},
                                   {1, 8,  h1, w1},
                                   {1, 20, h1, w1},
                                   {1, 4,  h2, w2},
                                   {1, 8,  h2, w2},
                                   {1, 20, h2, w2},
                                   {1, 4,  h3, w3},
                                   {1, 8,  h3, w3},
                                   {1, 20, h3, w3}};
        DLTensor *y[9];
        for (int i = 0; i < 9; i++)
            TVMArrayAlloc(out_shape[i], out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y[i]);
        
        for (int i = 0; i < 9; i += 3) {
            get_output(i, y[i]);
            get_output(i + 1, y[i + 1]);
            get_output(i + 2, y[i + 2]);

            ulsMatF clsMat(w[i / 3], h[i / 3], 4);
            ulsMatF regMat(w[i / 3], h[i / 3], 8);
            ulsMatF ptsMat(w[i / 3], h[i / 3], 20);


            TVMArrayCopyToBytes(y[i], clsMat.m_data, out_size[i] * sizeof(float));
            TVMArrayCopyToBytes(y[i + 1], regMat.m_data, out_size[i + 1] * sizeof(float));
            TVMArrayCopyToBytes(y[i + 2], ptsMat.m_data, out_size[i + 2] * sizeof(float));


            ac[i / 3].FilterAnchor(clsMat, regMat, ptsMat, proposals);
//            std::cout << "proposals:" << proposals.size() << std::endl;

        }

        // nms
        std::vector<Anchor> result;
        nms_cpu(proposals, nms_threshold, result);
//        printf("final proposals: %ld\n", result.size());

        // free buffer
        free(data_x);
        data_x = nullptr;
        TVMArrayFree(x);
        for (int i = 0; i < 9; i++)
            TVMArrayFree(y[i]);

        RetinaOutput output_;
        output_.result = result;
        output_.ratio.x = ratio_x;
        output_.ratio.y = ratio_y;
        return output_;
    }
};

using namespace cv;
int MTCNNTracking(MTCNN &detector, FR_MFN_Deploy &deploy);
int RetinaFaceTracking(RetinaFaceDeploy &deploy_track, FR_MFN_Deploy &deploy_rec);
