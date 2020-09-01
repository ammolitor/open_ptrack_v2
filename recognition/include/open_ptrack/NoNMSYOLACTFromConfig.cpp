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
#include <open_ptrack/nms_utils/nms.h>
#include <numeric> 
//#include "tvm_detection_helpers.hpp"
#include <torch/torch.h>
namespace F = torch::nn::functional;
using json = nlohmann::json;

//#ifndef OPEN_PTRACK_MODELS_BASED_SUBCLUSTER_H_
//#define OPEN_PTRACK_MODELS_BASED_SUBCLUSTER_H_

  

int arrayProduct(int a[], int n) 
{ 
    return std::accumulate(a, a + n, 1, multiplies<int>()); 
} 

namespace open_ptrack
{
  namespace models
  {

    std::vector<std::string> COCO_CLASS_NAMES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // box
    struct box{
        float xmin;
        float ymin;
        float xmax;
        float ymax;
    };

    // adjBox
    struct yolact_result{
        int id;
        float score;
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        open_ptrack::nms_utils::MatF mask;
        cv::Mat mask_img;

    };
    // boxInfo
    struct yolactresults{
        yolact_result* boxes;
        int num;
    };

    struct NMSPack{
        int max_num_detections;
        int top_k;
        float iou_threshold;
        float score_threshold;
    };

    struct ResultsPack{
        int img_height;
        int img_width;
        int n_dets;
        torch::Tensor classes;
        torch::Tensor scores;
        torch::Tensor boxes;
        torch::Tensor masks;
        bool has_results = false;

        torch::Tensor conf_preds; //[1, 80, 19248]
        torch::Tensor decoded_boxes; // [19248, 4]
        torch::Tensor mask; //[19248, 32]
    };

    torch::Tensor intersect(torch::Tensor box_a, torch::Tensor box_b){
        int n = box_a.size(0);
        int A = box_a.size(1);
        int B = box_b.size(1);

        auto box_amax_expand = box_a.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)}).unsqueeze(2).expand({n, A, B, 2});
        auto box_bmax_expand = box_b.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)}).unsqueeze(1).expand({n, A, B, 2});
        auto max_xy = torch::min(box_amax_expand, box_bmax_expand);

        auto box_amin_expand = box_a.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).unsqueeze(2).expand({n, A, B, 2});
        auto box_bmin_expand = box_b.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).unsqueeze(1).expand({n, A, B, 2});
        auto min_xy = torch::max(box_amin_expand, box_bmin_expand);
        auto sub = max_xy - min_xy;
        return torch::clamp_min(max_xy - min_xy, 0).prod(3);
    }

    torch::Tensor jaccard(torch::Tensor box_a, torch::Tensor box_b){
        torch::Tensor inter = intersect(box_a, box_b);
        auto box_a2 = box_a.index({torch::indexing::Slice(), torch::indexing::Slice(), 2});
        auto box_a0 = box_a.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
        auto box_a3 = box_a.index({torch::indexing::Slice(), torch::indexing::Slice(), 3});
        auto box_a1 = box_a.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
        
        auto box_b2 = box_b.index({torch::indexing::Slice(), torch::indexing::Slice(), 2});
        auto box_b0 = box_b.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
        auto box_b3 = box_b.index({torch::indexing::Slice(), torch::indexing::Slice(), 3});
        auto box_b1 = box_b.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});

        auto box_ax = box_a2 - box_a0;
        auto box_ay = box_a3 - box_a1;
        auto box_a_m = box_ax * box_ay;
        auto area_a = box_a_m.unsqueeze(2).expand_as(inter);

        auto box_bx = box_b2 - box_b0;
        auto box_by = box_b3 - box_b1;
        auto box_b_m = box_bx * box_by;
        auto area_b = box_b_m.unsqueeze(1).expand_as(inter);
        auto union_ = area_a + area_b - inter;
        auto iou = inter / union_;
        return iou;
    }


    std::tuple<torch::Tensor, torch::Tensor> sanitize_coordinates(torch::Tensor b1, torch::Tensor b2, int img_size, int padding, bool cast){
        auto _x1 = b1 * img_size;
        auto _x2 = b2 * img_size;
        auto x1 = torch::min(_x1, _x2);
        auto x2 = torch::max(_x1, _x2);
        x1 = at::clamp_min(x1 - padding, 0);
        x2 = at::clamp_max(x2 + padding, img_size);
        return std::make_tuple(x1, x2);
    }

    torch::Tensor crop(torch::Tensor masks, torch::Tensor boxes, int padding, torch::Device device){
        int h = masks.size(0);
        int w = masks.size(1);
        int n = masks.size(2);
        auto boxes0 = boxes.index({torch::indexing::Slice(), 0});
        auto boxes2 = boxes.index({torch::indexing::Slice(), 2});
        auto boxes1 = boxes.index({torch::indexing::Slice(), 1});
        auto boxes3 = boxes.index({torch::indexing::Slice(), 3});
        
        std::tuple<torch::Tensor, torch::Tensor> xs = sanitize_coordinates(boxes0, boxes2, w, padding, false);
        std::tuple<torch::Tensor, torch::Tensor> ys = sanitize_coordinates(boxes1, boxes3, h, padding, false);
        torch::Tensor x1 = std::get<0>(xs);
        torch::Tensor x2 = std::get<1>(xs);
        torch::Tensor y1 = std::get<0>(ys);
        torch::Tensor y2 = std::get<1>(ys);

        torch::Tensor rows = torch::arange(0, w).view({1, -1, 1}).expand({h, w, n}).to(device);
        torch::Tensor cols = torch::arange(0, h).view({-1, 1, 1}).expand({h, w, n}).to(device);

        auto masks_left  = rows >= x1.view({1, 1, -1});
        auto masks_right = rows <  x2.view({1, 1, -1});
        auto masks_up    = cols >= y1.view({1, 1, -1});
        auto masks_down  = cols <  y2.view({1, 1, -1});
        auto crop_mask = masks_left * masks_right * masks_up * masks_down;

        auto newmasks = masks * crop_mask.to(torch::kFloat);
        return newmasks;
    }

    void sanitize_boxes(ResultsPack *results_pack){
        //std::tuple<torch::Tensor, torch::Tensor> xs = sanitize_coordinates()
        auto boxes0 = results_pack->boxes.index({torch::indexing::Slice(), 0});
        auto boxes2 = results_pack->boxes.index({torch::indexing::Slice(), 2});
        auto boxes1 = results_pack->boxes.index({torch::indexing::Slice(), 1});
        auto boxes3 = results_pack->boxes.index({torch::indexing::Slice(), 3});
        std::tuple<torch::Tensor, torch::Tensor> xs = sanitize_coordinates(boxes0, boxes2, results_pack->img_width, 0, false);
        std::tuple<torch::Tensor, torch::Tensor> ys = sanitize_coordinates(boxes1, boxes3, results_pack->img_height, 0, false);
        torch::Tensor x1 = std::get<0>(xs);
        torch::Tensor x2 = std::get<1>(xs);
        torch::Tensor y1 = std::get<0>(ys);
        torch::Tensor y2 = std::get<1>(ys);
        results_pack->boxes = torch::stack({x1, y1, x2, y2}).t();
    }



    int fast_nms(ResultsPack *results_pack, NMSPack *nms_pack){
        auto conf_scores_tuple = torch::max(results_pack->conf_preds.squeeze(0), 0);
        auto conf_scores = std::get<0>(conf_scores_tuple);
        int batch_idx = 0;

        auto keep = conf_scores.gt(0.05);
        results_pack->conf_preds = results_pack->conf_preds.index({torch::indexing::Slice(), torch::indexing::Slice(), keep});
        results_pack->decoded_boxes = results_pack->decoded_boxes.index({keep});
        results_pack->masks = results_pack->masks.index({keep, torch::indexing::None});

        auto scores_indices = torch::sort(results_pack->conf_preds, 2, true);
        results_pack->conf_preds = std::get<0>(scores_indices);
        auto indices = std::get<1>(scores_indices);

        indices = indices.reshape({indices.size(1), indices.size(2)});
        auto idx = indices.index({torch::indexing::Slice(), torch::indexing::Slice(at::indexing::None, nms_pack->top_k)}).contiguous();
        results_pack->conf_preds = results_pack->conf_preds.reshape({results_pack->conf_preds.size(1), results_pack->conf_preds.size(2)});
        results_pack->conf_preds = results_pack->conf_preds.index({torch::indexing::Slice(), torch::indexing::Slice(at::indexing::None, nms_pack->top_k)});

        int num_classes = idx.size(0);
        int num_dets = idx.size(1);
        if (num_dets == 0){
            results_pack->n_dets = 0;
            return 0;
        }

        results_pack->decoded_boxes = torch::index_select(results_pack->decoded_boxes, 0, idx.view({-1})).view({num_classes, num_dets, 4});
        results_pack->masks = torch::index_select(results_pack->masks, 0, idx.view({-1})).view({num_classes, num_dets, -1});
        torch::Tensor iou = jaccard(results_pack->decoded_boxes, results_pack->decoded_boxes);
        iou = iou.triu_(1);
        std::tuple<torch::Tensor, torch::Tensor> iou_maxes = torch::max(iou, 1);
        auto iou_max = std::get<0>(iou_maxes);
        keep = iou_max.le(nms_pack->iou_threshold);

        torch::Tensor classes = torch::arange(num_classes).index({torch::indexing::Slice(), torch::indexing::None}).expand_as(keep);
        classes = classes.index({keep});
        results_pack->decoded_boxes = results_pack->decoded_boxes.index({keep});

        results_pack->masks = results_pack->masks.index({keep});
        results_pack->conf_preds = results_pack->conf_preds.index({keep});
    
        scores_indices = torch::sort(results_pack->conf_preds, 0, true);
        results_pack->conf_preds = std::get<0>(scores_indices);
        indices = std::get<1>(scores_indices);
    
        idx = indices.index({torch::indexing::Slice(torch::indexing::None, nms_pack->max_num_detections)});
        results_pack->conf_preds = results_pack->conf_preds.index({torch::indexing::Slice(torch::indexing::None, nms_pack->max_num_detections)});
        classes = classes.index({idx});
        results_pack->decoded_boxes = results_pack->decoded_boxes.index({idx});
        // working directly on the masks
        results_pack->masks = results_pack->masks.index({idx});
        results_pack->scores = results_pack->conf_preds;
        results_pack->boxes = results_pack->decoded_boxes;
        results_pack->n_dets = results_pack->decoded_boxes.size(0);
        results_pack->classes = classes;
        results_pack->has_results = true;
        std::cout << "amount: " << amount << std::endl;
        return 1;
    }


    void process_masks(ResultsPack *results_pack, torch::Tensor proto_mat, torch::Device device){
        results_pack->masks = at::matmul(proto_mat, results_pack->masks.transpose(1, 0));          
        results_pack->masks = torch::sigmoid(results_pack->masks).squeeze(0);
        results_pack->masks = crop(results_pack->masks, results_pack->boxes, 1, device);
        results_pack->masks = results_pack->masks.permute({2, 0, 1}).contiguous();
        results_pack->masks = torch::nn::functional::interpolate(results_pack->masks.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({results_pack->img_height, results_pack->img_width})).mode(torch::kBilinear)).squeeze(0);//align_corners(False)
        //binarize
        results_pack->masks.gt_(0.5);
    }

class YolactPytorch{
    private:
        //torch::jit::script::Module module;
        torch::jit::script::Module module;

    public:
        bool gpu = true;
        int detector_width;
        int detector_height;
        int detector_total_input;
        torch::Device device = torch::kCUDA;

        /**
         * function that reads both the yolo detector and the pose detector
         * 
        */
        YolactPytorch(std::string config_path) {
            detector_width = 550;
            detector_height = 550;
            gpu = true;
            json model_config;
            std::ifstream json_read(config_path);
            json_read >> model_config;
            std::string model_path = model_config["model_path"];
            detector_width = model_config["detector_width"];
            detector_height = model_config["detector_height"];
            // this model will only run on gpu...
            //gpu = model_config["gpu"]

            try {
                // Deserialize the ScriptModule from a file using torch::jit::load().
                module = torch::jit::load(model_path);
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading the model\n";
                //return -1;
            }

            module.to(device);
            module.eval();
        }
    
        /**
         * \brief function to normalize an image before it's processed by the network
         * \param[in] the raw cv::mat image
         * \return the normalized version of the iamge.
         */  

        torch::Tensor preprocess(cv::Mat frame){
            // switch method
            cv::Size new_size = cv::Size(detector_width, detector_height); // or is it height width????
            cv::Mat resized_image;
            cv::Mat processed_image;
            cv::resize(frame, resized_image, new_size, cv::INTER_LINEAR);
            resized_image.convertTo(resized_image, CV_32FC3);
            //cv::Mat mean(new_size, CV_32FC3, cv::Scalar(103.94, 116.78, 123.68));
            //cv::Mat std(new_size, CV_32FC3, cv::Scalar(57.38, 57.12, 58.40));
            cv::Mat mean(new_size, CV_32FC3, cv::Scalar(123.68, 116.78, 103.94));
            cv::Mat std(new_size, CV_32FC3, cv::Scalar( 58.40, 57.12,57.38));
            resized_image -= mean;
            resized_image /= std;
            cv::cvtColor(resized_image, processed_image,  cv::COLOR_BGR2RGB);
            cv::Mat channelsConcatenatedFloat;
            cv::Mat rgb[3];
            cv::split(processed_image, rgb);
            cv::Mat channelsConcatenated;
            cv::vconcat(rgb[2], rgb[1], channelsConcatenated);
            cv::vconcat(channelsConcatenated, rgb[0], channelsConcatenated);

            channelsConcatenated.convertTo(channelsConcatenatedFloat, CV_32FC3);
            assert(channelsConcatenatedFloat.isContinuous());
            std::vector<int64_t> dims{static_cast<int64_t>(processed_image.channels()),
                                        static_cast<int64_t>(processed_image.rows),
                                        static_cast<int64_t>(processed_image.cols)};
            at::TensorOptions options(at::kFloat);
            at::Tensor frame_tensor =
                torch::from_blob(channelsConcatenatedFloat.data, at::IntList(dims),
                                options.requires_grad(false))
                    .clone();  // clone is required to copy data from temporary object
            //frame_tensor.squeeze();
            return frame_tensor;
        }


        // we can set it externally with dynamic reconfigure
        //yolactresults* forward_full(cv::Mat frame, float override_threshold, float nms_threshold)
        void forward_ptr(cv::Mat frame, ResultsPack *results_pack)
        {

            cv::Size image_size = frame.size();
            int img_height = image_size.height;
            int img_width = image_size.width;

            torch::Tensor frame_tensor = preprocess(frame);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(frame_tensor.unsqueeze(0).cuda());

            // forward process model
            //DataPack data_pack;
            NMSPack nms_pack;
            //  ResultsPack results_pack;
            nms_pack.iou_threshold = 0.5;
            nms_pack.top_k = 200;
            nms_pack.max_num_detections = 100;
            results_pack->img_height = img_height;
            results_pack->img_width = img_width;
            using Clock = std::chrono::high_resolution_clock;
            using Timepoint = Clock::time_point;
            using Duration = std::chrono::duration<double>;
            auto start = Clock::now();
            auto output = module.forward(inputs); 
            auto end = Clock::now();
            auto elapsed = Duration(end - start).count();
            std::cout << "auto output = module.forward(inputs); time elapsed: " << elapsed << std::endl;

            results_pack->conf_preds = output.toTuple()->elements()[0].toTensor();// == cur_scores = conf_preds[batch_idx, 1:, :]
            results_pack->decoded_boxes = output.toTuple()->elements()[1].toTensor();
            //std::cout << decoded_boxes.toTensor().index({0}) << std::endl;
            results_pack->masks = output.toTuple()->elements()[2].toTensor();
            auto proto_out = output.toTuple()->elements()[3];
            auto proto_mat = proto_out.toTensor();

            start = Clock::now();
            bool success = fast_nms(results_pack, nms_pack);
            end = Clock::now();
            elapsed = Duration(end - start).count();
            std::cout << "bool success = fast_nms_ptr(&results_pack, nms_pack);: " << elapsed << std::endl;
            std::cout << "second run results_pack.classes_[1]_size: " << results_pack->classes.size(0) << std::endl;
            if(success){
                start = Clock::now();
                process_masks(results_pack, proto_mat, device);
                end = Clock::now();
                elapsed = Duration(end - start).count();
                std::cout << "process_masks(results_pack, proto_mat, device);: " << elapsed << std::endl;

                start = Clock::now();
                sanitize_boxes(results_pack);   
                end = Clock::now();
                elapsed = Duration(end - start).count();
                std::cout << "sanitize_boxes(results_pack);: " << elapsed << std::endl;
            }

        }
    };

  } /* namespace models */
} /* namespace open_ptrack */
//#include <open_ptrack/person_clustering/height_map_2d.hpp>
