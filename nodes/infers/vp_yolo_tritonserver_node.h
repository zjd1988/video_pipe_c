
#pragma once

#include "../vp_primary_infer_node.h"

namespace vp_nodes {
    // yolo tritonserver detector
    class vp_yolo_tritonserver_node: public vp_primary_infer_node
    {
    private:
        float score_threshold;
        float confidence_threshold;
        float nms_threshold;
    protected:
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer);
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
    public:
        vp_yolo_tritonserver_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_name = "", 
                            int         model_version = -1, 
                            std::string model_config_path = "", 
                            std::string labels_path = "", 
                            int input_width = 640, 
                            int input_height = 640, 
                            int batch_size = 1,
                            int class_id_offset = 0,
                            float score_threshold = 0.5,
                            float confidence_threshold = 0.5,
                            float nms_threshold = 0.5,
                            float scale = 1 / 255.0,
                            cv::Scalar mean = cv::Scalar(0),
                            cv::Scalar std = cv::Scalar(1),
                            bool swap_rb = true);
        ~vp_yolo_tritonserver_node();
    };
}