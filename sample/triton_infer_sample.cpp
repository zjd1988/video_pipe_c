#include <memory>
#include "VP.h"
#include "opencv2/opencv.hpp"
#include "../utils/logger/vp_logger.h"
#include "../utils/vp_tritonserver.h"

/*
* ## triton infer sample ##
* test TrtitonServer infer api
*/

#if triton_infer_sample

int main()
{
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    VP_INFO("init triton server infer");
    TRITON_SERVER_INIT("/video_pipe_c/models_test", 0);

    uint32_t test_count = 1000000;
    std::string model_name = "test";
    std::string model_version = "1";
    std::vector<cv::Mat> input_mats;
    std::vector<cv::Mat> output_mats;

    // read test image
    VP_INFO("preprocess input image");
    cv::Mat test_mat = cv::imread("/video_pipe_c/test_image/test.jpg");
    cv::Mat blob_mat;
    cv::Scalar mean = {0, 0, 0};
    cv::dnn::blobFromImage(test_mat, blob_mat, 1.0, cv::Size(640, 640), mean);
    blob_mat = blob_mat / 255.0f;
    input_mats.push_back(blob_mat);

    // std::string model_name = "simple";
    // std::string model_version = "1";
    // std::vector<cv::Mat> input_mats;
    // std::vector<cv::Mat> output_mats;

    // // read test image
    // VP_INFO("preprocess input image");
    // cv::Mat input_mat1 = cv::Mat(1, 16, CV_32SC1);
    // cv::Mat input_mat2 = cv::Mat(1, 16, CV_32SC1);

    // input_mats.push_back(input_mat1);
    // input_mats.push_back(input_mat2);

    // model infer
    while (test_count-- > 0)
    {
        VP_INFO("triton server infer model");
        TRITON_SERVER_INFER(model_name, model_version, input_mats, output_mats);
    }

    // triton server uninit
    VP_INFO("uninit triton server infer");
    TRITON_SERVER_UNINIT();
}

#endif