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
    TRITON_SERVER_INIT("/models", 0);

    std::string model_name = "face_detect";
    std::string model_version = "1";

    // read test image
    cv::VideoCapture capture("/video_pipe_c/test_video/test.avi");
    if (!capture.isOpened())
    {
        VP_ERROR("cannot open video file");
        return -1;
    }

    // infer every frame
    while (true)
    {
        std::vector<cv::Mat> input_mats;
        std::vector<cv::Mat> output_mats;
        cv::Mat frame;
        bool success = capture.read(frame);
        if (frame.empty() || !success)
        {
            VP_INFO("read video over");
            break;
        }
        cv::Mat input_mat;
        cv::Mat input_blob;
        frame.convertTo(input_mat, CV_32F);
        cv::dnn::blobFromImage(input_mat, input_blob);
        input_mats.push_back(input_blob);
        TRITON_SERVER_INFER(model_name, model_version, input_mats, output_mats);
    }

    // triton server uninit
    VP_INFO("uninit triton server infer");
    TRITON_SERVER_UNINIT();
}

#endif