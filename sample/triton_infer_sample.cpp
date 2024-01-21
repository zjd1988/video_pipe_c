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
    TRITON_SERVER_DEFAULT_INIT("/video_pipe_c/models_test", 0);

    uint32_t test_count = 1;
    std::string model_name = "test";   // yolov7 end2end onnx model
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

    // model infer
    while (test_count-- > 0)
    {
        VP_INFO("triton server infer model");
        TRITON_SERVER_INFER(model_name, model_version, input_mats, output_mats);
    }

    // draw detect results
    cv::Mat det = output_mats[0];
    int det_num = det.size[0];
    VP_INFO("detect " + std::to_string(det_num) + " objs .............");
    for (size_t i = 0; i < det_num; i++)
    {
        int offset = 7 * sizeof(float) * i;
        float* det_data = (float*)(det.data + offset);
        int x1 = int(det_data[1]);
        int y1 = int(det_data[2]);
        int x2 = int(det_data[3]);
        int y2 = int(det_data[4]);
        std::cout << det_data[0] << " " << det_data[1] << " " << det_data[2] << " " 
                  << det_data[3] << " " << det_data[4] << " " << det_data[5] << " " 
                  << det_data[6] << std::endl;
        cv::rectangle(test_mat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0));
    }

    cv::imwrite("/video_pipe_c/test_image/result.jpg", test_mat);

    // triton server uninit
    VP_INFO("uninit triton server infer");
    TRITON_SERVER_UNINIT();
}

#endif