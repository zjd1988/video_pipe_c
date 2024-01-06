
#include "VP.h"

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yunet_face_detector_node.h"
#include "../nodes/infers/vp_sface_feature_encoder_node.h"
#include "../nodes/osd/vp_face_osd_node_v2.h"
#include "../nodes/record/vp_record_node.h"
#include "../nodes/vp_rtsp_des_node.h"

#include "../utils/vp_tritonserver.h"

/*
* ## 1-1-1 triton server sample ##
* 1 video input, 1 infer task, and 1 output.
*/

#if _1_1_1_tritonserver_sample

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    TRITON_SERVER_INIT("/models", 0);

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "/video_pipe_c/test_video/test.avi");
    auto yunet_face_detector_0 = std::make_shared<vp_nodes::vp_yunet_face_detector_node>("yunet_face_detector_0", 
        "/models/face_detect/1/model.onnx", "face_detect", 1);
    auto sface_face_encoder_0 = std::make_shared<vp_nodes::vp_sface_feature_encoder_node>("sface_face_encoder_0", 
        "/models/face_recog/1/model.onnx", "face_recog", 1);
    auto osd_0 = std::make_shared<vp_nodes::vp_face_osd_node_v2>("osd_0");
    auto rtsp_des_0 = std::make_shared<vp_nodes::vp_rtsp_des_node>("rtsp_des_0", 0, 8000, "rtsp_0");

    // construct pipeline
    yunet_face_detector_0->attach_to({file_src_0});
    sface_face_encoder_0->attach_to({yunet_face_detector_0});
    osd_0->attach_to({sface_face_encoder_0});
    rtsp_des_0->attach_to({osd_0});

    file_src_0->start();

    // for debug purpose
    while(1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }
    TRITON_SERVER_UNINIT();
    return 0;
}

#endif