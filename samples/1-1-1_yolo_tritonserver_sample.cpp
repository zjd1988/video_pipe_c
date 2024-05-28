#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolo_tritonserver_node.h"
#include "../nodes/osd/vp_osd_node.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"
#include "../utils/vp_tritonserver.h"
/*
* ## multi detectors and classifiers sample ##
* show multi infer nodes work together.
* 1 detector and 2 classifiers applied on primary class ids(1/2/3).
*/

int main() {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();
    TRITON_SERVER_DEFAULT_INIT("/models", 0);

    // create nodes
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "test_video/yolov5_test.mp4", 0.6);
    /* primary detector */
    // labels (80 classes) for detector model
    auto primary_detector = std::make_shared<vp_nodes::vp_yolo_tritonserver_node>("primary_detector", 
        "models/yolo_models/yolov5_onnx/1/model.onnx", "yolov5_onnx", 1, "",
        "models/yolo_models/coco_80_labels_list.txt");
    auto osd_0 = std::make_shared<vp_nodes::vp_osd_node>("osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_o", 0);

    // construct pipeline
    primary_detector->attach_to({file_src_0});
    osd_0->attach_to({primary_detector});
    screen_des_0->attach_to({osd_0});

    // start
    file_src_0->start();

    // for debug purpose
    vp_utils::vp_analysis_board board({file_src_0});
    board.display();

    TRITON_SERVER_UNINIT();
}