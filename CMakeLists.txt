cmake_minimum_required(VERSION 2.6)

project(video_pipe)

add_definitions(-DAPI_EXPORTS)
add_definitions(-O2 -pthread)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# OpenCV
set(OpenCV_DIR "/video_pipe_c/opencv-4.6.0/build")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories( ${CUDA_INCLUDE_DIRS} )

# GST
include(FindPkgConfig)# equals `pkg-config --cflags --libs gstreamer-1.0`
pkg_check_modules(GST   REQUIRED gstreamer-1.0)
pkg_check_modules(GSTAPP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GST_RTSP REQUIRED gstreamer-rtsp-server-1.0)

message(STATUS "GStreamer library status:")
message(STATUS "    version: ${GST_VERSION}")
message(STATUS "    libraries: ${GST_LIBRARIES} ${GSTAPP_LIBRARIES} ${GST_RTSP_LIBRARIES}")
message(STATUS "    include path: ${GST_INCLUDE_DIRS}")
include_directories( ${GST_INCLUDE_DIRS} )
set (GST_DEPEND_LIBS ${GST_LIBRARIES} ${GSTAPP_LIBRARIES} ${GST_RTSP_LIBRARIES})

# paddle inference
set(PADDLE_INFERENCE_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/paddle_inference/paddle/include)
set(PADDLE_INFERENCE_LIB_DIRS ${CMAKE_SOURCE_DIR}/paddle_inference/paddle/lib)
set(GLOG_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/paddle_inference/third_party/install/glog/include)
set(GLOG_LIB_DIRS ${CMAKE_SOURCE_DIR}/paddle_inference/third_party/install/glog/lib)
set(MKLDNN_LIBS ${CMAKE_SOURCE_DIR}/paddle_inference/third_party/install/mkldnn/lib/libdnnl.so.3)
set(MKLML_LIBS ${CMAKE_SOURCE_DIR}/paddle_inference/third_party/install/mklml/lib/libiomp5.so
    ${CMAKE_SOURCE_DIR}/paddle_inference/third_party/install/mklml/lib/libmklml_intel.so)
set(PADDLE2ONNX_LIBS ${CMAKE_SOURCE_DIR}/paddle_inference/third_party/install/paddle2onnx/lib/libpaddle2onnx.so)
set(ONNXRUNTIME_LIBS ${CMAKE_SOURCE_DIR}/paddle_inference/third_party/install/onnxruntime/lib/libonnxruntime.so)
set(PADDLE_THIRDPARTY_LIBS ${MKLDNN_LIBS} ${MKLML_LIBS} ${PADDLE2ONNX_LIBS} ${ONNXRUNTIME_LIBS})
set(PADDLE_THIRDPARTY_LIBS )

# triton server
set(TRITON_SERVER_INCLUDE_DIRS /opt/tritonserver/include)
set(TRITON_SERVER_LIBS /opt/tritonserver/lib/libtritonserver.so)

# rapid json
set(RAPID_JSON_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/third_party/rapidjson/include)

# set include directories
include_directories( ${PADDLE_INFERENCE_INCLUDE_DIRS} ${TRITON_SERVER_INCLUDE_DIRS} ${RAPID_JSON_INCLUDE_DIRS})

# trt_vehicle
file(GLOB TRT_VEHICLE_SOURCES "third_party/trt_vehicle/**/*.cpp")
cuda_add_library(trt_vehicle SHARED ${TRT_VEHICLE_SOURCES})
target_link_libraries(trt_vehicle nvinfer ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# paddle_ocr
# file(GLOB PADDLE_OCR_SOURCES "third_party/paddle_ocr/**/*.cpp")
# add_library(paddle_ocr SHARED ${PADDLE_OCR_SOURCES})
# target_link_directories(paddle_ocr PRIVATE ${PADDLE_INFERENCE_LIB_DIRS} ${GLOG_LIB_DIRS})
# target_include_directories(paddle_ocr PRIVATE ${PADDLE_INFERENCE_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})
# target_link_libraries(paddle_ocr PRIVATE paddle_inference glog)

file(GLOB NODES "nodes/*.cpp")
file(GLOB INFERS_NODES "nodes/infers/*.cpp")
file(GLOB OSD_NODES "nodes/osd/*.cpp")
file(GLOB TRACK_NODES "nodes/track/*.cpp" "nodes/track/sort/*.cpp")
file(GLOB RECORD_NODES "nodes/record/*.cpp")
file(GLOB OBJECTS "objects/*.cpp")
file(GLOB ELEMENTS_OBJECTS "objects/elements/*.cpp")
file(GLOB SHAPES_OBJECTS "objects/shapes/*.cpp")
file(GLOB BA_OBJECTS "objects/ba/*.cpp")
file(GLOB UTILS_ANALYSIS_BOARD "utils/analysis_board/*.cpp")
file(GLOB LOGGER "utils/logger/*.cpp")
file(GLOB TRITON_SERVER_INFER "utils/*.cpp")
list(REMOVE_ITEM INFERS_NODES ${CMAKE_CURRENT_SOURCE_DIR}/nodes/infers/vp_ppocr_text_detector_node.cpp)


add_library(video_pipe SHARED ${NODES} ${INFERS_NODES} ${OSD_NODES} ${TRACK_NODES} ${RECORD_NODES}
    ${OBJECTS} ${ELEMENTS_OBJECTS} ${SHAPES_OBJECTS} ${BA_OBJECTS} ${UTILS_ANALYSIS_BOARD} ${LOGGER}
    ${TRITON_SERVER_INFER})
# target_link_libraries(video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} paddle_ocr trt_vehicle -lstdc++fs)
target_link_libraries(video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} trt_vehicle -lstdc++fs)

add_executable(1-1-1_sample sample/1-1-1_sample.cpp)
target_link_libraries(1-1-1_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(1-1-N_sample sample/1-1-N_sample.cpp)
target_link_libraries(1-1-N_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(1-N-N_sample sample/1-N-N_sample.cpp)
target_link_libraries(1-N-N_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(N-1-N_sample sample/N-1-N_sample.cpp)
target_link_libraries(N-1-N_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(N-N_sample sample/N-N_sample.cpp)
target_link_libraries(N-N_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(paddle_infer_sample sample/paddle_infer_sample.cpp)
target_link_libraries(paddle_infer_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(src_des_sample sample/src_des_sample.cpp)
target_link_libraries(src_des_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(trt_infer_sample sample/trt_infer_sample.cpp)
target_link_libraries(trt_infer_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(vp_logger_sample sample/vp_logger_sample.cpp)
target_link_libraries(vp_logger_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(face_tracking_sample sample/face_tracking_sample.cpp)
target_link_libraries(face_tracking_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(interaction_with_pipe_sample sample/interaction_with_pipe_sample.cpp)
target_link_libraries(interaction_with_pipe_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(message_broker_sample sample/message_broker_sample.cpp)
target_link_libraries(message_broker_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(message_broker_sample2 sample/message_broker_sample2.cpp)
target_link_libraries(message_broker_sample2 video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(vehicle_tracking_sample sample/vehicle_tracking_sample.cpp)
target_link_libraries(vehicle_tracking_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(record_sample sample/record_sample.cpp)
target_link_libraries(record_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})

add_executable(triton_infer_sample sample/triton_infer_sample.cpp)
target_link_libraries(triton_infer_sample video_pipe ${OpenCV_LIBS} ${TRITON_SERVER_LIBS} ${PADDLE_THIRDPARTY_LIBS} ${GST_DEPEND_LIBS})
