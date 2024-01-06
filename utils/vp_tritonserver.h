/********************************************
 * @Author: zjd
 * @Date: 2024-01-03
 * @LastEditTime: 2024-01-03
 * @LastEditors: zjd
 ********************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include "logger/vp_logger.h"
#include "triton/core/tritonserver.h"

namespace vp_utils
{

    class TritonTensor
    {
    public:
        TritonTensor(int dtype, const std::vector<int64_t>& shape, void* data = nullptr);
        ~TritonTensor();

        TRITONSERVER_DataType dataType() const { return m_dtype; }
        std::vector<int64_t> shape() const { return m_shape; }

        template<class T>
        T* base() const { return (T*)m_base; }

        size_t byteSize() const { return m_byte_size; }

    private:
        TRITONSERVER_DataType                m_dtype;
        std::vector<int64_t>                 m_shape;
        size_t                               m_byte_size;
        char*                                m_base = nullptr;
        bool                                 m_own = false;
    };

    typedef struct ModelInfo
    {
        std::string                             name;
        std::string                             version;
        std::vector<std::string>                inputs;
        std::vector<std::string>                outputs;
        std::vector<TRITONSERVER_DataType>      inputs_datatype;
        std::vector<TRITONSERVER_DataType>      outputs_datatype;
        std::vector<std::vector<int64_t>>       inputs_dims;
        std::vector<std::vector<int64_t>>       outputs_dims;
    } ModelInfo;

    // #define TRITON_SERVER_INIT(model_repository_path, verbose_level) \
    //     new vp_utils::TritonServerInfer(model_repository_path, verbose_level)

    #define TRITON_SERVER_INIT(model_repository_path, verbose_level) \
        vp_utils::TritonServerInfer::Instance().init(model_repository_path, verbose_level)

    #define TRITON_SERVER_INFER(model_name, model_version, inputs, outputs) \
        vp_utils::TritonServerInfer::Instance().infer(model_name, model_version, inputs, outputs)

    #define TRITON_SERVER_UNINIT() vp_utils::TritonServerInfer::Instance().uninit()

    class TritonServerInfer
    {
    public:
        static TritonServerInfer& Instance();
        void init(std::string model_repository_path, int verbose_level = 0, int timeout = 500);
        void uninit();
        // TritonServerInfer(std::string model_repository_path, int verbose_level = 0, int timeout = 500);
        // ~TritonServerInfer() = default;
        void infer(const std::string model_name, const std::string model_version, 
            const std::vector<std::shared_ptr<TritonTensor>>& input_tensors, 
            std::vector<std::shared_ptr<TritonTensor>>& output_tensors);

        void infer(const std::string model_name, const std::string model_version, 
            const std::vector<std::shared_ptr<TritonTensor>>& input_tensors, 
            std::vector<cv::Mat>& output_mats);

        void infer(const std::string model_name, const std::string model_version, 
            const std::vector<cv::Mat>& input_mats, std::vector<cv::Mat>& output_mats);

    private:
        TritonServerInfer() = default;
        ~TritonServerInfer() = default;
        void prepraeModelInputTensors(const ModelInfo& model_info, const std::vector<cv::Mat>& input_mats, 
            std::vector<std::shared_ptr<TritonTensor>>& input_tensors);

        void parseModelInferResponse(TRITONSERVER_InferenceResponse* response, const ModelInfo& model_info, 
            std::vector<std::shared_ptr<TritonTensor>>& output_tensors);

        void parseModelInferResponse(TRITONSERVER_InferenceResponse* response, const ModelInfo& model_info, 
            std::vector<cv::Mat>& output_mats);

    private:
        uint32_t                                                    m_api_version_major;
        uint32_t                                                    m_api_version_minor;
        std::string                                                 m_model_repository_path;
        int32_t                                                     m_verbose_level;
        std::shared_ptr<TRITONSERVER_Server>                        m_server;
        std::map<std::string, ModelInfo>                            m_models_info;
    };

}  // namespace vp_utils
