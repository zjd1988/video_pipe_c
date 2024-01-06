/********************************************
 * @Author: zhaojd
 * @Date: 2024-01-03
 * @LastEditTime: 2024-01-03
 ********************************************/
#include <memory>
#include <future>
#include <sstream>
#include <numeric>
#include <functional>
#include <algorithm>
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "vp_tritonserver.h"

namespace vp_utils
{

    #define FAIL(MSG)                                 \
    do {                                              \
        std::cerr << "error: " << (MSG) << std::endl; \
        exit(1);                                      \
    } while (false)

    #define FAIL_IF_ERR(X, MSG)                                       \
    do {                                                              \
        TRITONSERVER_Error* err__ = (X);                              \
        if (err__ != nullptr) {                                       \
        std::cerr << "error: " << (MSG) << ": "                       \
                    << TRITONSERVER_ErrorCodeString(err__) << " - "   \
                    << TRITONSERVER_ErrorMessage(err__) << std::endl; \
        TRITONSERVER_ErrorDelete(err__);                              \
        exit(1);                                                      \
        }                                                             \
    } while (false)

    #define THROW_IF_ERR(EX_TYPE, X, MSG)                                     \
    do {                                                                      \
        TRITONSERVER_Error* err__ = (X);                                      \
        if (err__ != nullptr) {                                               \
        auto ex__ = (EX_TYPE)(std::string("error: ") + (MSG) + ": " +         \
                                TRITONSERVER_ErrorCodeString(err__) + " - " + \
                                TRITONSERVER_ErrorMessage(err__));            \
        TRITONSERVER_ErrorDelete(err__);                                      \
        throw ex__;                                                           \
        }                                                                     \
    } while (false)

    #define IGNORE_ERR(X)                  \
    do {                                   \
        TRITONSERVER_Error* err__ = (X);   \
        if (err__ != nullptr) {            \
        TRITONSERVER_ErrorDelete(err__);   \
        }                                  \
    } while (false)

#ifdef TRITON_ENABLE_GPU
    #define FAIL_IF_CUDA_ERR(X, MSG)                                           \
    do {                                                                       \
        cudaError_t err__ = (X);                                               \
        if (err__ != cudaSuccess) {                                            \
        std::cerr << "error: " << (MSG) << ": " << cudaGetErrorString(err__)   \
                    << std::endl;                                              \
        exit(1);                                                               \
        }                                                                      \
    } while (false)
#endif  // TRITON_ENABLE_GPU

    // TritonServerInfer* tritonServerInferNew()
    // {

    // }

    static size_t getTritonDataTypeByteSize(TRITONSERVER_DataType dtype)
    {
        size_t byte_size = 0;
        switch (dtype)
        {
            case TRITONSERVER_TYPE_UINT8:
            case TRITONSERVER_TYPE_INT8:
                byte_size = sizeof(int8_t);
                break;
            case TRITONSERVER_TYPE_UINT16:
            case TRITONSERVER_TYPE_INT16:
                byte_size = sizeof(int16_t);
                break;
            case TRITONSERVER_TYPE_UINT32:
            case TRITONSERVER_TYPE_INT32:
            case TRITONSERVER_TYPE_FP32:
                byte_size = sizeof(int32_t);
                break;
            case TRITONSERVER_TYPE_UINT64:
            case TRITONSERVER_TYPE_INT64:
            case TRITONSERVER_TYPE_FP64:
                byte_size = sizeof(int64_t);
                break;
            default:
                FAIL("get invalid datatype " + std::to_string(int(dtype)) + " when get datatype bytesize");
        }
        return byte_size;
    }

    static TRITONSERVER_DataType convertVPDataTypeToTritonDataType(int vp_datatype)
    {
        // #define CV_8U   0
        // #define CV_8S   1
        // #define CV_16U  2
        // #define CV_16S  3
        // #define CV_32S  4
        // #define CV_32F  5
        // #define CV_64F  6
        if (0 == vp_datatype)
            return TRITONSERVER_TYPE_UINT8;
        else if (1 == vp_datatype)
            return TRITONSERVER_TYPE_INT8;
        else if (2 == vp_datatype)
            return TRITONSERVER_TYPE_UINT16;
        else if (3 == vp_datatype)
            return TRITONSERVER_TYPE_INT16;
        else if (4 == vp_datatype)
            return TRITONSERVER_TYPE_INT32;
        else if (5 == vp_datatype)
            return TRITONSERVER_TYPE_FP32;
        else if (6 == vp_datatype)
            return TRITONSERVER_TYPE_FP64;
        else
            return TRITONSERVER_TYPE_INVALID;
    }

    static int convertTrtionDataTypeToVPDataType(TRITONSERVER_DataType datatype)
    {
        if (TRITONSERVER_TYPE_UINT8 == datatype)
            return 0;
        else if (TRITONSERVER_TYPE_INT8 == datatype)
            return 1;
        else if (TRITONSERVER_TYPE_UINT16 == datatype)
            return 2;
        else if (TRITONSERVER_TYPE_INT16 == datatype)
            return 3;
        else if (TRITONSERVER_TYPE_INT32 == datatype)
            return 4;
        else if (TRITONSERVER_TYPE_FP32 == datatype)
            return 5;
        else if (TRITONSERVER_TYPE_FP64 == datatype)
            return 6;
        else
            return -1;
    }

    static TRITONSERVER_DataType convertStrToTritonDataType(std::string datatype_str)
    {
        if (0 == strcmp(datatype_str.c_str(), "UINT8"))
            return TRITONSERVER_TYPE_UINT8;
        else if (0 == strcmp(datatype_str.c_str(), "UINT16"))
            return TRITONSERVER_TYPE_UINT16;
        else if (0 == strcmp(datatype_str.c_str(), "UINT32"))
            return TRITONSERVER_TYPE_UINT32;
        else if (0 == strcmp(datatype_str.c_str(), "UINT64"))
            return TRITONSERVER_TYPE_UINT64;
        else if (0 == strcmp(datatype_str.c_str(), "INT8"))
            return TRITONSERVER_TYPE_INT8;
        else if (0 == strcmp(datatype_str.c_str(), "INT16"))
            return TRITONSERVER_TYPE_INT16;
        else if (0 == strcmp(datatype_str.c_str(), "INT32"))
            return TRITONSERVER_TYPE_INT32;
        else if (0 == strcmp(datatype_str.c_str(), "INT64"))
            return TRITONSERVER_TYPE_INT64;
        else if (0 == strcmp(datatype_str.c_str(), "FP32"))
            return TRITONSERVER_TYPE_FP32;
        else if (0 == strcmp(datatype_str.c_str(), "FP64"))
            return TRITONSERVER_TYPE_FP64;
        else
            return TRITONSERVER_TYPE_INVALID;
    }

    TritonTensor::TritonTensor(int dtype, const std::vector<int64_t>& shape, void* data)
    {
        m_dtype = convertVPDataTypeToTritonDataType(dtype);
        if (TRITONSERVER_TYPE_INVALID == m_dtype)
        {
            FAIL("invalid triton tensor datatype " + std::to_string(dtype));
        }
        m_shape = shape;
        int64_t ele_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        m_byte_size = ele_count * getTritonDataTypeByteSize(m_dtype);
        if (nullptr == data)
        {
            m_base = new char[m_byte_size];
            if (nullptr == m_base)
            {
                FAIL("malloc triton tensor memory buffer fail");
                return;
            }
            m_own = true;
        }
        else
            m_base = (char*)data;
    }

    TritonTensor::~TritonTensor()
    {
        if (m_own)
            delete []m_base;
    }

    static TRITONSERVER_Error* parseModelMetadata(const rapidjson::Document& model_metadata, ModelInfo& model_info)
    {
        std::string model_name = model_info.name;
        std::string model_version = model_info.version;
        std::string model_key = model_name + ":" + model_version;
        model_info.inputs_dims.clear();
        model_info.outputs_dims.clear();
        for (const auto& input : model_metadata["inputs"].GetArray())
        {
            std::string name = input["name"].GetString();
            std::string datatype_str = input["datatype"].GetString();
            TRITONSERVER_DataType datatype = convertStrToTritonDataType(datatype_str);
            if (TRITONSERVER_TYPE_INVALID == datatype)
            {
                FAIL("model " + model_key + " input:" + name + " contain unsupported datatype " + datatype_str);
            }
            std::vector<int64_t> shape_vec;
            for (const auto &shape_item : input["shape"].GetArray())
            {
                int64_t dim_value = shape_item.GetInt64();
                shape_vec.push_back(dim_value);
            }
            model_info.inputs.push_back(name);
            model_info.inputs_datatype.push_back(datatype);
            model_info.inputs_dims.push_back(shape_vec);
        }

        for (const auto& output : model_metadata["outputs"].GetArray())
        {
            std::string name = output["name"].GetString();
            std::string datatype_str = output["datatype"].GetString();
            TRITONSERVER_DataType datatype = convertStrToTritonDataType(datatype_str);
            if (TRITONSERVER_TYPE_INVALID == datatype)
            {
                FAIL("model " + model_key + " output:" + name + " contain unsupported datatype " + datatype_str);
            }
            std::vector<int64_t> shape_vec;
            for (const auto &shape_item : output["shape"].GetArray())
            {
                int64_t dim_value = shape_item.GetInt64();
                shape_vec.push_back(dim_value);
            }
            model_info.outputs.push_back(name);
            model_info.outputs_datatype.push_back(datatype);
            model_info.outputs_dims.push_back(shape_vec);
        }

        return nullptr;
    }

    static TRITONSERVER_Error* ResponseAlloc(TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
        size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
        int64_t preferred_memory_type_id, void* userp, void** buffer,
        void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
        int64_t* actual_memory_type_id)
    {
        // Initially attempt to make the actual memory type and id that we
        // allocate be the same as preferred memory type
        *actual_memory_type = preferred_memory_type;
        *actual_memory_type_id = preferred_memory_type_id;

        // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
        // need to do any other book-keeping.
        if (byte_size == 0)
        {
            *buffer = nullptr;
            *buffer_userp = nullptr;
            VP_DEBUG("allocated " + std::to_string(byte_size) + " bytes for result tensor " + tensor_name);
        }
        else
        {
            void* allocated_ptr = nullptr;

            switch (*actual_memory_type)
            {
            #ifdef TRITON_ENABLE_GPU
                case TRITONSERVER_MEMORY_CPU_PINNED: {
                    auto err = cudaSetDevice(*actual_memory_type_id);
                    if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                        (err != cudaErrorInsufficientDriver)) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "unable to recover current CUDA device: " +
                            std::string(cudaGetErrorString(err)))
                            .c_str());
                    }

                    err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
                    if (err != cudaSuccess) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "cudaHostAlloc failed: " +
                            std::string(cudaGetErrorString(err)))
                            .c_str());
                    }
                    break;
                }

                case TRITONSERVER_MEMORY_GPU: {
                    auto err = cudaSetDevice(*actual_memory_type_id);
                    if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                        (err != cudaErrorInsufficientDriver)) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "unable to recover current CUDA device: " +
                            std::string(cudaGetErrorString(err)))
                            .c_str());
                    }

                    err = cudaMalloc(&allocated_ptr, byte_size);
                    if (err != cudaSuccess) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        std::string(
                            "cudaMalloc failed: " + std::string(cudaGetErrorString(err)))
                            .c_str());
                    }
                    break;
                }
            #endif  // TRITON_ENABLE_GPU

                // Use CPU memory if the requested memory type is unknown
                // (default case).
                case TRITONSERVER_MEMORY_CPU:
                default:
                {
                    *actual_memory_type = TRITONSERVER_MEMORY_CPU;
                    allocated_ptr = malloc(byte_size);
                    break;
                }
            }

            // Pass the tensor name with buffer_userp so we can show it when
            // releasing the buffer.
            if (allocated_ptr != nullptr)
            {
                *buffer = allocated_ptr;
                *buffer_userp = new std::string(tensor_name);
                VP_DEBUG("allocated " + std::to_string(byte_size) + " bytes in "
                            + TRITONSERVER_MemoryTypeString(*actual_memory_type) +
                            " for result tensor " + tensor_name);
            }
        }

        return nullptr;  // Success
    }

    static TRITONSERVER_Error* ResponseRelease(TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
        size_t byte_size, TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
    {
        std::string* name = nullptr;
        if (buffer_userp != nullptr)
        {
            name = reinterpret_cast<std::string*>(buffer_userp);
        }
        else
        {
            name = new std::string("<unknown>");
        }

        VP_DEBUG("Releasing response buffer of size " + std::to_string(byte_size) + 
                    + " in " + TRITONSERVER_MemoryTypeString(memory_type) + 
                    + " for result " + *name);

        switch (memory_type)
        {
            case TRITONSERVER_MEMORY_CPU:
                free(buffer);
                break;
            #ifdef TRITON_ENABLE_GPU
                case TRITONSERVER_MEMORY_CPU_PINNED: 
                {
                    auto err = cudaSetDevice(memory_type_id);
                    if (err == cudaSuccess)
                    {
                        err = cudaFreeHost(buffer);
                    }
                    if (err != cudaSuccess)
                    {
                        std::cerr << "error: failed to cudaFree " << buffer << ": "
                                << cudaGetErrorString(err) << std::endl;
                    }
                    break;
                }
                case TRITONSERVER_MEMORY_GPU:
                {
                    auto err = cudaSetDevice(memory_type_id);
                    if (err == cudaSuccess)
                    {
                        err = cudaFree(buffer);
                    }
                    if (err != cudaSuccess)
                    {
                        std::cerr << "error: failed to cudaFree " << buffer << ": "
                                << cudaGetErrorString(err) << std::endl;
                    }
                    break;
                }
            #endif  // TRITON_ENABLE_GPU
            default:
                std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                            << std::endl;
                break;
        }
        delete name;
        return nullptr;  // Success
    }

    static void InferRequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
    {
        // TRITONSERVER_InferenceRequestDelete(request);
        std::promise<void>* barrier = reinterpret_cast<std::promise<void>*>(userp);
        barrier->set_value();
    }

    static void InferResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
    {
        if (response != nullptr)
        {
            // Send 'response' to the future.
            std::promise<TRITONSERVER_InferenceResponse*>* p =
                reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
            p->set_value(response);
            delete p;
        }
    }

    TritonServerInfer& TritonServerInfer::Instance()
    {
        static TritonServerInfer triton_server;
        return triton_server;
    }

    void TritonServerInfer::uninit()
    {
        m_server.reset();
    }

    void TritonServerInfer::init(std::string model_repository_path, int verbose_level, int timeout)
    {
        FAIL_IF_ERR(TRITONSERVER_ApiVersion(&m_api_version_major, &m_api_version_minor), 
            "getting Triton API version");
        if ((TRITONSERVER_API_VERSION_MAJOR != m_api_version_major) ||
            (TRITONSERVER_API_VERSION_MINOR > m_api_version_minor))
        {
            FAIL("triton server API version mismatch");
        }

        m_model_repository_path = model_repository_path;
        m_verbose_level = verbose_level;
        TRITONSERVER_ServerOptions* server_options = nullptr;
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options),
            "creating server options");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, model_repository_path.c_str()),
            "setting model repository path");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
            "setting verbose logging level");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(server_options, "/opt/tritonserver/backends"),
            "setting backend directory");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(server_options, "/opt/tritonserver/repoagents"),
            "setting repository agent directory");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
            "setting strict model configuration");

    #ifdef TRITON_ENABLE_GPU
        double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
    #else
        double min_compute_capability = 0;
    #endif  // TRITON_ENABLE_GPU
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(server_options, min_compute_capability),
            "setting minimum supported CUDA compute capability");

        // Create the server object using the option settings. The server
        // object encapsulates all the functionality of the Triton server
        // and allows access to the Triton server API. Typically only a
        // single server object is needed by an application, but it is
        // allowed to create multiple server objects within a single
        // application. After the server object is created the server
        // options can be deleted.
        TRITONSERVER_Server* server_ptr = nullptr;
        FAIL_IF_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options),
            "creating server object");
        FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options),
            "deleting server options");

        std::shared_ptr<TRITONSERVER_Server> server(server_ptr, TRITONSERVER_ServerDelete);
        m_server = std::move(server);
        // Wait until the server is both live and ready. The server will not
        // appear "ready" until all models are loaded and ready to receive
        // inference requests.
        size_t health_iters = 0;
        while (true)
        {
            bool live = false, ready = false;
            FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_server.get(), &live),
                "unable to get server liveness");
            FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_server.get(), &ready),
                "unable to get server readiness");
            VP_INFO(vp_utils::string_format("Server Health: live %d, ready %d", live, ready));
            if (live && ready)
            {
                break;
            }
            if (++health_iters >= 10)
            {
                FAIL("failed to find healthy inference server");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
        }

        // log server metadata
        {
            TRITONSERVER_Message* server_metadata_message;
            FAIL_IF_ERR(TRITONSERVER_ServerMetadata(m_server.get(), &server_metadata_message),
                "unable to get server metadata message");

            const char* buffer;
            size_t byte_size;
            FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(server_metadata_message, &buffer, &byte_size),
                "unable to serialize server metadata message");

            VP_INFO(std::string("Triton Server Metadata:"));
            VP_INFO(std::string(buffer, byte_size));

            FAIL_IF_ERR(TRITONSERVER_MessageDelete(server_metadata_message),
                "deleting server metadata message");
        }

        // init all models info
        {
            VP_INFO(std::string("init all models info"));
            // get model statistic message
            TRITONSERVER_Message* models_statistic_message;
            FAIL_IF_ERR(TRITONSERVER_ServerModelStatistics(m_server.get(), "", -1, &models_statistic_message),
                "unable to get models statistic message");

            const char* buffer;
            size_t byte_size;
            FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(models_statistic_message, &buffer, &byte_size),
                "unable to serialize models statistic message");

            VP_INFO(std::string("Triton Server Models Statistics:"));
            VP_INFO(std::string(buffer, byte_size));

            // parse model statistic message
            rapidjson::Document models_statistic_metadata;
            models_statistic_metadata.Parse(buffer, byte_size);
            if (models_statistic_metadata.HasParseError())
            {
                FAIL("error: failed to parse models statistic from JSON: " +
                    std::string(GetParseError_En(models_statistic_metadata.GetParseError())) +
                    " at " + std::to_string(models_statistic_metadata.GetErrorOffset()));
            }

            // delete model statistic message
            FAIL_IF_ERR(TRITONSERVER_MessageDelete(models_statistic_message),
                "deleting models statistic message");

            // init models info
            const rapidjson::Value &model_stats = models_statistic_metadata["model_stats"];
            for (auto &model_item : model_stats.GetArray())
            {
                ModelInfo model_info;
                std::string model_name = model_item["name"].GetString();
                std::string model_version = model_item["version"].GetString();
                std::string model_key = model_name + ":" + model_version;
                model_info.name = model_name;
                model_info.version = model_version;
                // get model metadata
                std::stringstream ss;
                int64_t model_version_int;
                ss << model_version;
                ss >> model_version_int;
                TRITONSERVER_Message* model_metadata_message;
                FAIL_IF_ERR(TRITONSERVER_ServerModelMetadata(m_server.get(), model_name.c_str(), model_version_int, &model_metadata_message), 
                    "unable to get model metadata message");
                const char* model_metadata_buffer;
                size_t model_metadata_byte_size;
                FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_metadata_message, &model_metadata_buffer, &model_metadata_byte_size),
                    "unable to serialize model metadata");

                VP_INFO(std::string("model " + model_key + " meatadata:"));
                VP_INFO(std::string(model_metadata_buffer, model_metadata_byte_size));

                // parse model metadata json
                rapidjson::Document model_metadata;
                model_metadata.Parse(model_metadata_buffer, model_metadata_byte_size);
                if (model_metadata.HasParseError())
                {
                    FAIL("error: failed to parse model " + model_key + std::string(" metadata from JSON: ") + 
                        std::string(GetParseError_En(model_metadata.GetParseError())) +
                        " at " + std::to_string(model_metadata.GetErrorOffset()));
                }

                // delete model metadata message
                FAIL_IF_ERR(TRITONSERVER_MessageDelete(model_metadata_message),
                    "deleting model metadata message");

                // check model name
                if (strcmp(model_metadata["name"].GetString(), model_name.c_str()))
                {
                    FAIL("unable to find metadata for model " + model_key);
                }

                // check model version
                bool found_version = false;
                if (model_metadata.HasMember("versions"))
                {
                    for (const auto& version : model_metadata["versions"].GetArray())
                    {
                        if (strcmp(version.GetString(), model_version.c_str()) == 0)
                        {
                            found_version = true;
                            break;
                        }
                    }
                }
                if (!found_version)
                {
                    FAIL("unable to find version 1 status for model");
                }

                FAIL_IF_ERR(parseModelMetadata(model_metadata, model_info), 
                    "parsing model " + model_key + " metadata");

                if (m_models_info.end() != m_models_info.find(model_key))
                {
                    VP_WARN("arleady init model info: " + model_key);
                    continue;
                }
                m_models_info[model_key] = model_info;
            }
        }
        return;
    }

    void TritonServerInfer::parseModelInferResponse(TRITONSERVER_InferenceResponse* response, 
        const ModelInfo& model_info, std::vector<std::shared_ptr<TritonTensor>>& output_tensors)
    {
        std::string model_name = model_info.name;
        std::string model_version = model_info.version;
        std::string model_key = model_name + ":" + model_version;
        // get model output count
        uint32_t output_count;
        FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
            "getting number of response outputs for model " + model_key);
        if (output_count != model_info.outputs.size())
        {
            FAIL("expecting " + std::to_string(model_info.outputs.size()) + " response outputs, got " + 
                std::to_string(output_count) + " for model " + model_key);
        }

        for (uint32_t idx = 0; idx < output_count; ++idx)
        {
            const char* cname;
            TRITONSERVER_DataType datatype;
            const int64_t* shape;
            uint64_t dim_count;
            const void* base;
            size_t byte_size;
            TRITONSERVER_MemoryType memory_type;
            int64_t memory_type_id;
            void* userp;

            FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(
                response, idx, &cname, &datatype, &shape, &dim_count, &base,
                &byte_size, &memory_type, &memory_type_id, &userp), "getting output info");

            if (cname == nullptr)
            {
                FAIL("unable to get output name for model " + model_key);
            }
            std::string output_name = model_info.outputs[idx];
            std::string name(cname);
            if (name != output_name)
            {
                FAIL("unexpected output '" + name + "' for model " + model_key);
            }

            auto expected_datatype = model_info.outputs_datatype[idx];
            if (datatype != expected_datatype)
            {
                FAIL("unexpected datatype '" + std::string(TRITONSERVER_DataTypeString(datatype)) + "' for '" +
                    name + "' , model " + model_key);
            }

            // parepare output tensor
            std::vector<int64_t> tensor_shape(shape, shape + dim_count);
            int vp_dtype = convertTrtionDataTypeToVPDataType(datatype);
            std::shared_ptr<TritonTensor> output_tensor(new TritonTensor(vp_dtype, tensor_shape));
            if (nullptr == output_tensor.get() || nullptr == output_tensor->base<void>())
            {
                FAIL("malloc tensor fail for output " + name + " ,model " + model_key);
            }
            // We make a copy of the data here... which we could avoid for
            // performance reasons but ok for this simple example.
            switch (memory_type)
            {
                case TRITONSERVER_MEMORY_CPU:
                {
                    VP_DEBUG(name + " is stored in system memory for model " + model_key);
                    memcpy(output_tensor->base<void>(), base, byte_size);
                    break;
                }

                case TRITONSERVER_MEMORY_CPU_PINNED:
                {
                    VP_DEBUG(name + " is stored in pinned memory for model " + model_key);
                    memcpy(output_tensor->base<void>(), base, byte_size);
                    break;
                }

            #ifdef TRITON_ENABLE_GPU
                case TRITONSERVER_MEMORY_GPU:
                {
                    VP_DEBUG(name + " is stored in GPU memory for model " + model_key);
                    FAIL_IF_CUDA_ERR(cudaMemcpy(output_tensor->base<void>(), base, byte_size, cudaMemcpyDeviceToHost),
                        "getting " + name + " data from GPU memory for model " + model_key);
                    break;
                }
            #endif

                default:
                    FAIL("unexpected memory type for model " + model_key);
            }
            output_tensors.push_back(output_tensor);
        }
        return;
    }

    void TritonServerInfer::infer(const std::string model_name, const std::string model_version, 
        const std::vector<std::shared_ptr<TritonTensor>>& input_tensors, 
        std::vector<std::shared_ptr<TritonTensor>>& output_tensors)
    {
        output_tensors.clear();
        // check model exists
        std::string model_key = model_name + ":" + model_version;
        if (m_models_info.end() == m_models_info.find(model_key))
        {
            FAIL("cannot not find model info for " + model_key);
        }
        const ModelInfo& model_info = m_models_info[model_key];

        // check input tensors size equal to model inputs
        if (input_tensors.size() != model_info.inputs.size())
        {
            FAIL("input tensors size " + std::to_string(input_tensors.size()) + 
                " not equalt to model inputs for " + model_key);
        }
        // When triton needs a buffer to hold an output tensor, it will ask
        // us to provide the buffer. In this way we can have any buffer
        // management and sharing strategy that we want. To communicate to
        // triton the functions that we want it to call to perform the
        // allocations, we create a "response allocator" object. We pass
        // this response allocate object to triton when requesting
        // inference. We can reuse this response allocate object for any
        // number of inference requests.
        TRITONSERVER_ResponseAllocator* allocator = nullptr;
        FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
            "creating response allocator for model " + model_key);

        // Create an inference request object. The inference request object
        // is where we set the name of the model we want to use for
        // inference and the input tensors.
        std::stringstream ss;
        int64_t model_version_int;
        ss << model_version;
        ss >> model_version_int;
        TRITONSERVER_InferenceRequest* irequest = nullptr;
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&irequest, m_server.get(), model_name.c_str(), model_version_int),
            "creating inference request, model " + model_key);

        // FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetId(irequest, "request_id"),
        //     "setting ID for the request, model " + model_key);

        std::unique_ptr<std::promise<void>> barrier = std::make_unique<std::promise<void>>();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(irequest, InferRequestRelease,
            reinterpret_cast<void*>(barrier.get())), "setting request release callback for model " + model_key);
        std::future<void> request_release_future = barrier->get_future();

        // FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(
        //     irequest, InferRequestRelease, nullptr), "setting request release callback for model " + model_key);

        // Add the model inputs to the request...
        for (size_t i = 0; i < model_info.inputs.size(); i++)
        {
            std::string input_name = model_info.inputs[i];
            std::vector<int64_t> input_shape = input_tensors[i]->shape();
            TRITONSERVER_DataType datatype = input_tensors[i]->dataType();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(irequest, input_name.c_str(), 
                datatype, &input_shape[0], input_shape.size()),
                "setting input: " + input_name + " meta-data for the request, model " + model_key);
            size_t input_size = input_tensors[i]->byteSize();
            const void* input_base = input_tensors[i]->base<void>();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(irequest, input_name.c_str(), input_base, input_size, 
                TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */),
                "assigning input: " + input_name + " data for request for " + model_key);
        }

        // Add the model outputs to the request...
        for (size_t i = 0; i < model_info.outputs.size(); i++)
        {
            std::string output_name = model_info.outputs[i];
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output_name.c_str()),
                "requesting output: " + output_name + " for the request");
        }

        // Perform inference by calling TRITONSERVER_ServerInferAsync. This
        // call is asynchronous and therefore returns immediately. The
        // completion of the inference and delivery of the response is done
        // by triton by calling the "response complete" callback functions
        // (InferResponseComplete in this case).
        {
            auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
            std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

            FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                irequest, allocator, nullptr /* response_allocator_userp */, InferResponseComplete, 
                reinterpret_cast<void*>(p)), "setting response callback for model " + model_key);

            FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_server.get(), irequest, nullptr /* trace */),
                "running inference for model " + model_key);

            // The InferResponseComplete function sets the std::promise so
            // that this thread will block until the response is returned.
            TRITONSERVER_InferenceResponse* completed_response = completed.get();
            FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(completed_response),
                "response status for model " + model_key);

            // parse model infer output from response
            parseModelInferResponse(completed_response, model_info, output_tensors);

            // delete model infer response
            FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(completed_response),
                "deleting inference response for model " + model_key);
        }

        request_release_future.get();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(irequest),
            "deleting inference request for model " + model_key);

        FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator),
            "deleting response allocator for model " + model_key);
        return;
    }

    void TritonServerInfer::parseModelInferResponse(TRITONSERVER_InferenceResponse* response, 
        const ModelInfo& model_info, std::vector<cv::Mat>& output_mats)
    {
        std::string model_name = model_info.name;
        std::string model_version = model_info.version;
        std::string model_key = model_name + ":" + model_version;
        // get model output count
        uint32_t output_count;
        FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
            "getting number of response outputs for model " + model_key);
        if (output_count != model_info.outputs.size())
        {
            FAIL("expecting " + std::to_string(model_info.outputs.size()) + " response outputs, got " + 
                std::to_string(output_count) + " for model " + model_key);
        }

        for (uint32_t idx = 0; idx < output_count; ++idx)
        {
            const char* cname;
            TRITONSERVER_DataType datatype;
            const int64_t* shape;
            uint64_t dim_count;
            const void* base;
            size_t byte_size;
            TRITONSERVER_MemoryType memory_type;
            int64_t memory_type_id;
            void* userp;

            FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(
                response, idx, &cname, &datatype, &shape, &dim_count, &base,
                &byte_size, &memory_type, &memory_type_id, &userp), "getting output info");

            if (cname == nullptr)
            {
                FAIL("unable to get output name for model " + model_key);
            }
            std::string output_name = model_info.outputs[idx];
            std::string name(cname);
            if (name != output_name)
            {
                FAIL("unexpected output '" + name + "' for model " + model_key);
            }

            auto expected_datatype = model_info.outputs_datatype[idx];
            if (datatype != expected_datatype)
            {
                FAIL("unexpected datatype '" + std::string(TRITONSERVER_DataTypeString(datatype)) + "' for '" +
                    name + "' , model " + model_key);
            }

            // parepare output tensor
            std::vector<int64_t> tensor_shape(shape, shape + dim_count);
            std::vector<int> mat_shape(tensor_shape.begin(), tensor_shape.end());
            int vp_dtype = convertTrtionDataTypeToVPDataType(datatype);
            cv::Mat output_mat(dim_count, &mat_shape[0], vp_dtype);
            if (true == output_mat.empty())
            {
                FAIL("malloc mat fail for output " + name + " ,model " + model_key);
            }
            // We make a copy of the data here... which we could avoid for
            // performance reasons but ok for this simple example.
            switch (memory_type)
            {
                case TRITONSERVER_MEMORY_CPU:
                {
                    VP_DEBUG(name + " is stored in system memory for model " + model_key);
                    memcpy((void*)output_mat.data, base, byte_size);
                    break;
                }

                case TRITONSERVER_MEMORY_CPU_PINNED:
                {
                    VP_DEBUG(name + " is stored in pinned memory for model " + model_key);
                    memcpy((void*)output_mat.data, base, byte_size);
                    break;
                }

            #ifdef TRITON_ENABLE_GPU
                case TRITONSERVER_MEMORY_GPU:
                {
                    VP_DEBUG(name + " is stored in GPU memory for model " + model_key);
                    FAIL_IF_CUDA_ERR(cudaMemcpy((void*)output_mat.data, base, byte_size, cudaMemcpyDeviceToHost),
                        "getting " + name + " data from GPU memory for model " + model_key);
                    break;
                }
            #endif

                default:
                    FAIL("unexpected memory type for model " + model_key);
            }
            output_mats.push_back(output_mat);
        }
        return;
    }

    void TritonServerInfer::infer(const std::string model_name, const std::string model_version, 
        const std::vector<std::shared_ptr<TritonTensor>>& input_tensors, std::vector<cv::Mat>& output_mats)
    {
        output_mats.clear();
        // check model exists
        std::string model_key = model_name + ":" + model_version;
        if (m_models_info.end() == m_models_info.find(model_key))
        {
            FAIL("cannot not find model info for " + model_key);
        }
        const ModelInfo& model_info = m_models_info[model_key];

        // check input tensors size equal to model inputs
        if (input_tensors.size() != model_info.inputs.size())
        {
            FAIL("input tensors size " + std::to_string(input_tensors.size()) + 
                " not equalt to model inputs for " + model_key);
        }

        // When triton needs a buffer to hold an output tensor, it will ask
        // us to provide the buffer. In this way we can have any buffer
        // management and sharing strategy that we want. To communicate to
        // triton the functions that we want it to call to perform the
        // allocations, we create a "response allocator" object. We pass
        // this response allocate object to triton when requesting
        // inference. We can reuse this response allocate object for any
        // number of inference requests.
        TRITONSERVER_ResponseAllocator* allocator = nullptr;
        FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
            "creating response allocator for model " + model_key);

        // Create an inference request object. The inference request object
        // is where we set the name of the model we want to use for
        // inference and the input tensors.
        std::stringstream ss;
        int64_t model_version_int;
        ss << model_version;
        ss >> model_version_int;
        TRITONSERVER_InferenceRequest* irequest = nullptr;
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&irequest, m_server.get(), model_name.c_str(), model_version_int),
            "creating inference request, model " + model_key);

        // FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetId(irequest, "request_id"),
        //     "setting ID for the request, model " + model_key);

        std::unique_ptr<std::promise<void>> barrier = std::make_unique<std::promise<void>>();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(irequest, InferRequestRelease,
            reinterpret_cast<void*>(barrier.get())), "setting request release callback for model " + model_key);
        std::future<void> request_release_future = barrier->get_future();

        // FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(
        //     irequest, InferRequestRelease, nullptr), "setting request release callback for model " + model_key);

        // Add the model inputs to the request...
        for (size_t i = 0; i < model_info.inputs.size(); i++)
        {
            std::string input_name = model_info.inputs[i];
            std::vector<int64_t> input_shape = input_tensors[i]->shape();
            TRITONSERVER_DataType datatype = input_tensors[i]->dataType();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(irequest, input_name.c_str(), 
                datatype, &input_shape[0], input_shape.size()),
                "setting input: " + input_name + " meta-data for the request, model " + model_key);
            size_t input_size = input_tensors[i]->byteSize();
            const void* input_base = input_tensors[i]->base<void>();
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                irequest, input_name.c_str(), input_base, input_size, TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */),
                "assigning input: " + input_name + " data for request for " + model_key);
        }

        // Add the model outputs to the request...
        for (size_t i = 0; i < model_info.outputs.size(); i++)
        {
            std::string output_name = model_info.outputs[i];
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output_name.c_str()),
                "requesting output: " + output_name + " for the request");
        }

        // Perform inference by calling TRITONSERVER_ServerInferAsync. This
        // call is asynchronous and therefore returns immediately. The
        // completion of the inference and delivery of the response is done
        // by triton by calling the "response complete" callback functions
        // (InferResponseComplete in this case).
        {
            auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
            std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

            FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                irequest, allocator, nullptr /* response_allocator_userp */, InferResponseComplete, 
                reinterpret_cast<void*>(p)), "setting response callback for model " + model_key);

            FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_server.get(), irequest, nullptr /* trace */),
                "running inference for model " + model_key);

            // The InferResponseComplete function sets the std::promise so
            // that this thread will block until the response is returned.
            TRITONSERVER_InferenceResponse* completed_response = completed.get();
            FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(completed_response),
                "response status for model " + model_key);

            // parse model infer output from response
            parseModelInferResponse(completed_response, model_info, output_mats);

            // delete model infer response
            FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(completed_response),
                "deleting inference response for model " + model_key);
        }

        request_release_future.get();
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(irequest),
            "deleting inference request for model " + model_key);

        FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator),
            "deleting response allocator for model " + model_key);

        return;
    }

    void TritonServerInfer::prepraeModelInputTensors(const ModelInfo& model_info, const std::vector<cv::Mat>& input_mats, 
        std::vector<std::shared_ptr<TritonTensor>>& input_tensors)
    {
        input_tensors.clear();
        std::string model_name = model_info.name;
        std::string model_version = model_info.version;
        std::string model_key = model_name + ":" + model_version;
        // check input tensors size equal to model inputs
        if (input_mats.size() != model_info.inputs.size())
        {
            FAIL("input mats size " + std::to_string(input_mats.size()) + 
                " not equalt to model inputs for " + model_key);
        }

        // construct input tensors
        for (size_t i = 0; i < input_mats.size(); i++)
        {
            std::string input_name = model_info.inputs[i];
            auto& input_mat = input_mats[i];
            if (input_mat.empty())
            {
                FAIL("model input " + input_name + " mat is empty for " + model_key);
            }
            int vp_dtype = input_mat.depth();
            auto input_dtype = convertVPDataTypeToTritonDataType(vp_dtype);
            auto expect_dtype = model_info.inputs_datatype[i];
            if (input_dtype != expect_dtype)
            {
                FAIL("model input " + input_name + " expect datatype " + std::to_string(expect_dtype) +
                    " but get " + std::to_string(input_dtype) + " for model " + model_key);
            }
            std::vector<int> mat_shape(&input_mat.size[0], &input_mat.size[0] + input_mat.dims);
            std::vector<int64_t> tensor_shape(mat_shape.begin(), mat_shape.end());
            std::shared_ptr<TritonTensor> input_tensor(new TritonTensor(vp_dtype, tensor_shape, input_mat.data));
            if (nullptr == input_tensor.get() || nullptr == input_tensor->base<void>())
            {
                FAIL("malloc model input " + input_name + " tensor fail for model " + model_key);
            }
            input_tensors.push_back(input_tensor);
        }

        return;
    }

    void TritonServerInfer::infer(const std::string model_name, const std::string model_version, 
        const std::vector<cv::Mat>& input_mats, std::vector<cv::Mat>& output_mats)
    {
        // check model exists
        std::string model_key = model_name + ":" + model_version;
        if (m_models_info.end() == m_models_info.find(model_key))
        {
            FAIL("cannot not find model info for " + model_key);
        }
        const ModelInfo& model_info = m_models_info[model_key];

        // prepare model input tensors
        std::vector<std::shared_ptr<TritonTensor>> input_tensors;
        prepraeModelInputTensors(model_info, input_mats, input_tensors);

        // infer
        infer(model_name, model_version, input_tensors, output_mats);

        return;
    }

} // namespace vp_utils