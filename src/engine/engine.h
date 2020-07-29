#ifndef ENGINE_H
#define ENGINE_H
#include <iostream>
#include <string>
#include <vector>
#include "NvInfer.h"

#ifndef NV_CUDA_CHECK
#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }
#endif

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
            : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportableSeverity;
};

class Engine
{
public:
    Engine(std::string model_path, const char* output_name = "detection_out",
           int output_size = 700, int width = 640, int height = 360, int device = 0);
    ~Engine();

    void forward(float* input);

public:
    int m_device_;

    Logger m_gLogger_;
    cudaStream_t m_cuda_stream_;
    nvinfer1::IRuntime* m_runtime_;
    nvinfer1::ICudaEngine* m_engine_;
    nvinfer1::IExecutionContext* m_context_;

    std::vector<void*> m_device_buffers_;

    const char* m_input_blob_name_;            // Input blob name
    const char* m_output_blob_name_; // Output blob name
    int m_input_index_;
    int m_output_index_;

    float* m_detection_out_;
    int m_width_;
    int m_height_;
    int m_output_size_;
};

#endif //ENGINE_H
