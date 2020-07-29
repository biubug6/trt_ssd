#include <fstream>
#include <assert.h>
#include "engine.h"
#include "NvInferPlugin.h"
#include "cuda.h"
#include "cuda_runtime.h"

Engine::Engine(std::string model_path, const char* output_name, int output_size, int width, int height, int device):
        m_input_blob_name_("data") ,m_output_blob_name_(output_name), m_output_size_(output_size), m_width_(width), m_height_(height), m_device_(device)
{
    m_gLogger_ = Logger();
    cudaSetDevice(m_device_);
    initLibNvInferPlugins(&m_gLogger_, "");
    m_runtime_ = nvinfer1::createInferRuntime(m_gLogger_);

    // read model
    std::vector<char> trtModelStreamFromFile;
    size_t size{0};
    std::ifstream file(model_path, std::ios::binary);
    if(file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);

        trtModelStreamFromFile.resize(size);
        file.read(trtModelStreamFromFile.data(), size);
        file.close();
    }

    // create engine
    m_engine_ = m_runtime_->deserializeCudaEngine(trtModelStreamFromFile.data(), size, nullptr);
    assert(m_engine_ != nullptr);

    // create context
    m_context_ = m_engine_->createExecutionContext();
    assert(m_context_ != nullptr);

    // allocateBuffers
    assert(m_engine_->getNbBindings() == 2);
    m_device_buffers_.resize(2, nullptr);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    m_input_index_ = m_engine_->getBindingIndex(m_input_blob_name_);
    m_output_index_ = m_engine_->getBindingIndex(m_output_blob_name_);

    // Create GPU buffers and a stream
    NV_CUDA_CHECK(cudaMalloc((void**)&m_device_buffers_[m_input_index_], 1 * 3 * m_height_ * m_width_ * sizeof(float))); // Data
    NV_CUDA_CHECK(cudaMalloc((void**)&m_device_buffers_[m_output_index_], m_output_size_ * sizeof(float)));      // Detection_out

    m_detection_out_ = new float[m_output_size_];

    NV_CUDA_CHECK(cudaStreamCreate(&m_cuda_stream_));

}

Engine::~Engine()
{
    cudaStreamDestroy(m_cuda_stream_);

    // destroy the engine
    m_context_->destroy();
    m_engine_->destroy();
    m_runtime_->destroy();

    for (auto& deviceBuffer : m_device_buffers_)
    {
        NV_CUDA_CHECK(cudaFree(deviceBuffer));
    }

    delete[] m_detection_out_;
}


void Engine::forward(float* input)
{
    cudaSetDevice(m_device_);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    NV_CUDA_CHECK(cudaMemcpyAsync(m_device_buffers_[m_input_index_], input,
                                  m_height_*m_width_ * 3 * sizeof(float), cudaMemcpyHostToDevice, m_cuda_stream_));
    m_context_->enqueue(1, m_device_buffers_.data(), m_cuda_stream_, nullptr);

    NV_CUDA_CHECK(cudaMemcpyAsync(m_detection_out_, m_device_buffers_[m_output_index_],
                                  m_output_size_ * sizeof(float), cudaMemcpyDeviceToHost, m_cuda_stream_));
    cudaStreamSynchronize(m_cuda_stream_);
}
