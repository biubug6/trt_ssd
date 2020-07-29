#include "personDetect.h"

Detect::Detect(Parameter& param)
{
    m_result_thresh_ = param.resultThresh;
    m_mean_ = param.mean;
    m_input_ = new float[param.width*param.height*3];
    kernel.reset(new Engine(param.model_path, param.output_name.c_str(), param.output_size, param.width, param.height, param.device));
}

Detect::~Detect()
{
    delete [] m_input_;
}

void Detect::infer(const cv::Mat &img, std::vector<std::vector<float>> &bboxs)
{
    cv::Mat img_resize;
    int org_width = img.cols;
    int org_height = img.rows;
    cv::resize(img, img_resize, cv::Size(kernel->m_width_, kernel->m_height_));

    int off_set1 = img_resize.cols*img_resize.rows;
    int off_set2 = img_resize.cols*img_resize.rows*2;

    for (int i = 0; i < img_resize.rows; ++i) {
        for (int j = 0; j < img_resize.cols; ++j) {
            m_input_[i*img_resize.cols + j] = float(img_resize.at<cv::Vec3b>(i, j)[0] - m_mean_.x);
            m_input_[i*img_resize.cols + j + off_set1] = float(img_resize.at<cv::Vec3b>(i, j)[1] - m_mean_.y);
            m_input_[i*img_resize.cols + j + off_set2] = float(img_resize.at<cv::Vec3b>(i, j)[2] - m_mean_.z);
        }
    }

    kernel->forward(m_input_);

    decode(bboxs, org_width, org_height);
}

void Detect::decode(std::vector<std::vector<float>>& output, int width, int height)
{
    float* det = kernel->m_detection_out_;
    for (int i = 0; i < 100; ++i)
    {
        std::vector<float> bbox;
        bbox.resize(5);
        if (det[2] > m_result_thresh_)
        {
            bbox[0] = std::max(0.f, det[3]*width);
            bbox[1] = std::max(0.f, det[4]*height);
            bbox[2] = std::min(width-1.f, det[5]*width);
            bbox[3] = std::min(height-1.f, det[6]*height);
            bbox[4] = det[2];
            output.emplace_back(bbox);

        } else{
            break;
        }
        det += 7;
    }
}
