#ifndef PERSON_DETECT_H
#define PERSON_DETECT_H
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "../engine/engine.h"

struct Parameter
{
    Parameter(std::string model_path, char* output_name = "detection_out",
                        int output_size = 700, int width = 640, int height = 360, int device = 0,
                        cv::Point3f mean=cv::Point3f(104, 117, 123), float resultThresh = 0.5):
            model_path(model_path), output_name(output_name), output_size(output_size),width(width), height(height),
            device(device), mean(mean), resultThresh(resultThresh)
    {};
    std::string model_path;
    std::string output_name;
    int output_size;
    int width;
    int height;
    int device;
    cv::Point3f mean;
    float resultThresh;            // threshold
};

class Detect
{
private:
    float m_result_thresh_;
    cv::Point3f m_mean_;
    float* m_input_;
    std::unique_ptr<Engine> kernel;

public:
    Detect(Parameter& param);
    ~Detect();

    void infer(const cv::Mat &img, std::vector<std::vector<float>> &bboxs);

    void decode(std::vector<std::vector<float>>& output, int width, int height);
};

#endif //PERSON_DETECT_H
