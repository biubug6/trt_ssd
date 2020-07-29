#include <iostream>
#include "../detect/personDetect.h"

int main()
{
    // init model
    std::string model_path = "../model/ssd.engine";
    Parameter param(model_path.c_str());
    Detect detector(param);

    // infer
    std::string img_path = "../img/000000013291.jpg";
    cv::Mat img = cv::imread(img_path.c_str(), cv::IMREAD_COLOR);
    std::vector<std::vector<float>> res;
    detector.infer(img, res);

    for (int i = 0; i < res.size(); ++i) {
        int x1 = res[i][0];
        int y1 = res[i][1];
        int x2 = res[i][2];
        int y2 = res[i][3];
        float conf = res[i][4];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255));
        cv::putText(img, std::to_string(conf), cv::Point(x1, y1+20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
    }
    cv::imwrite("res.jpg", img);
    return 0;
}
