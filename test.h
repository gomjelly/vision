#pragma once
 
#include <QtWidgets>

#include <opencv2/opencv.hpp>
 
using namespace cv;
 
class vision : public QMainWindow

{

    Q_OBJECT
 
public:

    vision(QWidget *parent = nullptr);

    ~vision();
 
private:

    QLabel* webcamLabel;

    QPushButton* closeButton;
 
    VideoCapture capture;
 
    CascadeClassifier face_cascade;

    cv::dnn::Net net;

    std::vector<String> classNames;
 
    void updateFrame();

};
 