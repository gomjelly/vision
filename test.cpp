[화요일 오후 1:05] WooHyun Kim (김우현)
#include "vision.h"

#include <vector>

#include <fstream>
 
vision::vision(QWidget *parent)

    : QMainWindow(parent)

{

    webcamLabel = new QLabel();
 
    closeButton = new QPushButton("close");
 
    capture = VideoCapture(0);

    //face_cascade.load("D:\\extLibs\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
 
    net = dnn::readNetFromDarknet("D:\\01_Workspace\\vision\\yolov3.cfg", "D:\\01_Workspace\\vision\\yolov3.weights");

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
 
    std::ifstream classNamesFile("D:\\01_Workspace\\vision\\coco.names");

    if (!classNamesFile.is_open()) {

        std::cerr << "Error: Failed to open COCO names file: coco.names" << std::endl;

        return; // 또는 예외 발생

    }

    std::string line;

    while (getline(classNamesFile, line)) {

        classNames.push_back(line);

    }
 
    classNamesFile.close();

 
    QGridLayout* layout = new QGridLayout();

    layout->addWidget(webcamLabel, 0, 0);

    layout->addWidget(closeButton, 1, 1);
 
    QWidget* centralWidget = new QWidget();

    centralWidget->setLayout(layout);
 
    QTimer* timer = new QTimer(this);

    connect(timer, &QTimer::timeout, this, &vision::updateFrame);

    timer->start(33); // 30 fps
 
    connect(closeButton, &QPushButton::clicked, this, &QMainWindow::close);
 
    this->setCentralWidget(centralWidget);

    this->setMinimumSize(800, 500);

}
 
vision::~vision()

{}
 
void vision::updateFrame() {

    Mat frame;

    capture >> frame;
 
    //if (!frame.empty()) {

    //    Mat gray_frame;

    //    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
 
    //    std::vector<Rect> faces;

    //    face_cascade.detectMultiScale(gray_frame, faces, 1.1, 3, 0, Size(30, 30));
 
    //    for (Rect face : faces) {

    //        rectangle(frame, face, Scalar(0, 255, 0), 2);

    //    }
 
    //    QImage image = QImage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);

    //    webcamLabel->setPixmap(QPixmap::fromImage(image));

    //}

    const int NUM_CLASSES = 1;

    const float NMS_THRESHOLD = 0.4;

    const cv::Scalar colors[] = {

    {0, 255, 255},

    {255, 255, 0},

    {0, 255, 0},

    {255, 0, 0}

    };

    const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);
 
    if (!frame.empty()) {

        auto output_names = net.getUnconnectedOutLayersNames();

        cv::Mat blob;

        std::vector<cv::Mat> detections;

        std::vector<int> indices[NUM_CLASSES];

        std::vector<cv::Rect> boxes[NUM_CLASSES];

        std::vector<float> scores[NUM_CLASSES];
 
        cv::resize(frame, frame, cv::Size(416, 416), cv::INTER_LINEAR);
 
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(frame.rows, frame.cols), cv::Scalar(), true, false, CV_32F);

        net.setInput(blob);

        net.forward(detections, output_names);
 
        // object detection using YOLOV4

        for (auto& output : detections) {

            const auto num_boxes = output.rows;

            for (int i = 0; i < num_boxes; i++) {

                //calculo das 5 predições para cada bounding box: x, y, w, h , confiança

                auto x = output.at<float>(i, 0) * frame.cols;

                auto y = output.at<float>(i, 1) * frame.rows;

                auto width = output.at<float>(i, 2) * frame.cols;

                auto height = output.at<float>(i, 3) * frame.rows;

                cv::Rect rect(x - width / 2, y - height / 2, width, height);
 
                for (int c = 0; c < NUM_CLASSES; c++) {

                    auto confidence = *output.ptr<float>(i, 5 + c);

                    if (confidence >= 0.4) {

                        boxes[c].push_back(rect);

                        scores[c].push_back(confidence);

                    }

                }

            }

        }

        // Realiza a supressão não máxima das bounding boxes e das pontuações  de confiança correspondentes.

        // eliminação de bounding boxes repetidas que identificam o mesmo objecto.

        for (int c = 0; c < NUM_CLASSES; c++)

            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
 
        // identificação dos objectos e correspondentes pontuações de confiança através de bounding boxes. 

        for (int c = 0; c < NUM_CLASSES; c++) {

            for (size_t i = 0; i < indices[c].size(); ++i) {

                const auto color = colors[c % NUM_COLORS];
 
                auto idx = indices[c][i];

                const auto& rect = boxes[c][idx];

                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);
 
                // coloco a identificação da classe do objeto contido na bounding box - pedestre ou garrafa por ex.

                std::ostringstream label_ss;

                label_ss << classNames[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];

                auto label = label_ss.str();
 
                int baseline;

                auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);

                // defino o rectangulo que define o objeto detectado

                cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);

                // coloco a identificação da classe do objecto detectado.

                cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));

            }

        }
 
        //auto blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(416, 416), Scalar(0, 0, 0), true, false);

        //net.setInput(blob);
 
        //std::vector<Mat> outs;

        //net.forward(outs);
 
        //    outs->detect(frame, classIds, scores, detections, 0.5, 0.4);
 
        //// 감지된 객체 표시

        //Mat visFrame = frame.clone();

        //for (int i = 0; i < detections.size(); ++i) {

        //    int id = classIds[i];

        //    float confidence = scores[i];

        //    Rect box = detections[i];
 
        //    if (confidence > 0.5) {

        //        rectangle(visFrame, box, Scalar(0, 0, 255), 2);

        //        putText(visFrame, classNames[id] + ": " + std::to_string(confidence), box.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);

        //    }

        //}
 
        // 결과 영상을 QLabel에 표시

        QImage image = QImage(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_BGR888);

        webcamLabel->setPixmap(QPixmap::fromImage(image));

    }

}