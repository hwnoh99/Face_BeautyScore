#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>  

using namespace cv;
using namespace std;

int main()
{

    //read an image
    cv::Mat img = cv::imread("model2.jpg");
    if (img.empty())
    {
        std::cout << std::endl << std::endl << "Could not read the image: " << "image_path" << std::endl;
        return 1;
    }

    cv::Mat gray;
    cvtColor(img, gray, cv::COLOR_RGB2GRAY);

    //load Cascade data for face and eyes detection
    CascadeClassifier face_cascade;
    CascadeClassifier eye_cascade;
    face_cascade.load("C:/Users/coela/downloads/frontalFace10/haarcascade_frontalface_alt.xml");
    eye_cascade.load("C:/Users/coela/downloads/frontalEyes35x16XML/frontalEyes35x16.xml");

    if (face_cascade.empty() || eye_cascade.empty())
    {
        cout << std::endl << std::endl << "Error Loading XML File" << endl;
        return 1;
    }

    //Face 
    // 
    //detect a face in image
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces);

    if (faces.empty())
    {
        cout << std::endl << std::endl << "Could not find the face" << endl;
        return 1;
    }
      
    //set the centre of the face and ensure that face is detected correctly
    Point faceCenter(faces[0].x + faces[0].width * 0.5, faces[0].y + faces[0].height * 0.5);
    cout << faces[0].width << faces[0].height << endl;
    ellipse(img, faceCenter, Size(faces[0].width * 0.5, faces[0].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

    //Eye
    //
    //detect a eye in image
    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(gray, eyes);

    //set the centre of the eye 
    Point eyeCenter(eyes[0].x + eyes[0].width * 0.5, eyes[0].y + eyes[0].height * 0.5);
    ellipse(img, eyeCenter, Size(eyes[0].width * 0.5, eyes[0].height * 0.5), 0, 0, 360, Scalar(128, 0, 128), 2, 4, 0);

    //compute the vertical distance between upper forehead and eyes centre ratio and it's SCORE(0.0 ~ 10.0). 
    double upperEye = eyeCenter.y - faces[0].y;
    double underEye = faces[0].y + faces[0].height - eyeCenter.y;

    double dif = abs(0.75 - (upperEye / underEye)); //score: how similar to 0.75;

    double score = (0.5 - dif) * 20;
    if (score < 0)
        score = 0.0;

    cout << std::endl << std::endl << score << endl;

    //create image window
    cv::namedWindow("Image Window");
    cv::imshow("Image Window", img);

    waitKey(0);

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu

