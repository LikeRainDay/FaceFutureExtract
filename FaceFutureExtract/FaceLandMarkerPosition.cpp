//设置人脸的标记点
#include <dlib\opencv.h>
#include <opencv2\opencv.hpp>
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing.h>
#include <dlib\gui_widgets.h>

//声明dlib的域
using namespace dlib;

using namespace std;

int main() {

	try {
		//首先进行获取摄像头
		cv::VideoCapture cap(0);

		if (!cap.isOpened()) {
			//如果摄像头没有开启
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}
		//Load face detection and pos estimation models 加载我们需要的脸部识别和姿态估计模型
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pos_modle;
		//将文件中的模型放置再pos_modle中
		deserialize("shape_predictor_68_face_landmarks.dat") >> pos_modle;

		//Grab and process frames until the main window is closed by the user
		//处理当前每一帧的图片
		while (cv::waitKey(30)!=27)
		{

			//Grab a frame 获取一帧
			cv::Mat temp;
			//将摄像头获取的当前帧图片放入到 中间文件中
			cap >> temp;
			//将其转化为RGB像素图片
			cv_image<bgr_pixel> cimg(temp);
			//开始进行脸部识别
			std::vector<rectangle> faces = detector(cimg);
			//发现每一个脸的pos估计 Find the pose of each face
			std::vector<full_object_detection> shapes;
			unsigned faceNumber=	faces.size();
			for (unsigned i = 0; i < faceNumber; i++)
				shapes.push_back(pos_modle(cimg, faces[i]));
			if (!shapes.empty()) {
				int faceNumber = shapes.size();
				for (int j = 0; j < faceNumber; j++)
				{
					for (int i = 0; i < 68; i++)
					{
						cv::circle(temp, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
						cv::putText(temp,to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), CV_FONT_HERSHEY_PLAIN,1, cv::Scalar(0, 0, 255));
						
					}
				}
			}
			//Display it all on the screen
			cv::imshow("Dlib标记", temp);
		}

	}
	catch (serialization_error &e) {
		cout << "You need dlib's default face landmarking file to run this example.(你需要添加landmark的bat文件，才可以跑这个实例)" << endl;
			cout << endl << e.what() << endl;
	}
	catch(exception &e){
		cout <<  e.what() << endl;

	}



}

