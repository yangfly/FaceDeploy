#include <iostream>

#include "core/mtcnn.h"

using namespace cv;
using namespace std;
using namespace face;
using namespace caffe;

int main()
{
  system("mkdir -p out/detect");
  // config logging
  FLAGS_minloglevel = 0;
  FLAGS_log_dir = "out/detect";
  ::google::InitGoogleLogging("mtcnn");
  //::google::SetLogDestination(::google::ERROR, "detect_");
  ::google::InstallFailureSignalHandler();

  #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
  #else
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(CPU_ID);
  #endif

  unique_ptr<Mtcnn> mtcnn(new Mtcnn(MTCNN_MODEL_DIR));
  Mat image = imread("test/imgs/surveillance.png");
  #ifdef MTCNN_PRECISE_LANDMARK
    vector<FaceInfo> infos = mtcnn->Detect(image, true);
  #else // MTCNN_PRECISE_LANDMARK
    vector<FaceInfo> infos = mtcnn->Detect(image, false);
  #endif // MTCNN_PRECISE_LANDMARK
  for (auto & info : infos)
  {
    rectangle(image, Rect(info.bbox.x1, info.bbox.y1, info.bbox.x2 - info.bbox.x1, info.bbox.y2 - info.bbox.y1), Scalar(255, 0, 0), 2);
    for (auto & point : info.fpts)
      circle(image, Point2f(point.x, point.y), 2, Scalar(0, 255, 0), -1);
    cout << info.bbox.x1 << " " << info.bbox.y1 << " " << info.bbox.x2 << " " << info.bbox.y2 << " " << info.score << endl;
    cout << info.fpts[0].x << " " << info.fpts[1].x << " " << info.fpts[2].x << " " << info.fpts[3].x << " " << info.fpts[4].x << endl;
    cout << info.fpts[0].y << " " << info.fpts[1].y << " " << info.fpts[2].y << " " << info.fpts[3].y << " " << info.fpts[4].y << endl;
  }
  imwrite("out/detect/detect.png", image);

  // test align
  for (int i = 0; i < infos.size(); i++)
  {
    auto & fpts = infos[i].fpts;
    Mat face;
    face = mtcnn->Align(image, fpts, 96);
    imwrite("out/detect/face_" + to_string(i) + "_" + to_string(96) + ".png", face);
    face = mtcnn->Align(image, fpts, 112);
    imwrite("out/detect/face_" + to_string(i) + "_" + to_string(112) + ".png", face);
  }

	return 0;
}
