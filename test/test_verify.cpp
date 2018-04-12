#include <iostream>

#include "core/mtcnn.h"
#include "core/amface.h"

using namespace cv;
using namespace std;
using namespace face;
using namespace caffe;

static int width;

Mat detect(const unique_ptr<Mtcnn> & mtcnn, const char* name)
{
  Mat image = imread(name);
  CHECK(!image.empty()) << name;
  #ifdef MTCNN_PRECISE_LANDMARK
    vector<FaceInfo> infos = mtcnn->Detect(image, true);
  #else // MTCNN_PRECISE_LANDMARK
    vector<FaceInfo> infos = mtcnn->Detect(image, false);
  #endif // MTCNN_PRECISE_LANDMARK
  CHECK(infos.size() == 1) << "Detected faces num not equal to 1" << name;
  Mat face = mtcnn->Align(image, infos[0].fpts, width);
  return move(face);
}

int main()
{
  system("mkdir -p out/verify");
  // config logging
  FLAGS_minloglevel = 0;
  FLAGS_log_dir = "out/verify";
  ::google::InitGoogleLogging("amface");
  ::google::InstallFailureSignalHandler();

  #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
  #else
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(CPU_ID);
  #endif

  unique_ptr<Mtcnn> mtcnn(new Mtcnn(MTCNN_MODEL_DIR));
  unique_ptr<AMFace> amface(new AMFace(AMFACE_MODEL_DIR));
  width = amface->input_size.width;

  Mat face11 = detect(mtcnn, "test/imgs/cdy_cdy_0_01.jpg");
  imwrite("out/verify/face11.png", face11);
  Mat face12 = detect(mtcnn, "test/imgs/cdy_cdy_0_02.jpg");
  imwrite("out/verify/face12.png", face12);
  Mat face21 = detect(mtcnn, "test/imgs/4404A.jpg");
  imwrite("out/verify/face21.png", face21);
  Mat face22 = detect(mtcnn, "test/imgs/4404B.jpg");
  imwrite("out/verify/face22.png", face22);

  // unique_ptr<AMFace> amface(nullptr);
  // amface.reset(new AMFace(AMFACE_MODEL_DIR));
  cv::Mat feat11 = amface->Forward(face11);
  cv::Mat feat12 = amface->Forward(face12);
  cv::Mat feat21 = amface->Forward(face21);
  cv::Mat feat22 = amface->Forward(face22);
  cout << "feature shape: " << feat11.rows << " x " << feat11.cols << endl;
  float * data = feat11.ptr<float>(0);
  cout << "feat11 left 5 :" << data[0] << " "
                             << data[1] << " "
                             << data[2] << " "
                             << data[3] << " "
                             << data[4] << endl;

  cout << "similarity:" << endl;
  cout << "11 vs 12 : " << amface->Similar(feat11, feat12) << endl;
  cout << "21 vs 22 : " << amface->Similar(feat21, feat22) << endl;

  cout << "11 vs 21 : " << amface->Similar(feat11, feat21) << endl;
  cout << "11 vs 22 : " << amface->Similar(feat11, feat22) << endl;
  cout << "12 vs 21 : " << amface->Similar(feat12, feat21) << endl;
  cout << "12 vs 22 : " << amface->Similar(feat12, feat22) << endl;

  return 0;
}