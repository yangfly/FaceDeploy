#include <iostream>
#include <chrono>

#include "core/mtcnn.h"
#include "core/amface.h"

using namespace cv;
using namespace std;
using namespace face;
using namespace caffe;

unique_ptr<Mtcnn> mtcnn(nullptr);
unique_ptr<AMFace> amface(nullptr);


/// @brief A tool timer
class Timer {
  using Clock = std::chrono::high_resolution_clock;
public:
  /// @brief start or restart timer
  inline void Tic() {
    start_ = Clock::now();
  }
  /// @brief stop timer
  inline void Toc() {
    end_ = Clock::now();
  }
  /// @brief return time in ms
  inline double Elasped() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    return duration.count();
  }

private:
  Clock::time_point start_, end_;
};

vector<Mat> time_detect(const char* name, int times)
{
  cout << "[Test] feature detecting begin ---------------" << endl;
  ::Timer timer;
  Mat image = imread(name);
  CHECK(!image.empty()) << name;
  vector<FaceInfo> infos;
  for (int i = 0; i < times; i++)
  {
    timer.Tic();
    #ifdef MTCNN_PRECISE_LANDMARK
      infos = mtcnn->Detect(image, true);
    #else // MTCNN_PRECISE_LANDMARK
      infos = mtcnn->Detect(image, false);
    #endif // MTCNN_PRECISE_LANDMARK
    timer.Toc();
    cout << "[" << image.cols << "x" <<image.rows << "] : " << fixed << setprecision(4) << timer.Elasped() << " ms" << endl;
  }
  vector<Mat> faces;
  for (const auto & info : infos)
  {
    Mat face = mtcnn->Align(image, info.fpts, amface->input_size.width);
    faces.push_back(move(face));
  }
  return move(faces);
}

Mat time_extract(const vector<Mat> & faces_, int max_)
{
  cout << "[Test] feature extracting begin ---------------" << endl;
  CHECK(!faces_.empty()) << "containing no face";
  vector<Mat> faces;
  Mat features;
  ::Timer timer;

  timer.Tic();
  amface->Forward(faces_[0]);
  timer.Toc();
  cout << 1 << " : " << fixed << setprecision(4) << timer.Elasped() << " ms" << endl;

  int j = 0;
  for (int i = 1; i <= max_; i++)
  {
    faces.push_back(faces_[j++].clone());
    timer.Tic();
    Mat feature = amface->Forward(faces);
    timer.Toc();
    cout << i << " : " << fixed << setprecision(4) << timer.Elasped() << " ms" << endl;
    if (j >= faces_.size())
      j = 0;
    if (features.rows < std::min(feature.rows, (int)faces_.size()))
      features = feature.clone();
  }
  return move(features);
}

double time_similar(const Mat & features)
{
  cout << "[Test] feature similar begin ---------------" << endl;
  CHECK(features.rows >= 2) << "no enough features";
  ::Timer timer;
  double sim;
  timer.Tic();
  sim = amface->Similar(features.row(0), features.row(1));
  timer.Toc();
  cout << " Similar : " << fixed << setprecision(4) << timer.Elasped() << " ms" << endl;
  return sim;
}

int main()
{
  system("mkdir -p out/time");
  // config logging
  FLAGS_minloglevel = 0;
  FLAGS_log_dir = "out/time";
  ::google::InitGoogleLogging("time");
  ::google::InstallFailureSignalHandler();

  #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
  #else
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(CPU_ID);
  #endif

  mtcnn.reset(new Mtcnn(MTCNN_MODEL_DIR));
  amface.reset(new AMFace(AMFACE_MODEL_DIR));
  
  vector<Mat> faces;
  faces = time_detect("test/imgs/surveillance.png", 3);
  cout << "face num: " << faces.size() << endl;
  Mat features = time_extract(faces, 20);
  cout << "features shape: " << features.cols << " x " << features.rows << endl;
  double sim = time_similar(features);
  cout << "similarity: " << sim << endl;

  return 0;
}