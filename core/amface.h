#ifndef FACE_AMFACE_HPP_
#define FACE_AMFACE_HPP_

// caffe
#include <caffe/caffe.hpp>
// OpenCV
#include <opencv2/opencv.hpp>
// std
#include <memory>

#include "core/common.h"
#include "core/config.h"

namespace face
{

// #define USE_OPENMP
// #ifdef USE_OPENMP
// 	#include <omp>
// 	#define _NUM_THREADS 4
// #endif

class AMFace
{
public:
	/// @brief Constructor.
	AMFace(const std::string & model_dir);
	/// @brief Cosine similarity between two features.
	float Similar(const cv::Mat & feature1, const cv::Mat & feature2);
	/// @brief Forward single face and get feature.
	cv::Mat Forward(const cv::Mat & face);
	/// @brief Forward multi-faces and get features.
	cv::Mat Forward(const std::vector<cv::Mat> & faces);
	/// @brief Verify between two images
	float Verify(const cv::Mat & face1, const cv::Mat& face2);

	/// @brief Tool variables
	cv::Size input_size;

private:
	/// @brief Behind network.
	std::shared_ptr<caffe::Net<float>> net;

	/// @brief Set batch size of network.
	void SetBatchSize(int batch_size);
	/// @brief Feed image into net (wrapper of WarpInputLayer)
	void Feed(const cv::Mat & face, int id);
	/// @brief Convert Blob to Mat
	cv::Mat ToMat(const caffe::Blob<float> * pBlob);
};

} // namespace face

#endif // FACE_AMFACE_HPP_