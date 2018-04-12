#ifndef FACE_MTCNN_HPP_
#define FACE_MTCNN_HPP_

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

// Only do normalization once before any image processing to save time.
/// #define NORM_FARST

/// @brief face regression
using Reg = BBox;

/// @brief face proposal
struct Proposal : public FaceInfo
{
	Reg reg;	// face regression
	Proposal(BBox&& bbox, float score, Reg&& reg) :
    FaceInfo(std::move(bbox), score), reg(std::move(reg)) {}
	Proposal(BBox&& bbox, float score, FPoints&& fpts, Reg&& reg) :
    FaceInfo(std::move(bbox), score, std::move(fpts)), reg(std::move(reg)) {}
};

class Mtcnn
{
public:
	/// @brief Constructor.
	Mtcnn(const std::string & model_dir);
	/// @brief Detect faces from images
	std::vector<FaceInfo> Detect(const cv::Mat & sample, bool precise_landmark);
	/// @brief Align image with facial points, with cp2tform and cv::warpAffine
	/// @param width: 96 or 112
	cv::Mat Align(const cv::Mat & sample, const FPoints & fpts, int width = 96);

private:
  enum NMS_TYPE {
		IoM,	// Intersection over Union
		IoU		// Intersection over Minimum
	};

	// networks
	std::shared_ptr<caffe::Net<float>> Pnet;
	std::shared_ptr<caffe::Net<float>> Rnet;
	std::shared_ptr<caffe::Net<float>> Onet;
	#ifdef MTCNN_PRECISE_LANDMARK
	  std::shared_ptr<caffe::Net<float>> Lnet;
  #endif // MTCNN_PRECISE_LANDMARK

	// reference standard facial points
	FPoints ref_96_112;
	FPoints ref_112_112;

  /// @brief Set batch size of network.
	void SetBatchSize(std::shared_ptr<caffe::Net<float> > net, const int batch_size);
	/// @brief Warp whole input layer into cv::Mat channels.
	std::vector<std::vector<cv::Mat> > WarpInputLayer(std::shared_ptr<caffe::Net<float> > net);
	/// @brief Create scale pyramid: down order
	std::vector<float> ScalePyramid(const int min_len);
	/// @brief Get bboxes from maps of confidences and regressions.
	std::vector<Proposal> GetCandidates(const float scale,
		const caffe::Blob<float>* regs, const caffe::Blob<float>* scores);
	/// @brief Non Maximum Supression with type 'IoU' or 'IoM'.
	std::vector<Proposal> NonMaximumSuppression(std::vector<Proposal>& pros,
		const float threshold, const NMS_TYPE type);
	/// @brief Refine bounding box with regression.
	void BoxRegression(std::vector<Proposal>& pros);
	/// @brief Convert bbox from float bbox to square.
	void Square(std::vector<BBox> & bboxes);
	void Square(BBox & bbox);
	/// @brief Crop proposals with padding 0.
	cv::Mat CropPadding(const cv::Mat& sample, const BBox& bbox);
	/// @brief Ensure faceInfo inside image.
	// void EnsureInside(const cv::Mat& sample, std::vector<FaceInfo>& infos);

	/// @brief Stage 1: Pnet get proposal bounding boxes
	std::vector<BBox> ProposalNetwork(const cv::Mat& sample);
	/// @brief Stage 2: Rnet refine and reject proposals
	std::vector<BBox> RefineNetwork(const cv::Mat& sample, std::vector<BBox>& bboxes);
	/// @brief Stage 3: Onet refine and reject proposals and regress facial landmarks.
	std::vector<FaceInfo> OutputNetwork(const cv::Mat& sample, std::vector<BBox>& bboxes);
	/// @brief Stage 4: Lnet refine facial landmarks
	#ifdef MTCNN_PRECISE_LANDMARK
		void LandmarkNetwork(const cv::Mat& sample, std::vector<FaceInfo>& infos);
	#endif // MTCNN_PRECISE_LANDMARK

	/// @brief cp2tform matlab export to c++
	/// @param type: 'similarity' or 'nonreflective similarity'
	cv::Mat cp2tform(const FPoints & src, const FPoints & dst, const char * type="similarity");
};	// class MTCNN

} // namespace face

#endif // FACE_MTCNN_HPP_