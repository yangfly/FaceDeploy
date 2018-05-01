// std
#include <vector>
#include <string>
#include <iostream>

#include "core/amface.h"

using namespace std;
using namespace cv;
using namespace face;

AMFace::AMFace(const string & model_dir)
{
	// Load the Net and Model.
	net = make_shared<caffe::Net<float>>(model_dir + "/AMFACE_r34.prototxt", caffe::TEST);
	net->CopyTrainedLayersFrom(model_dir + "/AMFACE_r34.caffemodel");
	// net = make_shared<caffe::Net<float>>(model_dir + "/CENTER.prototxt", caffe::TEST);
	// net->CopyTrainedLayersFrom(model_dir + "/CENTER.caffemodel");
	caffe::Blob<float>* input_layer = net->input_blobs()[0];
	input_size.height = input_layer->shape(2);
	input_size.width = input_layer->shape(3);

	// Check network input.
	CHECK(input_layer->channels() == 3) << "Network Input only support channel as 3.";
	CHECK(input_size.width == 96 || input_size.width == 112) << "Network Input only support width as 96 or 112.";
	CHECK(input_size.height == 112) << "Network Input only support height as 112.";
}

float AMFace::Similar(const Mat & feature1, const Mat & feature2)
{
	CHECK(feature1.cols == feature2.cols && feature1.rows == feature2.rows) << "Feature shape doesn't match";
	int len = feature1.cols;
	const float * data1 = feature1.ptr<float>(0);
	const float * data2 = feature2.ptr<float>(0);
	CHECK(data1 != data2) << "not same";
	float inp_ab = caffe::caffe_cpu_dot<float>(len, data1, data2);
	return 0.5 + 0.5 * inp_ab;
}

Mat AMFace::Forward(const Mat & face)
{
	SetBatchSize(1);
	Feed(face, 0);
	return move(ToMat(net->Forward()[0]));
}

Mat AMFace::Forward(const vector<Mat>& faces)
{
	int num = faces.size();
	Mat features;
	int i = 0, j;

	if (num > AMFACE_BATCH_SIZE)
	{
		SetBatchSize(AMFACE_BATCH_SIZE);
		for (; i <= num - AMFACE_BATCH_SIZE; i += AMFACE_BATCH_SIZE)
		{
			for (j = 0; j < AMFACE_BATCH_SIZE; j++)
				Feed(faces[i + j], j);
			features.push_back(ToMat(net->Forward()[0]));
		}
	}

	if (num - i > 0)
	{
		SetBatchSize(num - i);
		for (j = 0; j < num - i; j++)
			Feed(faces[i + j], j);
		features.push_back(ToMat(net->Forward()[0]));
	}

	return move(features);
}

float AMFace::Verify(const Mat & face1, const Mat& face2)
{
	vector<Mat> faces;
	faces.push_back(face1);
	faces.push_back(face2);
	Mat features = Forward(faces);
	return Similar(features.row(0), features.row(1));
}

void AMFace::SetBatchSize(const int batch_size) {
	caffe::Blob<float>* input_layer = net->input_blobs()[0];
	vector<int> input_shape = input_layer->shape();
	input_shape[0] = batch_size;
	input_layer->Reshape(input_shape);
	net->Reshape();
}

void AMFace::Feed(const Mat & face, int id)
{
	/* Feed image into input batch blob. */
	caffe::Blob<float>* input_layer = net->input_blobs()[0];
	int channels = input_layer->channels();
	int data_size = input_size.area() * channels;
	/* head pointer */
	float* input_data = input_layer->mutable_cpu_data() + id * data_size;
	std::vector<cv::Mat> input_channels;
	for (int i = 0; i < channels; ++i) {
		cv::Mat channel(input_size.height, input_size.width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += input_size.area();
	}
	/* Normalization: [0,255] -> [-1, 1] */
	cv::Mat normed;
	face.convertTo(normed, CV_32FC3, 1.0 / 128.0, -127.5 / 128.0);
    cvtColor(normed, normed, COLOR_BGR2RGB);
	cv::split(normed, input_channels);
}

Mat AMFace::ToMat(const caffe::Blob<float> * pBlob)
{
	int num = pBlob->num();
	int len = pBlob->channels();
	const float* data = pBlob->cpu_data();
	return move(Mat(num, len, CV_32FC1, const_cast<float*>(data)).clone());
}

