#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

const int shape = 640;
const int num_classes = 80;
const int outputLength = 8400;

struct DetectionResult
{
	cv::Rect bbox;
	float confidence;
	std::string className;
	DetectionResult(cv::Rect bbox, float confidence, std::string className) : bbox(bbox), confidence(confidence), className(className) {}
};


class PostProcess
{
public:
	std::vector<std::string> classNames;
	float factor;
	void readClassNames(const std::string filename);
	std::vector<DetectionResult> postProcess(float* inferResult);
};

