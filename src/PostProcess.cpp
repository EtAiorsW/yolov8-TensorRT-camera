#include "PostProcess.h"

void PostProcess::readClassNames(const std::string filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "ERROR: read class names file failed!" << std::endl;
		exit(1);
	}
	std::string line;
	while (std::getline(file, line))
	{
		classNames.push_back(line);
	}
	file.close();
}

std::vector<DetectionResult> PostProcess::postProcess(float* inferResult)
{
	cv::Mat output = cv::Mat((num_classes + 4), outputLength, CV_32F, inferResult);
	cv::transpose(output, output);

	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	for (int i = 0; i < output.rows; i++)
	{
		cv::Mat confidences_all = output.row(i).colRange(4, (num_classes + 4));
		cv::Point classIdsPoint;
		double confidence;
		cv::minMaxLoc(confidences_all, 0, &confidence, 0, &classIdsPoint);
		if (confidence > 0.25)
		{
			cv::Rect box;
			float cx = output.at<float>(i, 0);
			float cy = output.at<float>(i, 1);
			float ow = output.at<float>(i, 2);
			float oh = output.at<float>(i, 3);
			box.x = static_cast<int>((cx - 0.5 * ow) * factor);
			box.y = static_cast<int>((cy - 0.5 * oh) * factor);
			box.width = static_cast<int>(ow * factor);
			box.height = static_cast<int>(oh * factor);

			boxes.push_back(box);
			classIds.push_back(classIdsPoint.x);
			confidences.push_back(confidence);
		}
	}

	// nms
	std::vector<int> indexes;
	std::vector<DetectionResult> results;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		DetectionResult result(boxes[index], confidences[index], classNames[classIds[index]]);
		results.push_back(result);
		// std::cout << boxes[index] << '_' << confidences[index] << '_' << classNames[classIds[index]] << std::endl;
	}
	return results;
}