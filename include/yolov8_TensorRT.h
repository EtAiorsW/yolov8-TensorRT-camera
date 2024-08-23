#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h"
#include "PostProcess.h"

typedef struct tensorRT_nvinfer {
	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	void** dataBuffer;
	cudaStream_t stream;
} NvinferStruct;

void onnxToEngine(const char* onnxFile, int bit, int memorySize);

void onnxToEngineDynamicShape(const char* onnxFile, int bit, int memorySize, const char* nodeName, int* minShapes, int* optShapes, int* maxShapes);

void nvinferInit(const char* engineFile, NvinferStruct **ptr);

void nvinferInitDynamicShape(const char* engineFile, NvinferStruct** ptr, int maxBatchSize);

void copyHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data);

void nvinferInference(NvinferStruct* ptr);

void copyDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data);

void nvinferDelete(NvinferStruct* ptr);

void preProcess(cv::Mat* img, int length, float* factor, std::vector<float>& data);

cv::Mat drawResult(cv::Mat& img, std::vector<DetectionResult>& results);
