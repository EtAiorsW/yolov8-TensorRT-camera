#include "yolov8_TensorRT.h"
#include "Logger.h"
#include "read_csv.h"
#include "PostProcess.h"
#include <cmath>
#include <chrono>
#include <filesystem>


void onnxToEngine(const char* onnxFile, int bit, int memorySize)
{
	std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//获取文件路径
	std::string modelName = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//获取不带后缀的文件名名
	std::string engineFile = modelPath + modelName_ +  '_' + std::to_string(bit) + ".engine";

	// Logger类，用于记录日志
	Logger logger;
	// 构建器，获取cuda内核目录以获取最快的实现
	// 用于创建config、network、engine的其他对象的核心类
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	// 定义网络属性
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// 解析onnx网络文件
	// tensorRT模型类
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx文件解析类
	// 将onnx文件解析，并填充tensorRT网络结构
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	// 解析onnx文件
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) 
	{
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// 创建推理引擎
	// 创建生成器配置对象。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 设置最大工作空间大小(字节单位)
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);

	// 设置模型输出精度
	switch (bit)
	{
	case 8:  config->setFlag(nvinfer1::BuilderFlag::kINT8); break;
	case 16: config->setFlag(nvinfer1::BuilderFlag::kFP16); break;
	case 32: config->setFlag(nvinfer1::BuilderFlag::kTF32); break;
	}
	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// 将推理引擎保存到本地
	std::cout << "try to save engine file now" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	// 将模型转化为文件流数据
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	// 将文件保存到本地
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	// 销毁创建的对象
	modelStream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}


void onnxToEngineDynamicShape(const char* onnxFile, int bit, int memorySize, const char* nodeName, int* minShapes, int* optShapes, int* maxShapes)
{

}



void nvinferInit(const char* engineFile, NvinferStruct** ptr)
{
	std::ifstream filePtr(engineFile, std::ios::binary);
	if (!filePtr.good())
	{
		std::cerr << "read engine file failed!" << std::endl;
		return;
	}

	size_t size = 0;
	filePtr.seekg(0, filePtr.end);
	size = filePtr.tellg();
	filePtr.seekg(0, filePtr.beg);
	char* modelStream = new char[size];
	filePtr.read(modelStream, size);
	filePtr.close();

	Logger logger;
	NvinferStruct* p = new NvinferStruct();
	p->runtime = nvinfer1::createInferRuntime(logger);
	p->engine = p->runtime->deserializeCudaEngine(modelStream, size);
	p->context = p->engine->createExecutionContext();
	cudaStreamCreate(&(p->stream));
	int numBindings = p->engine->getNbBindings();
	p->dataBuffer = new void* [numBindings];
	delete[] modelStream;

	for (int i = 0; i < numBindings; i++)
	{
		nvinfer1::Dims dims = p->engine->getBindingDimensions(i);
		nvinfer1::DataType type = p->engine->getBindingDataType(i);
		std::vector<int> shape(dims.d, dims.d + dims.nbDims);
		size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
		switch (type)
		{
		case nvinfer1::DataType::kINT32:
		case nvinfer1::DataType::kFLOAT: size *= 4; break;
		case nvinfer1::DataType::kHALF: size *= 2; break;
		case nvinfer1::DataType::kBOOL:
		case nvinfer1::DataType::kINT8:
		default:break;
		}
		cudaMalloc(&(p->dataBuffer[i]), size);
	}
	*ptr = p;
}


void copyHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex);
	size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
	cudaMemcpyAsync(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice, ptr->stream);
}

void nvinferInference(NvinferStruct* ptr)
{
	ptr->context->enqueueV2(ptr->dataBuffer, ptr->stream, nullptr);
}

void copyDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	nvinfer1::Dims dims = ptr->engine->getBindingDimensions(nodeIndex);
	size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
	cudaMemcpyAsync(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost, ptr->stream);
	cudaStreamSynchronize(ptr->stream);
}

void nvinferDelete(NvinferStruct* ptr)
{
	int numBindings = ptr->engine->getNbBindings();
	for (int i = 0; i < numBindings; i++)
	{
		cudaFree(ptr->dataBuffer[i]);
		ptr->dataBuffer[i] = nullptr;
	}
	delete ptr->dataBuffer;
	ptr->dataBuffer = nullptr;
	ptr->context->destroy();
	ptr->engine->destroy();
	ptr->runtime->destroy();
	cudaStreamDestroy(ptr->stream);
	delete ptr;
}

void preProcess(cv::Mat* img, int length, float* factor, std::vector<float>& data)
{
	cv::Mat mat;
	int rh = img->rows;
	int rw = img->cols;
	int rc = img->channels();
	cv::cvtColor(*img, mat, cv::COLOR_BGR2RGB);
	int maxImageLength = rw > rh ? rw : rh;
	cv::Mat maxImage = cv::Mat::zeros(maxImageLength, maxImageLength, CV_8UC3);
	maxImage = maxImage * 255;
	cv::Rect roi(0, 0, rw, rh);
	mat.copyTo(cv::Mat(maxImage, roi));
	cv::Mat resizeImg;
	cv::resize(maxImage, resizeImg, cv::Size(length, length), 0.0f, 0.0f, cv::INTER_LINEAR);
	*factor = (float)((float)maxImageLength / (float)length);
	resizeImg.convertTo(resizeImg, CV_32FC3, 1 / 255.0);
	rh = resizeImg.rows;
	rw = resizeImg.cols;
	rc = resizeImg.channels();
	for (int i = 0; i < rc; ++i)
	{
		cv::extractChannel(resizeImg, cv::Mat(rh, rw, CV_32FC1, data.data() + i * rh * rw), i);
	}
}


cv::Mat drawResult(cv::Mat& img, std::vector<DetectionResult>& results)
{
	for (auto& result : results)
	{
		cv::rectangle(img, result.bbox, cv::Scalar(0, 255, 0), 2);
		cv::putText(img, result.className, cv::Point(result.bbox.x, result.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
	}
	return img;
}

int main()
{
	const char* onnxFile = "/home/wanggq/ubuntu/DL/TensorRT/yolov8/yolov8n.onnx";
	const char* engineFile = "/home/wanggq/ubuntu/DL/TensorRT/yolov8/yolov8n_16.engine";
	const char* labelpath = "/home/wanggq/ubuntu/DL/TensorRT/yolov8/coco.txt";
	
	//Convert onnx to engine
	// onnxToEngine(onnxFile, 16, 1024);
	// std::cout << "convert onnx to engine successfully!" << std::endl;

	
	// Initialize TensorRT
	std::cout << "---------start initialize TensorRT---------" << std::endl;
	NvinferStruct*p = new NvinferStruct();
	NvinferStruct** ptr = &p;
	nvinferInit(engineFile, ptr);


	float factor = 0;
	std::vector<float> inputs(shape * shape * 3);
	float* outputs = new float[outputLength * (num_classes + 4)];

	// Postprocess Init
	PostProcess postprocess;
	postprocess.readClassNames(labelpath);

	// Read Camear
	cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return 1;
    }

	while(true)
	{
		cv::Mat frame;
        if (!capture.read(frame)) {
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        preProcess(&frame, shape, &factor, inputs);
        copyHostToDeviceByIndex(p, 0, inputs.data());
		nvinferInference(p);
		copyDeviceToHostByIndex(p, 1, outputs);

		postprocess.factor = (float)factor;
		std::vector<DetectionResult> results = postprocess.postProcess(outputs);

        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        int fps = int(1000.0 / time);
        std::cout << "fps: " << std::right << std::setw(3) << fps << "  time: " << std::right << std::setw(2) << time << "ms" << std::endl;
		
		cv::Mat resultImage = drawResult(frame, results);
        cv::imshow("frame", frame);
        cv::waitKey(1);
	}
	cv::destroyAllWindows();

	// Release nvinfer struct memory
	nvinferDelete(*ptr);
	std::cout << "delete nvinfer struct" << std::endl;
	p = nullptr;

	return 0;
}

