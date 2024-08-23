#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept override
	{
		// suppress info-level messages
		//if (severity <= Severity::kINFO) {
		//	std::cout << msg << std::endl;
		//}
		std::cout << msg << std::endl;
	}
};
