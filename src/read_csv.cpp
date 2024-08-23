#include "read_csv.h"
#include <algorithm>

void read_csv(const char* filepath, float* input)
{
	std::ifstream file(filepath);
	std::string line;

	if (file.is_open())
	{
		std::getline(file, line);
		for (int i = 0; i < 300; i++)
		{
			std::getline(file, line);
			std::stringstream ss(line);
			std::string field;
			if (std::getline(ss, field, ','))
			{
				if (std::getline(ss, field, ','))
				{
					input[i] = std::stof(field);
				}
			}
		}
		file.close();
	}
	float maxVal = *std::max_element(input, input + 300);
	for (int i = 0; i < 300; i++)
	{
		input[i] /= maxVal;
	}
}