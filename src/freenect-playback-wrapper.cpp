#include <iostream>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <thread>
#include "freenect-playback-wrapper.h"

using namespace std;

FreenectPlaybackWrapper::FreenectPlaybackWrapper(std::string FreenectVideoFolder)
	: freenect_video_folder(FreenectVideoFolder)
{
	std::replace(freenect_video_folder.begin(), freenect_video_folder.end(), '\\', '/');

	std::string index = freenect_video_folder + "/INDEX.txt";

	reader.open(index);

	if (!reader.is_open())
	{
		cout << "Unable to find INDEX.txt inside folder: " << freenect_video_folder << endl;
		exit(1);
	}
}
bool IsLittleEndian()
{
	union
	{
		uint16_t i;
		uint8_t c[2];
	} testValue;

	testValue.i = 1;
	return testValue.c[0] == 1;
}

uint8_t FreenectPlaybackWrapper::GetNextFrame()
{
	if (finished)
		return 0;

	string line;

	bool updatedRGB = false;
	bool updatedDepth = false;

	string lastPath = "";

	if (previous_line != "")
		line = previous_line;
	else
		getline(reader, line);

	do
	{
		if (line[0] == 'a')
			continue;

		string fullPath = freenect_video_folder + "/" + line;
		lastPath = line;

		if (line[0] == 'r')
		{
			if (updatedRGB)
				break;

            RGB = cv::imread(fullPath);
			//RGB = cv::Mat(cvLoadImage(fullPath.c_str(), CV_LOAD_IMAGE_UNCHANGED));
			updatedRGB = true;
		}

		else if (line[0] == 'd')
		{
			if (updatedDepth)
				break;

			ifstream in(fullPath, ifstream::binary);
			string format_settings;
			getline(in, format_settings);

			uint32_t width, height, max_size;
			sscanf(format_settings.c_str(), "P5 %d %d %d", &width, &height, &max_size);

			DepthRaw = cv::Mat(cv::Size(width, height), CV_16UC1);

			in.read((char*) DepthRaw.data, width * height * sizeof(uint16_t));
			in.close();

			if (!IsLittleEndian())
			{
				uint8_t* rData = (uint8_t*)DepthRaw.data;
				for (unsigned int i = 0; i < width * height; i++, rData += 2)
				{
					std::swap(*rData, *(rData + 1));
				}
			}

			Depth = cv::Mat(cv::Size(width, height), CV_8UC1);
			uint16_t *rData = (uint16_t*)DepthRaw.data;
			uint8_t *dData = (uint8_t*)Depth.data;

			for (unsigned int i = 0; i < width * height; i++, rData++, dData++)
			{
				*dData = *rData >> 3;
			}

			updatedDepth = true;
		}
		else
		{
			finished = true;
			break;
		}

		previous_line = line;
	} while (getline(reader, line));

	double newTime = GetTimestampFromFilename(lastPath);

	if (previous_timestamp != 0)
	{
		double diff = newTime - previous_timestamp;
		diff *= 1000;
		
		chrono::milliseconds now = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch());
		chrono::milliseconds time_to_wait = chrono::milliseconds((int)diff) - (now - previous_time_ran);
		
		if (time_to_wait.count() > 0)
			this_thread::sleep_for(time_to_wait);
	}

	previous_time_ran = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch());
	previous_timestamp = newTime;

	if (reader.eof())
		finished = true;

	uint8_t returnValue = 0;

	if (updatedDepth)
		returnValue |= State::UPDATED_DEPTH;

	if (updatedRGB)
		returnValue |= State::UPDATED_RGB;

	return returnValue;
}

double FreenectPlaybackWrapper::GetTimestampFromFilename(std::string Filename)
{
	auto first = Filename.begin() + 2;
	auto last = Filename.begin() + 19;

	std::string final;
	final.resize(20);
	std::copy(first, last, final.begin());

	stringstream ss;
	ss << final;

	double returnValue;
	ss >> returnValue;

	return returnValue;
}