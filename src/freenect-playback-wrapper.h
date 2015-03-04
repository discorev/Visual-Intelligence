#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <cstdint>
#include <chrono>

enum State
{
	UPDATED_RGB = 0x01,
	UPDATED_DEPTH = 0x10
};

class FreenectPlaybackWrapper
{
public:

	// Constructs the class
	//
	// std::string FreenectVideoFolder Indicates the folder that the 
	//                                 kinect video is placed
	FreenectPlaybackWrapper(std::string FreenectVideoFolder);

	// Call when ready to get the next frame. Will pause the program
	// based on the timestamps in the original video.
	//
	// Returns a uint8_t that is generated from OR-ing the enum State
	//         values to specify whether RGB and/or Depth has been
	//         updated
	uint8_t GetNextFrame();

	// Stores the latest RGB image
	cv::Mat RGB;

	// Stores the latest Depth values scaled to range ( 0 - 255 ) CV_8UC1
	cv::Mat Depth;

	// Stores the latest Depth raw values with a range ( 0 - 2047 ) CV_16UC1
	cv::Mat DepthRaw;

protected:
	std::ifstream reader;
	uint8_t previous_state = 0;
	bool finished = false;

	std::string freenect_video_folder = "";
	std::string previous_line = "";

	double previous_timestamp = 0;
	std::chrono::milliseconds previous_time_ran;

	// Calculate the timestamp from the filename of an image
	double GetTimestampFromFilename(std::string Filename);
	

};