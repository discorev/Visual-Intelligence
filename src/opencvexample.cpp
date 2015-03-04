#include <opencv2/opencv.hpp>
#include "freenect-playback-wrapper.h"

#define THRESHOLD_VALUE 14

int main(int argc, char * argv[])
{
    std::string file = "./data";
    if(argc == 2)
    {
        file = std::string(argv[1]);
        if(file.compare("--help") == 0)
        {
            std::cout << "usage: visualIntelligence" << std::endl;
            std::cout << "       visualIntelligence kinect_folder" << std::endl;
            return 0; // finish help
        }
    }
    
    FreenectPlaybackWrapper wrap(file);

	cv::Mat currentRGB;
	cv::Mat currentDepth;

	// Create the RGB and Depth Windows
	cv::namedWindow("RGB", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);

	// The key value represents a key pressed on the keyboard,
	// where 27 is the ESCAPE key
	char key = '0';

	// The status represents the current status of the Playback
	// wrapper. 
	//
	// A value of 0 represents that it has finished
	// playback.
	//
	// The status can by bitwise AND to determine if the RGB or
	// Depth image has been updated using the State enum.
	uint8_t status = 255;

	while (key != 27 && status != 0)
	{
		// Loads in the next frame of Kinect data into the
		// wrapper. Also pauses between the frames depending
		// on the time between frames.
		status = wrap.GetNextFrame();

		// Determine if RGB is updated, and grabs the image
		// if it has been updated
		if (status & State::UPDATED_RGB)
			currentRGB = wrap.RGB;

		// Determine if Depth is updated, and grabs the image
		// if it has been updated
		if (status & State::UPDATED_DEPTH)
            cv::normalize(wrap.Depth, currentDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            //currentDepth = wrap.Depth;
        
        cv::Mat thresholded;
        cv::threshold( currentDepth, thresholded, THRESHOLD_VALUE, 255,0);
        
		// Show the images in the windows
		cv::imshow("RGB", currentRGB);
        cv::imshow("Depth", thresholded);//currentDepth);

		// Check for keyboard input
		key = cv::waitKey(10);
	}

	return 0;
}
