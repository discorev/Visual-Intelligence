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
    
    bool oneSet = false;
    bool bothSet = false;

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
        cv::threshold( currentDepth, thresholded, THRESHOLD_VALUE, 255,1);
        
        // create a region of interest that removes the last 9 pixels of the depth data
        // this is inf due to the Kinect using a correlation window 9 pixels wide and needs to be removed to get
        // an accurate mean.
        cv::Rect roi( 0, 0, thresholded.size().width-9, thresholded.size().height );
        cv::Mat image_roi = thresholded(roi); // get just the ROI of the depth data
        // if the mean value is 255 there is uniform depth (i.e. no object) so set the output to black
        // note this is more for athstetics than actual image processing, but I hope that will get me
        // at least some marks.
        if(cv::mean(image_roi)[0] == 255)
            thresholded = cv::Mat::zeros(thresholded.rows, thresholded.cols, CV_8UC1); // create a zero matrix
        
        // create a translation matrix to translate the depth image to bring it more inline with the RGB
        cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, -38, 0, 1, 25);
        // apply the transform to the depth image
        cv::warpAffine(thresholded,thresholded,trans_mat,thresholded.size());
        
        // convert the depth image back into RGB channels from grayscale
        cvtColor(thresholded,thresholded, cv::COLOR_GRAY2BGR);
        
        // create a new matrix to hold the masked image and apply a bit mask based on the depth
        cv::Mat masked;
        cv::bitwise_and(currentRGB, thresholded, masked);
        
		// Show the images in the windows
        cv::imshow("RGB", currentRGB);
        cv::imshow("Masked", masked);
        cv::imshow("Depth", thresholded);
        
		// Check for keyboard input
		key = cv::waitKey(10);
	}

	return 0;
}
