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
	cv::namedWindow("Object", cv::WINDOW_AUTOSIZE);

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
        
        // create a grayscale from the RGB
        cv::Mat src_gray;
        cv::cvtColor( currentRGB, src_gray, cv::COLOR_RGB2GRAY );
        cv::threshold( src_gray, src_gray, 190, 255,cv::THRESH_BINARY);
        //cv::blur( src_gray, src_gray, cv::Size(10,10) );
        
		// Determine if Depth is updated, and grabs the image
		// if it has been updated
		if (status & State::UPDATED_DEPTH)
            cv::normalize(wrap.Depth, currentDepth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            //currentDepth = wrap.Depth;
        
        cv::Mat thresholded;
        cv::threshold( currentDepth, thresholded, THRESHOLD_VALUE, 255,cv::THRESH_BINARY_INV);
        
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
        
        // ------   MASKING THE RGB IMAGE   -------
        
        // create a translation matrix to translate the depth image to bring it more inline with the RGB
        cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, -38, 0, 1, 25);
        // apply the transform to the depth image
        cv::warpAffine(thresholded,thresholded,trans_mat,thresholded.size());
        
        // ------ END MASKING THE RGB IMAGE -------
        
        // clean up the thresholded RGB image and then
        cv::bitwise_and(thresholded, src_gray, src_gray);
        cv::bitwise_or(thresholded, src_gray, thresholded);
        
        // ------ DETECT CONTOURS AND OBJECT IN IMAGE ------
        
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        
        cv::findContours( thresholded, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
        
        double largest_area=0;
        int largest_contour_index=0;
        
        for( int i = 0; i < contours.size(); i++ )
        {
            double a = cv::contourArea( contours[i],false);
            if(a > largest_area)
            {
                largest_area = a; largest_contour_index = i;
            }
        }
        
        /// Approximate contours to polygons + get bounding rects and circles
        std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
        cv::Rect boundRect;
        cv::Point2f center;
        float radius;
        
        for( int i = 0; i < contours.size(); i++ )
        {
            if(i == largest_contour_index)
            {
                cv::approxPolyDP( cv::Mat(contours[largest_contour_index]), contours_poly[largest_contour_index], 3, true );
                boundRect = cv::boundingRect( cv::Mat(contours_poly[largest_contour_index]) );
                cv::minEnclosingCircle( (cv::Mat)contours_poly[largest_contour_index], center, radius );
            }
        }
        
        /// Draw polygonal contour + bonding rects + circles
        cv::Mat drawing = cv::Mat::zeros( src_gray.size(), CV_8UC3 );
        cv::Mat masked = cv::Mat::zeros( src_gray.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            if(i == largest_contour_index)
            {
                cv::Scalar color = cv::Scalar( 255, 0, 0 );
                // if the bound rect bottom right is 590 or higher, the object is  still entering
                // the image
                if(boundRect.br().x <= 590)
                    cv::drawContours( masked, contours_poly, i, cv::Scalar(255,255,255), cv::FILLED );
                cv::drawContours( drawing, contours_poly, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
                cv::rectangle( drawing, boundRect.tl(), boundRect.br(), color, 2, 8, 0 );
                cv::circle( drawing, center, (int)radius, color, 2, 8, 0 );
            }
        }
        
        // ------ END DETECTING CONTOURS ------
        
        
        // use the contour region as a mask
        cv::bitwise_and(currentRGB, masked, masked);
        
		// Show the images in the windows
        cv::imshow("RGB", currentRGB);
        cv::imshow("Object", drawing);
        cv::imshow("Masked", masked);
        
		// Check for keyboard input
		key = cv::waitKey(10);
	}

	return 0;
}
