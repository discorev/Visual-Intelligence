#include <opencv2/opencv.hpp>
#include "freenect-playback-wrapper.h"

#define THRESHOLD_VALUE 100 //14

// create a nice little structure for each frame
struct frame {
    cv::Mat RGB;
    cv::Mat Depth;
};

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
    std::vector<std::vector<frame>> all_objects;
    
    std::vector<frame> current_object;
    
    FreenectPlaybackWrapper wrap(file);
    
    frame current; // store the grame in a struct, this makes it easier to save them :D
    // when RGB or Depth are updated, consider it a new frame

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
    
    // is there an object on the screen or not
    bool object_on_screen = false;

	while (key != 27 && status != 0)
	{
		// Loads in the next frame of Kinect data into the
		// wrapper. Also pauses between the frames depending
		// on the time between frames.
		status = wrap.GetNextFrame();

		// Determine if RGB is updated, and grabs the image
		// if it has been updated
		if (status & State::UPDATED_RGB)
            current.RGB = wrap.RGB;
        
        // create a grayscale from the RGB
        cv::Mat src_gray;
        cv::cvtColor( current.RGB, src_gray, cv::COLOR_RGB2GRAY );
        cv::threshold( src_gray, src_gray, 190, 255,cv::THRESH_BINARY);
        
		// Determine if Depth is updated, and grabs the image
		// if it has been updated
		if (status & State::UPDATED_DEPTH)
            current.Depth = wrap.Depth+158; // push the background to inf
                                           // this makes the normalization spread the arm and object further
        cv::normalize(current.Depth, current.Depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        
        // apply a threshold to the image
        cv::Mat thresholded;
        cv::threshold( current.Depth, thresholded, THRESHOLD_VALUE, 255,cv::THRESH_BINARY_INV);
        
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
        
        // save tresholded for use later & display it on screen
        cv::Mat depth_raw = thresholded.clone(); // initalize with a clone the matrix rather than pointer to it
        cv::imshow("Raw Depth", thresholded);
        
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
        
        std::vector<cv::Point> largestContour(contours.at(largest_contour_index));
        
        /// Approximate contours to polygons + get bounding rects and circles
        std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
        cv::Rect boundRect;
        cv::Point2f center;
        float radius;
        
        
        if(contours.size() > 0)
        {
            cv::approxPolyDP( cv::Mat(contours[largest_contour_index]), contours_poly[largest_contour_index], 3, true );
            boundRect = cv::boundingRect( cv::Mat(contours_poly[largest_contour_index]) );
            cv::minEnclosingCircle( (cv::Mat)contours_poly[largest_contour_index], center, radius );
            
            // attempt to remove noise by removing areas that are too small
            if(boundRect.size().width < 100 && (boundRect.size().height < 40 || boundRect.br().x > 590))
            {
                boundRect = cv::Rect(0,0,0,0);
                contours_poly.clear();
                center = cv::Point2f(0,0);
                radius = 0;
            }
        }
        
        /// Draw polygonal contour + bonding rects + circles
        cv::Mat drawing = cv::Mat::zeros( src_gray.size(), CV_8UC3 );
        cv::Mat masked = cv::Mat::zeros( src_gray.size(), CV_8UC3 );
        if(contours.size() > 0)
        {
            cv::Scalar color = cv::Scalar( 255, 0, 0 );
            // if the bound rect bottom right is 590 or higher, the object is  still entering
            // the image
            if(boundRect.area() > 0 && boundRect.br().x <= 590)
            {
                cv::drawContours( masked, contours_poly, largest_contour_index, cv::Scalar(255,255,255), cv::FILLED );
                if(!object_on_screen)
                {
                    std::cout << "Object has appeared" << std::endl;
                    object_on_screen = true;
                }
                frame store = {current.RGB(boundRect), depth_raw(boundRect)}; // create the frame with just the ROI
                current_object.push_back(store);
            } else if(object_on_screen)
            {
                std::cout << "Object has gone!" << std::endl;
                object_on_screen = false;
                all_objects.push_back(current_object);
                current_object.clear();
            }
            cv::drawContours( drawing, contours_poly, largest_contour_index, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
            cv::rectangle( drawing, boundRect.tl(), boundRect.br(), color, 2, 8, 0 );
            cv::circle( drawing, center, (int)radius, color, 2, 8, 0 );
        }
        
        // ------ END DETECTING CONTOURS ------
        
        
        // use the contour region as a mask
        cv::bitwise_and(current.RGB, masked, masked);
        
		// Show the images in the windows
        cv::imshow("RGB", current.RGB);
        cv::imshow("Object", drawing);
        cv::imshow("Masked", masked);
        
		// Check for keyboard input
		key = cv::waitKey(10);
	}
    cv::destroyAllWindows();
    std::cout << all_objects.size() << " Objects have been found in the video" << std::endl;
    
    // run through the bloody lot
    for(std::vector<frame> object_frames : all_objects)
    {
        for(int i=0; i<object_frames.size();i++)
        {
            frame cur = object_frames.at(i);
            cv::imshow("RGB", cur.RGB);
            cv::imshow("Depth", cur.Depth);
            cv::waitKey(0);
        }
    }

	return 0;
}
