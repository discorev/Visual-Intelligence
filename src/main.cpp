#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "freenect-playback-wrapper.h"

// Key settings
#define THRESHOLD_VALUE 100 //14        // Filtering threshold value for depth
#define STD_SIZE cv::Size(64,64)      // Standardized size of image for SVM

// SVM learning flags
#define LEARNING false                  // Should this run be used to generate the SVM model?
#define SVM_LINEAR_KERNAL true          // Use Linear kernel or radial basis function
// debugging flags
#define DEBUG_NOISE_FILTER false        // Output the co-ordinates of the bottom right corner of noise that's filtered

// a structure to hold the kinect data frames
struct frame {
    cv::Mat RGB;
    cv::Mat Depth;
};

// this holds a "processed frame", the original frame, plus masked RGB and polygon used for the mask.
struct processed_frame {
    frame original_frame;
    cv::Mat masked_RGB;
    std::vector<cv::Point> countour_poly;
};

int main(int argc, char * argv[])
{
    // by default just look for a directory ./data containing the kinect data.
    std::string file = "./data";
    // if the user has passed an argument, process it.
    if(argc == 2)
    {
        file = std::string(argv[1]); // read in the argument as the location for the kinect data
        if(file.compare("--help") == 0) // in the case the argument was a request for help, give some!
        {
            std::cout << "usage: visualIntelligence" << std::endl;
            std::cout << "       visualIntelligence kinect_folder" << std::endl;
            return 0; // finish help
        }
    }
    
#if !(LEARNING)
    // load trained SVM from SVM.xml
    cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>("SVM.xml");
    // load the class labels from from labels.txt
    std::vector<std::string>labels = std::vector<std::string>();
    std::ifstream labelFile("labels.txt");
    // whilst the file is open, read each line (label) and save it to the vector
    if(labelFile.is_open())
    {
        std::string line;
        while(std::getline(labelFile, line))
        {
            labels.push_back(line);
        }
    } else {
        // there was an error reading the labels. As the vector is used to output,
        // this is currently an error that must result in the termination of the program
        std::cerr << "Failed to open labels file." << std::endl;
        return -1;
    }
    int *confusion; // create the int with the number of labels
#else
    // if we're in learning mode, declare the vectors needed to hold the objects to be trained on!
    std::vector<std::vector<processed_frame>> all_objects; // this holds all of the objects that will be used as the training set
    std::vector<processed_frame> current_object;           // this holds the current object on the screen whilst playing back the kinect data.
#endif
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
#if !(LEARNING)
    // if not learning, additional flag for if the object has been recognised by the SVM
    // and current_object to hold the class (this allows detecting changes between frames)
    bool unrecognised_Object = true;
    int current_object = -1;
#endif

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
        {
            // create a translation matrix to translate the depth image to bring it more inline with the RGB
            cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, -38, 0, 1, 25);
            // apply the transform to the depth image
            cv::warpAffine(wrap.Depth,current.Depth,trans_mat,current.Depth.size(),cv::INTER_LINEAR, cv::BORDER_CONSTANT, 255);
            current.Depth = current.Depth+158; // push the background to inf
                                            // this makes the normalization spread the arm and object further
            cv::imshow("Depth", current.Depth);
        }
        
        cv::Mat depth_raw = current.Depth.clone(); // initalize with a clone the matrix rather than pointer to it
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
        
        // clean up the thresholded RGB image and then
        cv::bitwise_and(thresholded, src_gray, src_gray);
        cv::bitwise_or(thresholded, src_gray, thresholded);
        
        // use dilation on the thresholded bitmask. This "blurs" the edges and will join areas that
        // are only seperated by a small margin. The blur factor is determined by blur_factor
        int blur_factor = 4;
        // generate an elliptic element to be used in the dilation
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size( 2*blur_factor+1, 2*blur_factor+1), cv::Point(blur_factor,blur_factor));
        // apply the dilation using the element generated.
        cv::dilate(thresholded, thresholded, element);
        
        // display the thresholded image on the screen.
        cv::imshow("Raw Depth", thresholded);
        
        // ------ DETECT CONTOURS AND OBJECT IN IMAGE ------
        
        // controus will hold the contours found in the image
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        
        cv::findContours( thresholded, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
        
        // find the largest contour by area - This should be the object!
        double largest_area=0;
        int largest_contour_index=-1;
        
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
        cv::Rect boundRect; // this is the rectangle around the main contour
        cv::Point2f center; // center of a circle bounding the main contour
        float radius;       // radius of a circle bounding the main contour
        
        // if there are contours, and a largest has been found out of them then we want to find the
        // binding box and circle for this "main" contour
        if(contours.size() > 0 && largest_contour_index > -1)
        {
            cv::approxPolyDP( cv::Mat(contours[largest_contour_index]), contours_poly[0], 3, true );
            boundRect = cv::boundingRect( cv::Mat(contours_poly[0]) );
            cv::minEnclosingCircle( (cv::Mat)contours_poly[0], center, radius );
            
            // attempt to remove noise by removing areas that are too small
            if(boundRect.size().width < 100 && (boundRect.size().height < 40 || boundRect.br().x > 590 || (boundRect.size().height < 100 && boundRect.br().y > 440)))
            {
#if DEBUG_NOISE_FILTER
                std::cout << boundRect.br() << std::endl;
#endif
                boundRect = cv::Rect(0,0,0,0);
                contours_poly.clear();
                center = cv::Point2f(0,0);
                radius = 0;
            }
        }
        
        /// Draw polygonal contour + bonding rects + circles
        cv::Mat drawing = cv::Mat::zeros( src_gray.size(), CV_8UC3 );
        cv::Mat masked = cv::Mat::zeros( src_gray.size(), CV_8UC3 );
        cv::Mat temp;
        if(contours.size() > 0)
        {
            cv::Scalar color = cv::Scalar( 255, 0, 0 );
            // if the bound rect bottom right is 590 or higher, the object is  still entering
            // the image
            if(boundRect.area() > 0 && boundRect.br().x <= 590)
            {
                // convert the contour into a filled bit mask and then apply it to the RGB image to only
                // see the object that is found in the depth
                cv::drawContours( masked, contours_poly, 0, cv::Scalar(255,255,255), cv::FILLED );
                cv::bitwise_and(current.RGB, masked, masked);
                
                if(!object_on_screen)
                {
                    std::cout << "Object has appeared" << std::endl;
#if !(LEARNING)
                    confusion = new int[labels.size()] {};
#endif
                    object_on_screen = true;
                }
#if !(LEARNING)
                // see if we can recognise the object
                cv::Mat stdSize, svmPredict;
                // resize each image/ROI to a standard size (defined as STD_SIZE) and save it in stdSize
                cv::resize(masked(boundRect).clone(), stdSize, STD_SIZE, cv::INTER_CUBIC);// cv::INTER_NEAREST);
                svmPredict.push_back(stdSize.reshape(1,1)); // make the image into a line vector
                cv::cvtColor(depth_raw, depth_raw, cv::COLOR_GRAY2BGR);
                cv::resize(depth_raw(boundRect).clone(), stdSize, STD_SIZE, cv::INTER_CUBIC);
                svmPredict.push_back(stdSize.reshape(1,1));
                svmPredict = svmPredict.reshape(1,1);
                svmPredict.convertTo(svmPredict, CV_32FC1); // ensure that vector is of the right type
                int object_class = svm->predict(svmPredict);// get the class prediction from the SVM
                confusion[object_class] += 1; // increment this class in the confusion matrix
                if(unrecognised_Object)
                {
                    // if not recognised yet, print the label found in the SVM and set that it has been recognised
                    std::cout << labels[object_class] << std::endl;
                    unrecognised_Object = false;
                } else if(current_object != object_class)
                {
                    // if it has been recognised but the class for this frame is different to the current, give a message!
                    std::cout << "Object changed to: " << labels[object_class] << " - that was unexpected!" << std::endl;
                }
                current_object = object_class; // update the current object class
#else
                // if learning, make sure that the new frame is pushed back onto the current object vector
                frame store = {current.RGB(boundRect), depth_raw(boundRect).clone()}; // create the frame with just the ROI
                processed_frame proc_f = {store, masked(boundRect).clone(), contours_poly[0]};
                current_object.push_back(proc_f);
#endif
            } else if(object_on_screen)
            {
                std::cout << "Object has gone!" << std::endl;
                object_on_screen = false;
#if !(LEARNING)
                std::cout << "Confusion matrix:" << std::endl;
                // show confusion matrix
                for(int i=0; i<labels.size(); i++)
                {
                    std::cout << "\t" << labels[i] << " " << confusion[i] << std::endl;
                }
                delete[] confusion; // free the memory now.
                
                // reset the recognised flag and object class
                unrecognised_Object = true;
                current_object = -1;
#else
                // push back the current object onto the stack of objects to be learnt
                all_objects.push_back(current_object);
                current_object.clear();
#endif
            }
            // draw the main contour for the object, it's bounding rect and circle to be displayed on screen
            cv::drawContours( drawing, contours_poly, 0, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
            cv::rectangle( drawing, boundRect.tl(), boundRect.br(), color, 2, 8, 0 );
            cv::circle( drawing, center, (int)radius, color, 2, 8, 0 );
        }
        
        // ------ END DETECTING CONTOURS ------
        
		// Show the images in the windows
        cv::imshow("RGB", current.RGB);
        cv::imshow("Object", drawing);
        cv::imshow("Masked", masked);
        
		// Check for keyboard input
		key = cv::waitKey(10);
	}
    // after having gone through all the kinect data, close all the windows that've been opened
    cv::destroyAllWindows();
#if LEARNING
    std::cout << all_objects.size() << " Objects have been found in the video" << std::endl;

    cv::Mat classes;
    cv::Mat trainingData;
    
    cv::Mat trainingImages;
    std::vector<int> trainingLabels;
    // run through the saved frames and convert them to training data.
    int label = 0;
    for(std::vector<processed_frame> object_frames : all_objects)
    {
        for(int i=0; i<object_frames.size();i++)
        {
            frame cur = object_frames.at(i).original_frame;
            cv::Mat stdSize, svmTrain, depthTmp;
            // resize each image/ROI to a standard size of 64px x 64px and save it in stdSize
            // UPDATE: the standard size is now a define as STD_SIZE so that it can be altered.
            cv::resize(object_frames.at(i).masked_RGB, stdSize, STD_SIZE, cv::INTER_CUBIC); // cv::INTER_NEAREST);
            svmTrain.push_back(stdSize.reshape(1,1));
            // UPDATE 2: push the depth onto the train image as well
            cv::cvtColor(object_frames.at(i).original_frame.Depth, depthTmp, cv::COLOR_GRAY2RGB);
            cv::resize(depthTmp, stdSize, STD_SIZE, cv::INTER_CUBIC);
            svmTrain.push_back(stdSize.reshape(1,1));
            trainingImages.push_back(svmTrain.reshape(1,1));
            trainingLabels.push_back(label); // set the label for the current vector.
        }
        label++;
    }
    cv::Mat(trainingImages).copyTo(trainingData);
    trainingData.convertTo(trainingData, CV_32FC1);
    cv::Mat(trainingLabels).copyTo(classes);
    
    cv::ml::SVM::Params params;
#if SVM_LINEAR_KERNAL
    params.kernelType = cv::ml::SVM::LINEAR;
#else
    params.kernelType = cv::ml::SVM::RBF;
#endif
    std::cout << "Training the SVM" << std::endl;
    cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::train<cv::ml::SVM>(trainingData, cv::ml::ROW_SAMPLE, classes, params);
    std::cout << "SVM trained.\n Saving to SVM.xml" << std::endl;
    svm->save("SVM.xml");
    std::cout << "SVM has been saved!" << std::endl;
#endif

	return 0;
}
