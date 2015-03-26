# Visual Intelligence
This is the code for the coursework I wrote for the Visual Intelligence module I took in my final year. It requires a folder containing data recorded from the Xbox Kinect sensor using a library such as freenect in order to run. (This data has been ommited from the repository for copyright reasons).

Currently there is a flag in the opencvexample.cpp file that sets if the compiled program should be used to train the SVM classifier and save the trained classifier to a file (SVM.xml) or should load a classifier from SVM.xml and use it to classify objects. Later this will be converted to a further (optional) command line arguments to prevent having to recompile.

## Compiling

To compile cd to the directory the code is in then run `cmake .` followed by `make`, e.g.

```
cd ./Visual-Intelligence
cmake .
make
```
## Usage
To run the program you must either have the Kinect recording in the same directory as the executable in a folder called `data` or, must provide the location of the Kinect recording as the first argument.

E.g.

If the kinect recording is located in `./data/` simply run:
<br />`./visualIntelligence`

Otherwise
<br />`./visualIntelligence /Path/to/Kinect/Data/`

## References
The OpenCV library (http://opencv.org)
