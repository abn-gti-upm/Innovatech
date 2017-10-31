/************************************************************************************************** 
* Copyright (C) 2015-2016 Lorena García de Lucas (lgl@gti.ssr.upm.es)
* Grupo de Tratamiento Digital de Imágenes, Universidad Politécnica de Madrid (www.gti.ssr.upm.es)
*
* This file is part of the project "Spatial Grid of Foveatic Classifiers Detector"
*
* Extended by Andrés Bell Navas (abn@gti.ssr.upm.es) 2016-2017
*
* This software is to be used only for research and educational purposes.
* Any reproduction or use for commercial purposes is prohibited
* without the prior express written permission from the author. 
*
* Check included file license.txt for more details.
**************************************************************************************************/
#if !defined DETECTOR
#define DETECTOR
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "frameProcessor.h"
#include "haar_source/haarfeatures.h"
#include "haar_source/traincascade_features.h"

#include <pthread.h>
#include "svm_multithread.h"
#include "pegasos_optimize.h"
#include "simple_sparse_vec_hash.h"
#include "WeightVector.h"

using namespace std;
using namespace cv;

class Detector : public FrameProcessor {

public:
	/** VARIABLES --------------------------------------------------------------------------------------------*/
	//enums
	enum Mode { GENERATION = 0, DETECTION = 1, TEST = 2 };				//define operation mode
	enum GridType { RECTANGULAR = 0, QUINCUNX = 1, CIRCULAR = 2 };		//define pattern of used grid
	enum DescriptorType { HOG = 0, HAAR = 1 };							//define used features

	//general flags
	bool showGrid;						//show grid of points (each point is associated with one classifier)	
	bool showActivations;				//show individual activations of classifiers (magenta points)
	bool showActivationsAndConfidence;	//show individual activations of classifiers and associated confidence
	bool showNotTrained;				//show not trained classifiers (yellow point)
	bool showBoundary;					//show boundary of the ROI (region of interest)

	bool verbose;						//additional console messages

	//flags when using HAAR features
	bool useMasks;						//defines if segmentation masks are used for detection
	bool showCentroids;					//defines if traditional centroid computation should be done and shown

	//flag when using leave-one-out mode (necessary to update for multiclass)
	bool useCrossValidation;			//use leave-one-out mode

	//flag to use threads in classification
	bool useThreads;

	//flag to use Pegasos for classification
	bool usePegasos;

	//flag for decision strategy (either one-vs-all or get all)
	bool oneVsAll;

	/** FUNCTIONS --------------------------------------------------------------------------------------------*/
	//Constructor
	Detector();

	//Public setters
	void setFrameSize(Size size);						//set frame dimensions
	void setMode(Mode chosenMode);						//set operation mode
	void setGridType(GridType type);					//set grid type
	void setDescriptorType(DescriptorType type);		//set descriptor type
	void setDimensions(int numRows, int numCols);		//set number of rows and columns of grid (rectangular and quincunx grids)
	void setGroundTruthFolder(string filename);		    //set ground truth folder (generation and testing only)
	void setActivationsFolder(string filename);			//set activations matrix folder (testing only)
	void setTrainedModelsFolder(string folder);			//set previously trained models folder (containing .yml files)
	void setPrefix(string prefix);						//set prefix of trained models obtained with massive testing of parameters
	//void setMinGroupThreshold(float thresh);			//set minimum confidence for group neighbors
	void setMinGroupThresholds(vector <double> threshs); //set minimum confidence for group neighbors for each class
	//void setHyperplaneThreshold(float thresh);          //set hyperplane threshold translation
	void setHyperplaneThresholds(vector <double> threshs); //set hyperplane threshold translations for each class
	
	void writeOutputVideo(string filename);				//set true to output video, and set filename (must end in .avi)

	void setBoundaryHeight(int height);					//set height of the boundary
	void setInputFile(string filename);					//set sequence filename

	//Preprocess function (general, before frame by frame processing): loading and configuring the detector
	void preprocess();

	//Process function (general, to be called within each frame)
	void process(cv::Mat &frame, cv::Mat &output);


private:
	/** VARIABLES -------------------------------------------------------------------------------------------*/
	//general variables
	Size frame_size;					//size of frames to process
	int numRowsClassifier;				//number of rows of the classifiers' grid (rectangular or quincunx grid)
	int numColsClassifier;				//number of columns of the classifiers' grid (rectangular or quincunx grid)		
	Mode mode;							//operation mode
	GridType gridType;					//grid type
	DescriptorType descriptorType;		//descriptor to use
	int heightBoundary;					//height of the boundary
	vector<string> class_names;			//Names of the considered classes
	
	vector <Ptr<SVM>> svm_ptr;			//vector of SVM classifiers pointers
	vector <vector<Ptr<SVM>>> classes_svm_ptr; //Vector which contains for each class a vector of SVM classifiers pointers	

	Mat grid;							//grid of classifiers' positions (numCols*numRows x 2)
	Mat positions;						//reshaped grid for alternative processing: numRows x numCols x 2 channels (1st channel x_coordinate, 2nd channel y_coordinate)

	CvMLData csvFile;					//object to read .csv files/data
	int numFrame;						//number of frame currently being processed (frame counter)

	VideoWriter writer;					//object to save output video
	bool saveVideo;						//bool to set recording of output video
	string videoName;					//output video filename (must end in .avi)

	string inputFile;					//input directory

	//mode specific variables (variables needed for detection are also needed for test mode)
	//float hyperplane_threshold;
	vector <double> hyperplane_thresholds;
	int numNeighbors;							//(generation) number of neighbors for associating classifiers
	string groundTruthFolder;					//(generation/test) name of folder containing ground truth info for all frames in sequence
	Mat groundTruth;							//(generation/test) matrix with ground truth info of all frames in sequence
	vector <Mat> classesGroundtruths;			//(generation) collection of matrixes with ground truth info of all frames in sequence
	Mat activations;							//(generation/test) (numSamples x numClassifiers) matrix containing classifiers activations-> activations(#sample, #classifier) = 255 if classifier activated, 0 otherwise
	vector <Mat> classesActivations;
	int totalFrames;							//(generation/test) number of total frames in sequence
	vector <vector<Point>> sectors;				//(generation/test) set of sectors from image (for statistical purposes)
	vector<float> gt_per_sector;				//(generation) vector containing ground truth points distributed over sectors
	vector<vector<float>> classes_gt_per_sector; 
	string trainedModelsFolder;					//(detection) path to folder containing previously trained SVM models
	string modelsPrefix;						//(detection) prefix of all used trained models contained in the folder
//	float minGroupThreshold;					//(detection) minimum confidence of group of classifiers to consider positive prediction
	vector <double> minGroupThresholds;			//(detection) for each class, minimum confidence of group of classifiers to consider positive prediction
	vector <int> ActivePoints;					//(detection) indicates which points of the grid have at least one trained classifier (i.e. in the point there are not only dummy svms).

	//struct svm_multithread::svm_ids;								//(detection) struct which indicates the specific classifier (for threading)

	//vector<svm_multithread> svm_id;
	vector <struct svm_ids> variable_svm_classifiers;	//(detection) vector which contains information about each classifier (for threading)

	vector <pthread_t> variable_threads;		//(detection) vector of thread handlers associated to active classifiers
	vector <pthread_t> detection_threads;		//(detection) vector of thread handlers associated to each class (to compute detections)
	//vector <struct svm_ids> svm_ids_data;		

	string activationsFolder;					//(test) name of folder containing theorical activations of classifiers for each frame in sequence
	Mat realActivated;							//(test) matrix containing real activated classifiers through detection
	vector<Mat> class_realActivated;			//(test) vector containing for each class a matrix about real activated classifiers through detection
	Mat foundPositions;							//(test) matrix containing detections for each of the frames in sequence (totalNumFrames x maxNumberFoundPoints x 2 channels)
	vector <Mat> classes_foundPositions;		//(test) vector of matrixes containing detections for each class and for each of the frames in sequence (num_classes) x (totalNumFrames x maxNumberFoundPointsForClass x 2 channels)
	int totalTP, totalFP, totalTN, totalFN;		//(test) metrics
	int trainedClassifiers;						//(test) number of trained classifiers
	int framesNotSkipped;						//(test/leave-one-out mode) number of test frames which have at least one classifier previously trained
	vector <int> framesWithModels;				//(test/leave-one-out mode) vector containing which frames of the sequence have trained classifiers

	//Variable which indicates the length of the feature vector (especially for Haar features when using Pegasos)
	int vector_length;

	//BackgroundSubtraction/HAAR specific variables
	BackgroundSubtractorMOG2 bg;	//background subtractor object
	CvHaarEvaluator eval;			//feature extractor - Haar
	int slidingWin_width, slidingWin_height, sliding_step_h, sliding_step_w;	//sliding window params
	Size finalROI_size;										//final size of the ROI were HAAR features are computed					

	//HOG specific variables
	HOGDescriptor hog;				//feature extractor - HOG
	int step_w, step_h;				//stride for sliding window
	int max_scale;					//max scale - resizes original imagen until this maximun is reached
	Mat means;						//matrix contaning means used in features' normalization
	Mat sigmas;						//matrix contaning standard deviations used in features' normalization

	simple_sparse_vector means_pegasos;		//matrix contaning means used in features' normalization (Pegasos Format)
	simple_sparse_vector sigmas_pegasos;	//matrix contaning standard deviations used in features' normalization (Pegasos Format)

	//Time profiling
	double totalDuration;
	double totalBgsubDuration;
	double totalPredictionDuration;
	double totalFeatExtractionDurationHOG;
	double totalFeatExtractionDurationHAAR;

	/** FUNCTIONS ------------------------------------------------------------------------------------------*/
	//general functions
	void createGrid(double numRows, double numCols, Size frameSize, Mat& grid);
	void createQuincunxGrid(double numRows, double numCols, Size frameSize, Mat& grid);
	void createCircularGrid(Size frameSize, Mat& grid, int center_x_offset, int center_y_offset, double num_circles, double num_min_points);

	void setSVMs(Mat &grid);							//set and initialize array of classifiers

	void getSectorsCirc(vector<vector<Point>> &sectors, Size frameSize, int center_x_offset, int center_y_offset, double num_circles, double num_min_points);
	void getSectorsRect(vector<vector<Point>> &sectors, Size frameSize, double numRows, double numCols);
	void getSectorsQuinc(vector<vector<Point>> &sectors, Mat &grid, Size frameSize, double numRows, double numCols);
	vector<int> findSector(vector<vector<Point>> sectors, Point point);

	void drawGrid(Mat& grid, Mat& frame, Scalar color, bool showSVMnumber);
	void drawGridPoint(Mat &grid, int number, Mat& frame, Scalar color,int class_label);
	void drawGridPointAndConfidence(Mat &grid, int number, Mat &frame, Scalar color, float &confidence,int class_label);
	void drawPointAndConfidence(Point point, Mat &frame, Scalar color, float &confidence);	
	
	void loadCsv(string filename, Mat &data);
	void saveMatToCsv(Mat &matrix, string filename);
	void saveDetectionsToTxt(vector <Mat> &foundPositions,string filename);
	void readVector(string filename, simple_sparse_vector &data);

	void saveMainVariables(Point center, int num_circles);		//save main grid variables for later use
	void loadMainVariables(string filename);					//load main grid variables from file
	void loadDescriptorVariables(string filename);				//load main descriptor variables
	void loadClassNames(string filename);						//load class names

	void readDirectory(const string& directoryName, vector<string>& filenames); 

	void getBBox(Point foundPoint, Point2f center, Point2f offset, vector<Point2f> &bbox);
	void addColorbar(Mat &image, double min, double max, bool sideShow, bool separateShow, double x_offset, double y_offset);

	//features' extraction
	void computeHaarFeatures(Mat &frame, vector<float> &descriptors, int step_w, int step_h, int overlapping_step_h, int overlapping_step_w);
	void computeHOGFeatures(Mat &frame, vector<float> &descriptors, int step_w, int step_h, int max_scale);
	
	//Alternative features'extraction functions
	void computeHOGFeaturesPegasos(Mat &frame, simple_sparse_vector& descriptors, int step_w, int step_h, int max_scale);
	void computeHaarFeaturesPegasos(Mat &frame, simple_sparse_vector &descriptors, int step_w, int step_h, int overlapping_step_h, int overlapping_step_w);

	//generation specific functions
	void getNearestClassifiers(Mat &grid, Point &point, vector<int> &numNearestClassifier, int &numNeighbors);
	void associateGT(Mat &output, int numNeighbors, int class_index);
	void drawGT(Mat &output, vector<Point> &points, int class_index);
	void drawLine(Mat &img, Point pt1, int classifier);
	void getSamplesDistribution(Mat &activations, Mat &sampleDistribution);

	//detection specific functions
	void detect(Mat &frame, vector<float> descriptors,simple_sparse_vector descriptors_pegasos);
	void detect_circular(Mat &frame, vector<float> descriptors,simple_sparse_vector descriptors_pegasos);
	void getConnectedComp(Mat &image, vector<vector<Point> > &contours);									//only for HAAR
	void getSegmentationMasks(Size frame_size, vector<vector<Point> > &contours, vector<Mat> &seg_masks);	//only for HAAR
	void getCentroids(Size frame_size, vector<vector<Point> > &contours);									//only for HAAR

	//test specific functions (errors and metrics)
	void getMetrics(Mat groundTruthActivations, Mat realActivations, int &TP, int& FP, int& TN, int& FN);
	void getError2(string gtfilename, Mat positionsFound);
	void getError3(string gtfilename, Mat positionsFound);
	//void getError4(string gtfilename, Mat positionsFound, int &truePositives, int &extraDetectedPoints, int &notDetectedPoints);
	void getError4(string gtfilename, vector <Mat> &positionsFound, int &truePositives, int &extraDetectedPoints, int &notDetectedPoints);
	//Preprocess function (per operation mode, before frame by frame processing): loading and configuring the detector
	void preprocessGeneration();
	void preprocessDetection();
	void preprocessTest();

	//Process function (per operation mode, to be called within each frame)
	void processGeneration(cv::Mat &frame, cv::Mat &output);
	void processDetection(cv::Mat &frame, cv::Mat &output);
	void processTest(cv::Mat &frame, cv::Mat &output);

};

#endif
