/**************************************************************************************************
* This file is part of the project "Spatial Grid of Foveatic Classifiers Detector"
*
* Created by Andrés Bell Navas (abn@gti.ssr.upm.es) 2016-2017
**************************************************************************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <pthread.h>
#include "WeightVector.h"

using namespace cv;

	/**Struct which indicates the specific classifier, for threading (only input parameters)*/
	struct svm_ids{

		//Input parameters
		double ht;
		int num_thread;
		int class_label;
		int numClassifier;
		int num_frame;
		Mat descriptor_vector;
		simple_sparse_vector *descriptor_vector_pegasos;
		WeightVector W;
		Ptr<SVM> svm_pointer;
		
	};


	/**Struct with the required input parameters to compute final detections*/
	struct detection_parameters{

		int numRowsGrid;
		int numColsGrid;
		vector <double> mgt;

		vector<string> names;

		CvMat current_frame;
		CvMat classifiers_positions;

	};

	/**Multithreading function to predict and compute confidence*/ 
	void *prediction_thread(void *threadarg);

	/**Multithreading function to compute final detections*/
	void *detection_thread(void *threadarg);