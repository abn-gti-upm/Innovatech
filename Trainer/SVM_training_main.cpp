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
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <fstream>
#include <pthread.h>
#include "simple_sparse_vec_hash.h"
#include "pegasos_optimize.h"
#include <windows.h>

#include "haar_source/haarfeatures.h"
#include "haar_source/traincascade_features.h"

using namespace cv;
using namespace std;

CvMLData csv;	//structure to load from .csv file

//vector<string> Class_names;

static void help(){
	cout
		<< "**************************************************************************\n"
		<< "* SPATIAL GRID OF FOVEATIC CLASSIFIERS - SVM MODELS TRAINER\n"
		<< "*\n"
		<< "* Trainer for the Spatial Grid of Foveatic Classifiers detector.\n"
		<< "* Requires one or more \"activations\" folders which contain\n" 
		<< "* previously generated activation files with main application to train.\n"
		<< "* When only computing features, activations are not necessary.\n"
		<< "*\n"
		<< "* This program has been extended from the original as part of the\n"
		<< "* following Master Thesis:\n"
		<< "*  - Title: Desarrollo e implementacion de un sistema para el reconocimiento\n"
		<< "*           multi-clase de vehiculos en tiempo real basado en clasificadores\n"
		<< "*           foveaticos\n"
		<< "*  - Author: Andres Bell (abn@gti.ssr.upm.es/andresbellnavas@gmail.com)\n"
		<< "*  - Grupo de Tratamiento de Imagenes, Universidad Politecnica de Madrid\n"
		<< "*    (GTI-UPM www.gti.ssr.upm.es)\n"
		<< "*\n"
		<< "* This program has been created as part of the following Master Thesis:\n"
		<< "*  - Title: Development of an algorithm for people detection\n"
		<< "*           using omnidirectional cameras\n"
		<< "*  - Author: Lorena Garcia (lgl@gti.ssr.upm.es/lorena.gdelucas@gmail.com)\n"
		<< "*  - Grupo de Tratamiento de Imagenes, Universidad Politecnica de Madrid\n"
		<< "*    (GTI-UPM www.gti.ssr.upm.es)\n"
		<< "*\n"
		<< "* For help about program usage, type --help\n"
		<< "**************************************************************************\n"
		<< endl;
}

static void help2(){
	cout
		<< "**************************************************************************\n"
		<< "* List of input parameters: In general, they need a number which indicate the different number\n"
		<< "* of combinations to test, because this trainer can work with massive introduction of parameters\n"
		<< "*\n"
		<< "** General common mandatory params:\n"
		<< "*    -input <number of lists> <number of folders> <list of folder paths> : input folders\n"
		<< "*           containing training images or feature vectors. Indicate total\n"
		<< "*           number of lists, the number of folders of each list, and then list of folders\n"
		<< "*			(separated by spaces)\n"
		<< "*    -descriptor <HOG|HAAR> : descriptor to extract features\n"
		<< "*    -num_classifier <vector<number>> : first classifier to train for each class \n"
		<< "*						(default 0 trains all classifiers) (optional parameter)\n"
		<< "*	 -useThreads: train svms with threads or not \n"
		<< "*    -pegasosTraining: train svms with Pegasos code, which makes possible online \n"
		<< "*           training\n"
		<< "*    -saveVectors: save computed feature vectors in order to be able to reuse them\n"
		<< "*    -onlyComputeFeatures: only compute feature vectors or also train classifiers\n"
		<< "*    -balancedTraining: train each svm with a reasonable size of dataset with the same\n"
		<< "*			number of positive and negative samples when possible (only works in Pegasos)\n"
		<< "*			or with all input data"
		<< "*\n"
		<< "** General common params not strictly mandatory:\n"
		<< "*    -activations <number of lists> <number of folders> <list of folder paths> : folders \n"
		<< "*                 with saved activations matrix. Indicate total number of lists, the number of\n"
		<< "*                 folders in each list, and then list of folders (separated by spaces).\n"
		<< "*				  Not mandatory if only features will be computed.\n"
		<< "*	 -models <number of folders> <list of folder paths>: folders with pre-trained models with\n"
		<< "*                 Pegasos (to update). Indicate total number of folders, and then list of folders\n"
		<< "*	 -division <number>: Value between 0 and 1 which indicates the allowed proportion between positive\n"
		<< "*				  and negative samples (in concrete, this number is the fraction for the minority)\n"
		<< "*	 -max_batch_size <number>: When balancing datasets, indicate the maximum number of samples allowed\n"
		<< "*                 which are split into positive and negative samples\n"
		<< "*\n"
		<< "** Regularization specific params:\n"
		<< "*	 -regularization <number of parameters: regularization parameters of each SVMs; in Pegasos \n"
		<< "*				  these are the initial lambdas, which will be reducing a little during training. \n"
		<< "*                 In OpenCV, these are the C values.\n"
		<< "*	 -iterations: <numbers>: max number of iterations in the training of each classifier \n"
		<< "*				  (only in Pegasos, and there must be the same number of iteration parameters as in\n"
		<< "*				  the regularization)\n"
		<< "*\n"
		<< "** HOG specific params:\n"
		<< "*    -win_sizes <number of configurations> <width height> : window size to compute features\n"
		<< "*    -win_steps <number of configurations> <width height> : sliding step of previous window\n"
		<< "*    -max_scales <number of configurations> <number of scales>: max scales of the Gaussian\n"
		<< "*                multirresolution pyramid\n"
		<< "*\n"
		<< "** HAAR specific params:\n"
		<< "*    -win_sizes <number of configurations> <width height> : original window size to compute\n"
		<< "*                features\n"
		<< "*    -win_steps <number of configurations> <width height> : sliding step of previous window\n"
		<< "*    -haar_final_wins <number of configurations> <width height> : final window to compute features\n"
		<< "*                (resized)\n"			
		<< "*\n\n"
		<< "*******************************************************************************\n"
		<< endl;
}


void readDirectory(const string& directoryName, vector<string>& filenames);
void compute_hog(Mat &img, Mat& descriptor, HOGDescriptor hog, int step_w, int step_h);
void compute_hog_pegasos(Mat &img, simple_sparse_vector& descriptor, uint &dimension, HOGDescriptor hog, int step_w, int step_h);
void compute_haar(Mat& img, Mat& descriptor, CvHaarEvaluator eval, int step_w, int step_h, int overlapping_step, Size finalROI_size);
void compute_haar_pegasos(Mat& img, simple_sparse_vector& descriptor, uint &dimension, CvHaarEvaluator eval, int step_w, int step_h, int overlapping_step, Size finalROI_size);
void train_svm(int type, Mat& gradient_lst, vector< int > labels, const string& output_model_name, double &precision, double &recall, double &f_score, double &best_ht, double C_value);
void *train_svm_thread(void *threadarg);
void *train_svm_thread_pegasos(void *threadarg);
struct svm_data;
//void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels, const string& output_model_name);
void loadCsv(string filename, Mat &data);
void saveMatToCsv(Mat &matrix, string filename);

//Variable to store the individual statistics of each classifier, organized by classes
Mat classes_precisions;
Mat classes_recalls;
Mat classes_fscores;
Mat thresholds;
pthread_mutex_t mutex_training;		//Mutex variable to control access and modification of shared data

/**Read and store filenames from directoryName folder*/
void readDirectory(const string& directoryName, vector<string>& filenames)
{
	glob(directoryName, filenames, false);
}

/**Computes HOG descriptors over whole image img*/
void compute_hog(Mat &img, Mat& descriptor, HOGDescriptor hog, int step_w, int step_h, int max_scale){
	
	Mat image = img.clone();
	int scale = 1;					//parameter to resize image (pyramid) - first set to 1 (original size)
	int scale_factor = 2;			//scale factor (scale gets multiplied by this factor); minimum = 2
	vector< float > descriptors;
	do {
		vector<float>winDescriptor;
		hog.compute(image, winDescriptor, Size(step_w, step_h));   //computes HOG descriptor
		descriptors.insert(descriptors.end(), winDescriptor.begin(), winDescriptor.end());

		scale = scale * scale_factor;      //update of scale parameter
		pyrDown(image, image, Size((image.cols + 1) / scale_factor, (image.rows + 1) / scale_factor));     //creation of multiscale pyramid (default = half width, half height)
	} while (scale <= max_scale);

	descriptor.push_back(Mat(descriptors).clone());
}

/**Computes HOG descriptors over whole image img and obtain the result according to Pegasos format*/
void compute_hog_pegasos(Mat &img, simple_sparse_vector& descriptor, uint &dimension, HOGDescriptor hog, int step_w, int step_h, int max_scale){

	Mat image = img.clone();
	uint key = 0;

	int scale = 1;					//parameter to resize image (pyramid) - first set to 1 (original size)
	int scale_factor = 2;			//scale factor (scale gets multiplied by this factor); minimum = 2
	do {
		vector<float>winDescriptor;
		hog.compute(image, winDescriptor, Size(step_w, step_h));   //computes HOG descriptor

		for (int i = 0; i < winDescriptor.size(); i++){

			//if (winDescriptor.at(i)!=0){
			//descriptor.addElement(key, winDescriptor.at(i));
			descriptor.addElement(winDescriptor.at(i));
			//}
			key++;
		}

		scale = scale * scale_factor;      //update of scale parameter
		pyrDown(image, image, Size((image.cols + 1) / scale_factor, (image.rows + 1) / scale_factor));     //creation of multiscale pyramid (default = half width, half height)
	} while (scale <= max_scale);

	dimension = key;
}

/**Computes HAAR descriptors over whole image img*/
void compute_haar(Mat& img, Mat& descriptor, CvHaarEvaluator eval, int step_w, int step_h, int overlapping_step, Size finalROI_size){

	Mat gray = img.clone();
	vector< float > descriptors;

	for (int i = 0; i < gray.rows - step_h + 1; i += overlapping_step)
	{
		for (int j = 0; j < gray.cols - step_w + 1; j += overlapping_step)
		{
			Mat imageROI = gray(cv::Rect(j, i, step_w, step_h));  //select image ROI (window size)
			resize(imageROI, imageROI, finalROI_size);
			eval.setImage(imageROI, 0, 0);
			for (int i = 0; i < eval.features.size(); i++){
				float result = eval.operator()(i, 0);
				descriptors.push_back(result);
			}
		}
	}

	descriptor.push_back(Mat(descriptors).clone());
}

/**Computes HAAR descriptors over whole image img and obtain the result according to Pegasos format*/
void compute_haar_pegasos(Mat& img, simple_sparse_vector& descriptor, uint &dimension, CvHaarEvaluator eval, int step_w, int step_h, int overlapping_step, Size finalROI_size){

	Mat gray = img.clone();
	uint key = 0;

	for (int i = 0; i < gray.rows - step_h + 1; i += overlapping_step)
	{
		for (int j = 0; j < gray.cols - step_w + 1; j += overlapping_step)
		{
			Mat imageROI = gray(cv::Rect(j, i, step_w, step_h));  //select image ROI (window size)
			resize(imageROI, imageROI, finalROI_size);
			eval.setImage(imageROI, 0, 0);
			for (int i = 0; i < eval.features.size(); i++){
				
				float result = eval.operator()(i, 0);
				//if (result != 0){
					//descriptor.addElement(key, result);
				descriptor.addElement(result);
				//}
				key++;

			}
		}
	}

	dimension = key;

}

/**Train SVM classifier according to descriptors and labels, and saves model*/
void train_svm(int type, Mat& gradient_lst, vector< int > labels, const string& output_model_name, double &precision, double &recall, double &f_score, double &best_ht, double C_value){

	clog << "Start training...";
	SVM svm;
	CvSVMParams params;
	/* Default values to train SVM */
	//params.coef0=0.0;
	//params.degree=0;
	//params.term_crit = TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 20000, 1e-7);
	//params.gamma = 0;
	params.kernel_type = SVM::LINEAR;
	params.nu = 0.5;
	//params.p = 0.1; // for EPSILON_SVR, epsilon in loss function?

	// -- Use weights to optimize unbalanced datasets (much more negatives than positive samples)
	vector<double> weights {0.500, 0.500};
	double positives = countNonZero(labels);	//count positive samples
	double negatives = labels.size() - positives;	//count negative samples
	if (positives==labels.size()){
		params.svm_type = 102;
	}
	else{
	params.svm_type = type; // C_SVC=100 //NU_SVC=101; ONE_CLASS=102; // EPSILON_SVR=103; // may be also NU_SVR=104; // do regression task

	double neg_weight = positives / labels.size();
	double pos_weight = 1 - neg_weight;

	weights = {neg_weight, pos_weight};

	}

	CvMat class_weights = Mat(weights);
	params.class_weights = &class_weights;
	

	bool crossValidation = false;
	if (!crossValidation){
		params.C = C_value; // From paper, soft classifier C=0.01
		svm.train(gradient_lst, (Mat)labels, Mat(), Mat(), params);
	}
	else{
	//// -- adaptive C with cross-validation
	ParamGrid Cgrid = ParamGrid(0.01, 1, 10/3); // (10/3 originally); set third parameter to zero if no optimization of C

	int k_fold = gradient_lst.rows;
	//double positives = countNonZero(labels);	//count positive samples
	//int k_fold = round(sqrt(positives));
	//if (k_fold < 2){ k_fold = 2; }
	
	svm.train_auto(gradient_lst, (Mat)labels, Mat(), Mat(), params, k_fold, Cgrid, CvSVM::get_default_grid(CvSVM::GAMMA), CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
	}
	
	clog << "...[done]" << endl;

	svm.save(output_model_name.c_str());

	// Once the model is trained, it is possible to assess how it fits the training data
	bool evaluate = true;
	
	f_score = 0.0;

	if (evaluate){
	Ptr<SVM> new_model=new SVM;
	new_model->load(output_model_name.c_str());

	vector<double>ht{ 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

	best_ht = ht[0];

	vector<int>TPs(ht.size());
	vector<int>FPs(ht.size());
	vector<int>FNs(ht.size());

	for (int i = 0; i < gradient_lst.rows; i++){
	
		float prediction = new_model->predict(gradient_lst.row(i),true);

		for (int j = 0; j<ht.size(); j++){	//Test several hyperplane thresholds and obtain which fits best in the training data
			if (prediction<ht[j]){

				if (labels[i] == 1){
					TPs[j] += 1;



				}
				else{
					FPs[j] += 1;


				}

			}
			else{
				if (labels[i] == 1){
					FNs[j] += 1;

				}
			}

			if (i == gradient_lst.rows - 1){


				double new_f_score = (double)(2 * TPs[j]) / (double)((2 * TPs[j]) + FPs[j] + FNs[j]);

				if (new_f_score > f_score){
					f_score = new_f_score;
					precision = (double)TPs[j] / (double)(TPs[j] + FPs[j]);
					recall = (double)TPs[j] / (double)(TPs[j] + FNs[j]);
					best_ht = ht[j];
				}


			}

		}
	
	}

	}

	else{
		
		precision = 0.0;
		recall = 0.0;
	
	}
}

/**Load .csv contents to a Mat*/
void loadCsv(string filename, Mat &data){
	if (csv.read_csv(filename.c_str()) == -1){
		cout << "Error - .csv file couldn't be loaded" << endl; exit(-1);
	};
	const CvMat* data_ptr = csv.get_values();  //matrix with .csv data
	data = Mat(data_ptr);
}

/**Save a Mat into a .csv file*/
void saveMatToCsv(Mat &matrix, string filename){
	ofstream outputFile(filename);
	outputFile << format(matrix, "CSV") << endl;
	outputFile.close();
}

/**Save main HOG descriptor variables in a .yml to later reuse them*/
void saveDescriptorVariablesHOG(HOGDescriptor hog, int step_w, int step_h, int num_comb, int totalComb, int max_scale){

	string filename = "descriptorParams";
	if (totalComb>1){
		filename = to_string(num_comb) + "_" + filename + ".yml";

	}
	else{
		filename = filename + ".yml";

	}

	FileStorage fs(filename, FileStorage::WRITE);
	fs << "Descriptor" << "HOG";
	fs << "HOG_params";
	fs << "{";
	fs << "Cell_size" << hog.cellSize;
	fs << "Block_size" << hog.blockSize;
	fs << "Block_stride" << hog.blockStride;
	fs << "Number_bins" << hog.nbins;
	fs << "Window_width" << hog.winSize.width;
	fs << "Window_height" << hog.winSize.height;
	fs << "Window_horizontal_stride" << step_w;
	fs << "Window_vertical_stride" << step_h;
	fs << "Max_scale" << max_scale;
	fs << "}";

	fs.release();
}

/**Save main HAAR descriptor variables in a .yml to later reuse them*/
void saveDescriptorVariablesHAAR(int slidingWin_width, int slidingWin_height, int sliding_step, Size finalROI_size, int dimension, int num_comb,int totalComb){

	string filename = "descriptorParams";
	if (totalComb>1){
		filename = to_string(num_comb) + "_" + filename + ".yml";
	
	}
	else{
		filename = filename + ".yml";
	
	}

	FileStorage fs(filename, FileStorage::WRITE);
	fs << "Descriptor" << "HAAR";
	fs << "HAAR_params";
	fs << "{";
	fs << "Sliding_window_width" << slidingWin_width;
	fs << "Sliding_window_height" << slidingWin_height;
	fs << "Sliding_window_stride_step" << sliding_step;
	fs << "Final_ROI_size" << finalROI_size;
	fs << "Feature_Vector_Length" << (int) dimension;
	fs << "}";

	fs.release();
}

void saveClassNames(vector <string> class_names){
	FileStorage fs("classNames.yml", FileStorage::WRITE);
	for(int i = 0; i < class_names.size(); i++) {
		if (i==0){
	fs << "Class names " << "[";
		}
	fs << class_names[i];
	if (i == class_names.size() - 1){
	fs << "]";
	}
	}
	

	fs.release();

}

void saveDetectionsToTxt(Mat &foundPositions, string filename){

	vector <string> sequence_filenames;
	string inputFile = "D:/u/abn/secuencias/directorio_prueba";
	readDirectory(inputFile,sequence_filenames); //inputFolder
	ofstream outputfile;
	vector <string> class_names = {"bus", "car trailer", "car","cyclist","lorry trailer","lorry","motorbike","pedestrian","van", "vantrailer"}; // lo conoceriamos

	for (int i = 0; i < sequence_filenames.size();i++){
	
		if (i==0){
			outputfile.open(filename);
		}
		string image_name = sequence_filenames[i].erase(0, inputFile.size() + 1);
		Mat frameFoundPositions = foundPositions.row(i);
		cv::Mat_<Vec3i>::iterator it = frameFoundPositions.begin<Vec3i>();
		cv::Mat_<Vec3i>::iterator itend = frameFoundPositions.end<Vec3i>();
		for (; it != itend; ++it){
			if ((*it)[0] != 0 || (*it)[1] != 0){
				
				outputfile << image_name + " " + class_names[(*it)[2]] + " " + to_string(-1) + " " + to_string((*it)[0]) + " " + to_string((*it)[1]) + " " +
					to_string(-1) + " " + to_string(-1) + "\n";
			}
		}
	}

	outputfile.close();

}

/**Read info from a file exported with Pegasos Code and store it in a simple_sparse_vector*/
void readVector(string filename, simple_sparse_vector &data, uint &dimension){

	std::ifstream data_file(filename.c_str());
	if (!data_file.good()) {
		std::cerr << "error w/ " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	string buf;

	getline(data_file, buf);

	size_t parenthesis_pos = buf.find("(");
	size_t comma_pos = buf.find(",");
	uint count = 0;
	//size_t second_pos = buf.find("(",parenthesis_pos+1);
	while (parenthesis_pos != string::npos){

		//uint key = stoi(buf.substr(parenthesis_pos + 1, comma_pos - parenthesis_pos + 1));
		parenthesis_pos = buf.find("(", parenthesis_pos + 1);

		string substr = buf.substr(comma_pos + 1, parenthesis_pos - 3 - comma_pos);

		float val = stof(buf.substr(comma_pos + 1, parenthesis_pos - 3 - comma_pos));

		comma_pos = buf.find(",", comma_pos + 1);

		//data.addElement(key, val);
		data.addElement(val);
		count++;
	}
	dimension = count;
	data_file.close();
}


struct svm_data
{
	//General attributes
	int svm_type;
	vector <int> output_labels;
	string svm_name;
	int numSVM;
	int class_label;
	double regularization;

	//Specific OpenCV arguments
	CvMat input_vectors;


	//Specific Pegasos arguments
	simple_sparse_vector *train_data;
	string svm_name_input;
	uint num_features;
	int max_iter;
	vector<int> batch_sizes;
	vector<vector<int>>samples_positions;

};


void *train_svm_thread_pegasos(void *threadarg){

	bool evaluate = true;

	simple_sparse_vector *Dataset;
	vector <int> Labels;
	uint dimension;
	int numClassifier;
	string input_filename;
	string output_filename;
	int num_class;
	double lambda;
	int max_iterations;

	vector<vector<int>> samples_pos;
	vector<int>batch_size;

	struct svm_data *input_data;

	Sleep(1);
	input_data = (struct svm_data *) threadarg;
	Dataset = input_data->train_data;
	Labels = input_data ->output_labels;
	dimension = input_data->num_features;
	numClassifier = input_data->numSVM;
	input_filename = input_data->svm_name_input;
	output_filename = input_data->svm_name;
	num_class = input_data->class_label;
	lambda = input_data->regularization;
	max_iterations = input_data->max_iter;

	samples_pos = input_data->samples_positions;
	batch_size = input_data->batch_sizes;

	printf("Start training SVM #%d...\n", numClassifier);

	// --- Set input parameters for training with Pegasos Code (in this case, they are the same for all classes
	// and classifiers: if we want to make them different, this part has to be inside a for loop, in the main)

	double max_th_lambda = 3;	//Boundary in reduction of lambdaEff
	double max_lambda = 4;	//Minimum value of lambdaEff when it reaches the boundary

	double improve_th = 0.00005;	//If improvement in the cost function is less than this value, the training ends
								//assuming that now it is optimized

	double class_th = 1;		//Threshold to use in the loss function, a value >1 makes more possibilities to have a larger margin
	double decision_th = 7;

	int step_0 = (int)round(batch_size[0] *1/ 4);
	int step_1 = (int)round(batch_size[1] *1/ 4);

	//Start program

	int eta_rule_type = 0; double eta_constant = 0.0;
	int projection_rule = 0; double projection_constant = 0.0;

	//uint num_examples = Labels.size();

	// Initialization of classification vector
	WeightVector W(dimension);	//Use this default constructor for first training; use other constructor for model importation
	WeightVector BestW(dimension);
	double best_obj; //the zero solution
	double best_loss;
	int best_iter = -1;

	if (input_filename != "") {

		// read weight vector
		std::ifstream model_file_import(input_filename.c_str());
		if (!model_file_import.good()) {
			std::cerr << "error w/ " << input_filename << std::endl;
			exit(EXIT_FAILURE);
		}
		W = WeightVector(dimension, model_file_import);
		model_file_import.close();
	}

	Mat statistics(15, 7, CV_64FC1, Scalar(-1));

	double current_obj_value = 0.0;
	double current_loss_value = 0.0;

	//double TP;
	//double FP;
	//double FN;

	double lambdaEff = lambda; //Initialization of real lambda, which will be descending

	int it0_begin = 0;
	int it1_begin = 0;

	// ---------------- Main Loop -------------------
	for (int i = 0; i < max_iterations; ++i) {

		long startTime = get_runtime();

		//std::cout << "Iteration # " << i << std::endl;

		// Calculate objective value
		//TP = 0.0;
		//FP = 0.0;
		//FN = 0.0;

		double new_norm_value = W.snorm();
		double new_obj_value = new_norm_value * lambdaEff / 2.0;
		double new_loss_value = 0.0;
		for (uint j = 0; j < Labels.size(); ++j) {
			double prediction = W*Dataset[j];
			double cur_loss = class_th - Labels[j] * (prediction);
			if (cur_loss < 0.0) cur_loss = 0.0;
			
			//if (prediction > decision_th){

			//	if (Labels[j] == +1){
			//		TP += 1.0;
			//	}
			//	else{
			//		FP += 1.0;
			//	}
			//}
			//else{

			//	if (Labels[j] == +1){
			//		FN += 1.0;
			//	}

			//}
			
			new_loss_value += cur_loss / Labels.size();
			new_obj_value += cur_loss / Labels.size();
		}

		if (i > 0){

			if (i == 1){
				best_obj = new_obj_value;
				best_iter = 1;
			}

			if (new_obj_value <= best_obj){
				best_iter = i;
				best_obj = new_obj_value;
				best_loss = new_loss_value;
				BestW = W;
			}
		}

		//std::cout << "New Loss value: " << new_loss_value << std::endl;
		//std::cout << "New Obj value: " << new_obj_value << std::endl;

		//std::cout << "True positives: " << TP << std::endl;
		//std::cout << "False positives: " << FP << std::endl;
		//std::cout << "False negatives: " << FN << std::endl;

		//std::cout << "Lambda: " << lambdaEff << std::endl;

		if (i>0 && (((current_obj_value - new_obj_value) >= 0) && ((current_obj_value - new_obj_value)<improve_th))){

			statistics.at<double>(0, 0) = best_obj;
			statistics.at<double>(0, 1) = best_loss;
			statistics.at<double>(2, 0) = i;
			break;
		}

		else{
			current_obj_value = new_obj_value;
			current_loss_value = new_loss_value;
			if (batch_size[0] != samples_pos[0].size()){
			it0_begin = it0_begin + step_0;
			}
			if (batch_size[1] != samples_pos[1].size()){
			it1_begin = it1_begin + step_1;
			}

			//Experimental (reduce lambda a little in each iteration)
			//if (i > 0){
			//	lambdaEff = lambdaEff + (lambda / 200);
			//	if (lambdaEff >= max_th_lambda){
			//		lambdaEff = max_lambda;
			//	}
			//}
		}

		vector<int> frame_selection; //Concrete frames selected from the dataset
		vector<int>::iterator it_begin = frame_selection.begin();

		int count_positive = 0;
		int count_negative = 0;

		if (batch_size[1] == samples_pos[1].size()){

			it_begin = frame_selection.insert(it_begin, samples_pos[1].begin(), samples_pos[1].end());
			count_positive = batch_size[1];

		}

		else if (it1_begin > samples_pos[1].size() - 1){

			it1_begin = it1_begin - samples_pos[1].size();

		}

		if (batch_size[0] == samples_pos[0].size()){

			it_begin = frame_selection.insert(it_begin, samples_pos[0].begin(), samples_pos[0].end());
			count_negative = batch_size[0];
		}

		else if (it0_begin > samples_pos[0].size() - 1){

			it0_begin = it0_begin - samples_pos[0].size();

		}

		int it0 = it0_begin;
		int it1 = it1_begin;

		while (count_positive < batch_size[1] || count_negative < batch_size[0]){

			if (it0 > samples_pos[0].size() - 1){

				it0 = 0;

			}
			if (it1 > samples_pos[1].size() - 1){

				it1 = 0;

			}

			if (count_negative < batch_size[0] && !(std::find(frame_selection.begin(), frame_selection.end(), samples_pos[0][it0]) != frame_selection.end())){
				frame_selection.push_back(samples_pos[0][it0]);
				count_negative++;
				it0++;
			}

			if (count_positive < batch_size[1] && !(std::find(frame_selection.begin(), frame_selection.end(), samples_pos[1][it1]) != frame_selection.end())){
				frame_selection.push_back(samples_pos[1][it1]);
				count_positive++;
				it1++;
			}



		}

		std::random_shuffle(frame_selection.begin(), frame_selection.end());

		// learning rate
		double eta;
		if (eta_rule_type == 0) { // Pegasos eta rule
			eta = 1 / (lambdaEff * (i + 2));
		}
		else if (eta_rule_type == 1) { // Norma rule
			eta = eta_constant / sqrt(i + 2);
			// solve numerical problems
			//if (projection_rule != 2)
			W.make_my_a_one();
		}
		else {
			eta = eta_constant;
		}

		// gradient indices and losses
		std::vector<uint> grad_index;
		std::vector<double> grad_weights;

		// calc sub-gradients
		//for (int j = 0; j < exam_per_iter; ++j) {
		for (int j = 0; j < frame_selection.size(); ++j){

			// choose random example

			//uint r = ((int)rand()) % num_examples;

			uint r = frame_selection[j];

			// calculate prediction
			double prediction = W*Dataset[r];

			// calculate loss
			double cur_loss = class_th - Labels[r] * prediction;
			if (cur_loss < 0.0) cur_loss = 0.0;

			// and add to the gradient
			if (cur_loss > 0.0) {
				grad_index.push_back(r);
				//grad_weights.push_back(eta*Labels[r] / exam_per_iter);
				grad_weights.push_back(eta*Labels[r] / frame_selection.size());
			}

		}
		//}

		// scale w 
		W.scale(1.0 - eta*lambdaEff);

		// and add sub-gradients
		for (uint j = 0; j<grad_index.size(); ++j) {
			W.add(Dataset[grad_index[j]], grad_weights[j]);
		}

		// Project if needed
		if (projection_rule == 0) { // Pegasos projection rule
			double norm2 = W.snorm();
			if (norm2 > 1.0 / lambdaEff) {
				W.scale(sqrt(1.0 / (lambdaEff*norm2)));
			}
		}
		else if (projection_rule == 1) { // other projection
			double norm2 = W.snorm();
			if (norm2 > (projection_constant*projection_constant)) {
				W.scale(projection_constant / sqrt(norm2));
			}
		} // else -- no projection

		if (i == max_iterations - 1){
		
			//TP = 0.0;
			//FP = 0.0;
			//FN = 0.0;

			new_norm_value = W.snorm();
			new_obj_value = new_norm_value * lambdaEff / 2.0;
			new_loss_value = 0.0;
			for (uint j = 0; j < Labels.size(); ++j) {
				double prediction = W*Dataset[j];
				double cur_loss = class_th - Labels[j] * (prediction);
				if (cur_loss < 0.0) cur_loss = 0.0;

				//if (prediction > decision_th){

				//	if (Labels[j] == +1){
				//		TP += 1.0;
				//	}
				//	else{
				//		FP += 1.0;
				//	}
				//}
				//else{

				//	if (Labels[j] == +1){
				//		FN += 1.0;
				//	}

				//}

				new_loss_value += cur_loss / Labels.size();
				new_obj_value += cur_loss / Labels.size();
			}

			if (i == 0){
				best_iter = 0;
				best_obj = new_obj_value;
			}

			if (new_obj_value <= best_obj){
				best_iter = i;
				best_obj = new_obj_value;
				best_loss = new_loss_value;
				BestW = W;
			}

			statistics.at<double>(0, 0) = best_obj;
			statistics.at<double>(0, 1) = best_loss;
			statistics.at<double>(2, 0) = i;
		
		}

		long endTime = get_runtime();

		long train_time = endTime - startTime;

		//std::cout << "Time elapsed: " << train_time << "\n" << std::endl;

	}

	// update timeline
	//endTime = get_runtime();	//optional
	//train_time = endTime - startTime;	//optional
	//startTime = get_runtime();	//optional

	// Calculate objective value (optional)
	//norm_value = W.snorm();
	//obj_value = norm_value * lambdaEff / 2.0;
	//loss_value = 0.0;

	if (evaluate){
		

		vector<double>ht{ decision_th, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 30.0 };

		double precision;
		double recall;
		double max_fscore;
		double min_FP;
		double best_ht = decision_th;

		vector<int>TPs(ht.size());
		vector<int>FPs(ht.size());
		vector<int>FNs(ht.size());

		for (int i = 0; i < Labels.size(); i++){

			double prediction = BestW*Dataset[i];

			for (int j = 0; j<ht.size(); j++){	//Test several hyperplane thresholds and obtain which fits best in the training data
				statistics.at<double>(j, 6) = ht[j];
				if (prediction>ht[j]){

					if (Labels[i] == 1){
						TPs[j] += 1;

					}
					else{
						FPs[j] += 1;
					}

				}
				else{
					if (Labels[i] == 1){
						FNs[j] += 1;

					}
				}

				if (i == Labels.size() - 1){

					double f_score = (double)(2 * TPs[j]) / (double)((2 * TPs[j]) + FPs[j] + FNs[j]);
					double new_precision = (double)TPs[j] / (double)(TPs[j] + FPs[j]);
					double new_recall = (double)TPs[j] / (double)(TPs[j] + FNs[j]);

					if (j == 0){

						max_fscore = f_score;
						min_FP = FPs[j];
						precision = new_precision;
						recall = new_recall;

					}

					statistics.at<double>(j, 2) = new_precision;
					statistics.at<double>(j, 3) = new_recall;
					statistics.at<double>(j, 4) = f_score;
					statistics.at<double>(j, 5) = FPs[j];

					if (TPs[j] + FNs[j] == 0){
					
						if (FPs[j]<min_FP){
						
							min_FP = FPs[j];
							best_ht = ht[j];
							max_fscore = f_score;
							precision = new_precision;
							recall = new_recall;
						
						}
					
					}

					else{
					if (f_score > max_fscore){
						max_fscore = f_score;
						precision = new_precision;
						recall = new_recall;
						best_ht = ht[j];
						min_FP = FPs[j];
					}
					}

				}

			}

		}

		if (isnan(precision)){
			precision = -1.0;
		}
		if (isnan(recall)){
			recall = -1.0;
		}
		if (isnan(max_fscore)){
			max_fscore = -1.0;
		}
		
		string statistics_name = output_filename.substr(0,output_filename.size()-4) + ".csv";

		saveMatToCsv(statistics, statistics_name);


		//Freeze the actual situation about shared variables and update them one thread by one
		pthread_mutex_lock(&mutex_training);

		classes_precisions.at<double>(num_class, numClassifier) = precision;
		classes_recalls.at<double>(num_class, numClassifier) = recall;
		classes_fscores.at<double>(num_class, numClassifier) = max_fscore;
		thresholds.at<double>(num_class, numClassifier) = best_ht;

		pthread_mutex_unlock(&mutex_training);

	}

	printf("... SVM #%d [done]\n", numClassifier);

	// finally, print the model to the model_file (important: introduce a name)
	if (output_filename != "noModelFile") {
		std::ofstream model_file(output_filename);
		if (!model_file.good()) {
			std::cerr << "error w/ " << output_filename << std::endl;
			exit(EXIT_FAILURE);
		}
		BestW.print(model_file);
		model_file.close();
	}

	pthread_exit((void*)numClassifier);
	return 0;
}

void *train_svm_thread(void *threadarg){

	//initialize attributes of the struct
	int type;
	CvMat gradient_lst;
	vector <int> labels;
	string output_model_name;
	int numClassifier;
	int num_class;
	double C_value;
	struct svm_data *input_data;

	Sleep(1);
	input_data = (struct svm_data *) threadarg;
	type = input_data->svm_type;
	//gradient_lst = input_data->input_vectors.clone();
	gradient_lst = input_data->input_vectors;
	labels = input_data->output_labels;
	output_model_name = input_data->svm_name;
	numClassifier = input_data->numSVM;
	num_class = input_data->class_label;
	C_value = input_data->regularization;
	
	//Configure SVM parameters and train the classifier
	printf("Start training SVM #%d...\n", numClassifier);
	SVM svm;
	CvSVMParams params;

	params.kernel_type = SVM::LINEAR;
	params.nu = 0.5;

	// -- Use weights to optimize unbalanced datasets (much more negatives than positive samples)
	vector<double> weights {0.500, 0.500};
	double positives = countNonZero(labels);	//count positive samples
	double negatives = labels.size() - positives;	//count negative samples
	if (positives==labels.size()){
		params.svm_type = 102;
	}
	else{
	params.svm_type = type; // C_SVC=100 //NU_SVC=101; ONE_CLASS=102; // EPSILON_SVR=103; // may be also NU_SVR=104; // do regression task

	double neg_weight = positives / labels.size();
	double pos_weight = 1 - neg_weight;

	weights = { neg_weight, pos_weight };

	}

	CvMat class_weights = Mat(weights);
	params.class_weights = &class_weights;

	params.C = C_value;

	//// -- adaptive C with cross-validation
	//ParamGrid Cgrid = ParamGrid(0.01, 1, 10/3); // (10/3 originally); set third parameter to zero if no optimization of C

	//int k_fold = gradient_lst.rows;

	//svm.train_auto(gradient_lst, (Mat)labels, Mat(), Mat(), params, k_fold, Cgrid, CvSVM::get_default_grid(CvSVM::GAMMA), CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);

	svm.train(Mat(&gradient_lst), (Mat)labels, Mat(), Mat(), params);

	printf("... SVM #%d [done]\n",numClassifier);
	
	svm.save(output_model_name.c_str());

	// Once the model is trained, it is possible to assess how it fits the training data
	bool evaluate = true;
	
	if (evaluate){

		vector<double>ht{ 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };

		Ptr<SVM> new_model = new SVM;
		new_model->load(output_model_name.c_str());

		double precision = 0.0;
		double recall = 0.0;
		double max_fscore = 0.0;
		double best_ht = ht[0];

		vector<int>TPs(ht.size());
		vector<int>FPs(ht.size());
		vector<int>FNs(ht.size());

		for (int i = 0; i < gradient_lst.rows; i++){

			float prediction = new_model->predict(Mat(&gradient_lst).row(i), true);

			for (int j = 0; j<ht.size(); j++){	//Test several hyperplane thresholds and obtain which fits best in the training data
			if (prediction<ht[j]){	

				if (labels[i] == 1){
					TPs[j] += 1;



				}
				else{
					FPs[j] += 1;


				}

			}
			else{
				if (labels[i] == 1){
					FNs[j] += 1;

				}
			}

			if (i == gradient_lst.rows - 1){
			

				double f_score = (double) (2 * TPs[j]) / (double)((2 * TPs[j]) + FPs[j] + FNs[j]);
			
				if (f_score > max_fscore){
					max_fscore = f_score;
					precision = (double)TPs[j] / (double)(TPs[j] + FPs[j]);
					recall = (double)TPs[j] / (double)(TPs[j] + FNs[j]);
					best_ht = ht[j];
				}


			}

			}

		}

		//Freeze the actual situation about shared variables and update them one thread by one
		pthread_mutex_lock(&mutex_training);

		classes_precisions.at<double>(num_class, numClassifier) = precision;
		classes_recalls.at<double>(num_class, numClassifier) = recall;
		classes_fscores.at<double>(num_class, numClassifier) = max_fscore;
		thresholds.at<double>(num_class, numClassifier) = best_ht;

		pthread_mutex_unlock(&mutex_training);

	}

	pthread_exit((void*) numClassifier);
	return 0;
}

int main(int argc, char** argv)
{

	help();		//show program info

	//-------------------------------------INPUT PARAMETERS------------------------------------------------------//

	double duration = static_cast<double>(cv::getTickCount());

	vector <int> firstClassifier = { 0 };//{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };	//With this input, the same classes (or at least, the same number of classes) are assumed in all training sessions

	//vector <vector<int>> excluded_classifiers = { { 320, 56, 87, 13 }, {27, 15, 185} };

	//bool resultado = std::find(excluded_classifiers[1].begin(), excluded_classifiers[1].end(), 87) == excluded_classifiers[1].end();

	bool useThreads = true;											//Use threads for training

	bool pegasosTraining = false;									//Train SVM with Pegasos Code, which allows online training

	vector<string> modelsDirectory = { "" };						//Group of folders which contain the models to be updated (only works with Pegasos Code and only for online training mode)
																	//They must include the final slash "/" and if it is the case, the prefix "number_"

	bool saveVectors = false;										//Store feature vectors in external files (this is invalidated if training database is composed by precomputed features)

	bool onlyComputeFeatures = false;								//Only compute feature vectors or also train (if true, set "saveVectors" to true)

	//Input parameters to test and combine

	//Input folders containing training images (represented by either own images or computed feature vectors)
	vector<vector<string>> inputFolders = {  
		{ "C:/Andres/OrdenadorGTI/secuencias/Asturias_Night1Training1", "C:/Andres/OrdenadorGTI/secuencias/Asturias_Night1Training1brillo" }  };

	//Groups of folders with activations matrices, previously computed with GENERATION mode
	vector <vector<string>> activationFolders = { { "C:/Andres/OrdenadorGTI/ActivationsDirectory/Asturias_Night1Training1_21x26_bueno", "C:/Andres/OrdenadorGTI/ActivationsDirectory/Asturias_Night1Training1_21x26_brillo" } };

	//Descriptor extractor (HOG or HAAR) and its properties (with default values)
	string descriptorType = "HOG";

	//- for HOG -
	vector<int> hog_window_widths = {16};  //64;	//hog window width (16 for dgt images)
	vector<int> hog_window_heights = {16}; //48;	//hog window height
	vector<int> steps_w = {16};
	vector<int> steps_h = {16};
	vector<int> max_scales = {4};				//max scale - resizes original image until this maximum is reached

	//- for HAAR -
	vector<int> haar_slidingWin_widths = {96};			//haar computing window width
	vector<int> haar_slidingWin_heights = {72};			//haar computing window height
	vector<int> haar_sliding_steps = {64};				//haar computing sliding step
	vector <Size> haar_finalROI_sizes = { Size(32, 24) };	//haar final ROI size (resized from computing size)	

	//- SVM parameters -
	vector<double> regularization_terms = {0.01}; //C values in OpenCV, lambdas in Pegasos

	//Pegasos
	vector <int> max_iters = {18000};	//Use the same number of inputs as in the regularization

	//- Dataset Balance -
	bool balanceDataset = true;
	int max_batch_size = 5000;
	double division = 0.4;	//Typically 0.5, 0.4 or 0.3

	//-----------------------------------------------------------------------------------------------------------//

	//-- Command line arguments parsing
	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--help"))
		{
			help2();
			cout << "Press enter to exit...";
			cin.ignore();
			return 0;
		}
		if (!strcmp(argv[i], "-input"))
		{
			inputFolders.clear();
			int number_inputs = stoi(argv[++i]);	//number of input folders (if training images or feature vectors are split in several folders)
			for (int n = 0; n < number_inputs; n++){
				inputFolders[n].clear();
				int num_folders = stoi(argv[++i]);
				for (int m = 0; m < num_folders; m++){
					inputFolders[n].push_back(argv[++i]);
				}
			}
		}
		else if (!strcmp(argv[i], "-activations"))
		{
			activationFolders.clear();
			int number_folders = stoi(argv[++i]);	//number of activations folders (if training images are split in several folders)
			for (int n = 0; n < number_folders; n++){
				activationFolders[n].clear();
				int num_folders = stoi(argv[++i]);
				for (int m = 0; m < num_folders; m++){
					activationFolders[n].push_back(argv[++i]);
				}
				
			}
		}
		else if (!strcmp(argv[i]," -models")){
			modelsDirectory.clear();
			int number_directories = stoi(argv[++i]);	//Each directory contains pre-trained models with a determined configuration
			for (int n = 0; n < number_directories;n++){
				modelsDirectory.push_back(argv[++i]);
			}

		}
		else if (!strcmp(argv[i], "-descriptor"))
		{
			descriptorType = string(argv[i + 1]);
		}
		else if (!strcmp(argv[i], "-win_size"))
		{
			if (descriptorType == "HOG"){

				hog_window_widths.clear();
				
				int number_hog_widths = stoi(argv[++i]);
				for (int n = 0; n < number_hog_widths; n++){
					hog_window_widths.push_back(stoi(argv[++i]));
				}

				hog_window_heights.clear();

				int number_hog_heights = stoi(argv[++i]);
				for (int n = 0; n < number_hog_heights; n++){
					hog_window_heights.push_back(stoi(argv[++i]));
				}

				max_scales.clear();

				int num_max_scales = stoi(argv[++i]);
				for (int n = 0; n < num_max_scales;n++){
					max_scales.push_back(stoi(argv[++i]));
				}

			}
			else if (descriptorType == "HAAR"){

				haar_slidingWin_widths.clear();

				int number_haar_widths = stoi(argv[++i]);

				for (int n = 0; n < number_haar_widths; n++){
					haar_slidingWin_widths.push_back(stoi(argv[++i]));
				}

				haar_slidingWin_heights.clear();

				int number_haar_heights = stoi(argv[++i]);

				for (int n = 0; n < number_haar_heights; n++){
					haar_slidingWin_heights.push_back(stoi(argv[++i]));
				}

			}
		}
		else if (!strcmp(argv[i], "-win_step"))
		{
			if (descriptorType == "HOG"){

				steps_w.clear();

				int num_steps_w = stoi(argv[++i]);

				for (int n = 0; n < num_steps_w; n++){
					steps_w.push_back(stoi(argv[++i]));
				}

				steps_h.clear();

				int num_steps_h = stoi(argv[++i]);

				for (int n = 0; n < num_steps_h; n++){
					steps_h.push_back(stoi(argv[++i]));
				}

			}
			else if (descriptorType == "HAAR"){

				haar_sliding_steps.clear();

				int num_haar_sliding_steps = stoi(argv[++i]);

				for (int n = 0; n < num_haar_sliding_steps; n++){
					haar_sliding_steps.push_back(stoi(argv[++i]));
				}
			}
		}
		else if (!strcmp(argv[i], "-haar_final_win"))
		{
			if (descriptorType == "HAAR"){

				haar_finalROI_sizes.clear();

				int num_haar_finalROI_sizes = stoi(argv[++i]);

				if (num_haar_finalROI_sizes % 2==0){
				
					for (int n = 0; n < num_haar_finalROI_sizes; n++){

						int w = stoi(argv[++i]);
						int h = stoi(argv[++i]);

						haar_finalROI_sizes.push_back(Size(w, h));
					}
				
				}

			}
		}
		else if (!strcmp(argv[i], "-num_classifier"))
		{

			firstClassifier.clear();
			int num_firstClassifiers = stoi(argv[++i]);
			for (int n = 0; n < num_firstClassifiers; n++){
				firstClassifier.push_back(stoi(argv[++i])); // Need a way to verify this
			}
		}
		else if (!strcmp(argv[i], "-regularization"))
		{
			regularization_terms.clear();
			int num_regul = stoi(argv[++i]);
			for (int n = 0; n < num_regul; n++){
				regularization_terms.push_back(stoi(argv[++i])); // Need a way to verify this
			}
		}
		else if (!strcmp(argv[i], "-iterations")){
		
			max_iters.clear();
			int num_maxiter = stoi(argv[++i]);
			for (int n = 0; n < num_maxiter; n++){
				max_iters.push_back(stoi(argv[++i])); // Need a way to verify this
			}
		
		}
		else if (!strcmp(argv[i], "-useThreads")){
			useThreads = true;
		}
		else if (!strcmp(argv[i], "-pegasosTraining")){
			pegasosTraining = true;
		}
		else if (!strcmp(argv[i], "-saveVectors")){
			saveVectors = true;
		}
		else if (!strcmp(argv[i], "-onlyComputeFeatures")){
			onlyComputeFeatures = true;
		}
		else if (!strcmp(argv[i], "-balancedTraining")){
			balanceDataset = true;
		}
		else if (!strcmp(argv[i], "-division")){
			division=stod(argv[++i]);	
		}
		else if (!strcmp(argv[i], "-max_batch_size")){
			max_batch_size = stoi(argv[++i]);
		}

	}

	//-- Determine number of possible combinations for massive testing of parameters

	int i1_max;
	int i2_max;
	int i3_max;
	int i4_max;
	int i5_max;
	int i6_max;
	int i7_max = 1;
	int i8_max;
	int i9_max;

	int total_comb;

	vector <string> class_names;

	char buffer[100];		//Buffer to store feature vectors adequately: in chronological order

	//Online training and offline training must be done in different training sessions
	if (pegasosTraining && modelsDirectory[0]!= ""){
	
		i7_max = modelsDirectory.size();	//In this case, this size must be equal to the number of input groups, and activations, descriptor types
											//and SVM parameters.
											//There would not be crossed combinations, only "vertical" combinations

		total_comb = i7_max;				//Only online training sessions are allowed

		//Disable all for loops but one, whose iterator will be common for all
		i1_max = 1;
		i2_max = 1;
		i3_max = 1;
		i4_max = 1;
		i5_max = 1;
		i6_max = 1;
		i8_max = 1;
		i9_max = 1;

	}

	else{

		i6_max = inputFolders.size();			//With the current philosophy, we only consider several groups if they indicate feature configurations
												//In this case, it is very important to only consider one configuration of the feature parameters, to avoid
												//unnecesary repetitions

		//When only computing and storing features, activations files and regularization parameters are not necessary at all
		if (onlyComputeFeatures){
		
			i5_max = 1;
			i9_max = 1;
		}

		else{
		i5_max = activationFolders.size();		//Note: this is not necessary if this program is executed only to compute feature vectors: furthermore, there must be one input

		i9_max = regularization_terms.size();	//When used for Pegasos, the number of lambdas must be equal to the number of different iterations because they are aligned

		}

	//Consider HOG or Haar in the training session

	if (descriptorType == "HOG"){
		i1_max = hog_window_widths.size();
		i2_max = hog_window_heights.size();
		i3_max = 1;	//Assumed steps equal to the respective widths and heights
		i4_max = 1;	//No parameter for haar_finalROI_size
		i8_max = max_scales.size();
	}
	else{
		i1_max = haar_slidingWin_widths.size();
		i2_max = haar_slidingWin_heights.size();
		i3_max = haar_sliding_steps.size();
		i4_max = haar_finalROI_sizes.size();
		i8_max = 1;
	}

	total_comb = i1_max*i2_max*i3_max*i4_max*i5_max*i6_max*i8_max*i9_max;
	}

	int num_comb = -1; //Set to -1 if training is executed with only one configuration

	if (total_comb > 1){
		num_comb = 0;
	}

	double duration2 = 0.0;	//Chronometre for each combination

	//Start all for loops
	//Loop for folders where there are pre-trained models 
	for (int i7 = 0; i7 < i7_max; i7++){

		//Get the instant of a new combination if it starts here
		if (duration2==0.0 && total_comb>1){
			duration2 = static_cast<double>(cv::getTickCount());
		}

		//Set the same name of the previous model (including prefix) That must be done in each iteration
		//Do not introduce pre-trained models in the local folder of the training project, because they are caught
		//when the input parameter is simply ""

		size_t last_slash = modelsDirectory[i7].rfind("/");

		string svm_prefix = modelsDirectory[i7].substr(last_slash + 1, modelsDirectory[i7].size() - 1 - last_slash);

	
	//Loop for groups of databases (in form of feature vectors or images)
	for (int i6 = 0; i6 < i6_max; i6++){

		//Get the instant of a new combination if it starts here
		if (duration2 == 0.0 && total_comb>1){
			duration2 = static_cast<double>(cv::getTickCount());
		}

		//Set the same iterator of this loop or change it to the iterator of pre-trained models
		int i6_iter;

		if (pegasosTraining && modelsDirectory[0] != ""){
			i6_iter = i7;
		}
		else{
			i6_iter = i6;
		}

	// -- Load input training images or feature vectors from folders
	vector<string> filenames;
	// number of training images in each folder
	vector <int> input_sizes;
	//Input is represented by images or feature vectors (not valid to mix them: inside directories not, and between groups it may be possible)
	bool readVectors=false;
	for (int i = 0; i < inputFolders[i6_iter].size(); i++){
		vector<string> temp_filenames;
		readDirectory(inputFolders[i6_iter][i], temp_filenames);
		input_sizes.push_back(temp_filenames.size());
		for (int j = 0; j < temp_filenames.size(); j++){
			// Do not include the .info file generated with ViPER
			size_t posInfoExt = temp_filenames[j].find(".info");
			
			//Assuming that all directories in a group have only either precomputed features or images 
			if (j == 0){
				size_t posExt;
				if (!pegasosTraining){
					posExt = temp_filenames[j].find(".csv");
				}
				else{
					posExt = temp_filenames[j].find(".yml");
				}
				//In this case, it is supposed that images are represented by feature vectors
				if (posExt != string::npos){
				
					readVectors = true;
				}

			}

			if (posInfoExt == string::npos){
			filenames.push_back(temp_filenames[j]);
			}
			else{
				input_sizes[i]--; // exclude from initial estimation of the number of images the .info file 
			}
		}
		
	}

	// -- Initialize descriptor extractor (HOG or HAAR)
	
	for (int i1 = 0; i1 < i1_max; i1++){
	for (int i2 = 0; i2 < i2_max; i2++){
	for (int i3 = 0; i3 < i3_max; i3++){
	for (int i4 = 0; i4 < i4_max; i4++){
	for (int i8 = 0; i8 < i8_max; i8++){

		//Get the instant of a new combination if it starts here
		if (duration2 == 0.0 && total_comb>1){
			duration2 = static_cast<double>(cv::getTickCount());
		}

		//Set the same iterator of this loop or change it to the iterator of pre-trained models

		int i1_iter;
		int i2_iter;
		int i3_iter;
		int i4_iter;
		int i8_iter;

		if (pegasosTraining && modelsDirectory[0] != ""){
			i1_iter = i7;
			i2_iter = i7;
			i3_iter = i7;
			i4_iter = i7;
			i8_iter = i7;
		}
		else{
			i1_iter = i1;
			i2_iter = i2;
			i3_iter = i3;
			i4_iter = i4;
			i8_iter = i8;
		}

	cout << "\nNew combination of features in combination # " + to_string(num_comb) << endl;

	//for HOG -----------------------------------------------------------------------------------
	//HOGDescriptor hog(Size(), Size(16,16), Size(8,8), Size(8,8), 9);	//alternative constructor
	HOGDescriptor hog;			//hog instance
	hog.winSize = Size(hog_window_widths[i1_iter], hog_window_heights[i2_iter]);

	//for HAAR ----------------------------------------------------------------------------------
	CvFeatureParams f;
	Ptr<CvFeatureParams> haarParams = f.create(0);		//pointer to HaarFeatureParams object
	CvHaarEvaluator eval;
	eval.init(haarParams, 1, haar_finalROI_sizes[i4_iter]);

	int history = 30;
	float threshold = 16;
	bool bShadowDetection = false;	//don't realize shadow detection
	BackgroundSubtractorMOG2 bg = BackgroundSubtractorMOG2(history, threshold, bShadowDetection);
	float learningRate = 0.01;							//set to 0.01 to avoid flickering

	if (descriptorType == "HOG" && !readVectors) { saveDescriptorVariablesHOG(hog, hog_window_widths[i1_iter], hog_window_heights[i2_iter], num_comb, total_comb,max_scales[i8_iter]); }

		// -- Read training images and compute descriptors
		Mat train_features;		//matrix containing descriptors (each row = one image descriptor)
		vector <simple_sparse_vector> train_samples;	//Matrix containing descriptors in Pegasos Format (each row = one image descriptor)
		uint dimension;

		for (int i = 0; i < filenames.size(); i++){

			Mat descriptors;
			simple_sparse_vector descriptor_pegasos;

			//Compute feature vectors if samples are images
			if (!readVectors){

			Mat read_image = imread(filenames[i], 0);

			if (descriptorType == "HOG"){
				if (pegasosTraining){
					compute_hog_pegasos(read_image, descriptor_pegasos, dimension, hog, hog_window_widths[i1_iter], hog_window_heights[i2_iter],max_scales[i8_iter]);
					//compute and save HOG descriptors
				}
				else{	
					compute_hog(read_image, descriptors, hog, hog_window_widths[i1_iter], hog_window_heights[i2_iter], max_scales[i8_iter]); //compute and save HOG descriptors
				}
			}
			else{
				bg.operator ()(read_image, read_image, learningRate);	//computes foreground image
				if (pegasosTraining){			
					compute_haar_pegasos(read_image, descriptor_pegasos, dimension, eval, haar_slidingWin_widths[i1_iter], haar_slidingWin_heights[i2_iter], haar_sliding_steps[i3_iter], haar_finalROI_sizes[i4_iter]); //compute and save HAAR descriptors
					if (i == 0){
						saveDescriptorVariablesHAAR(haar_slidingWin_widths[i1_iter], haar_slidingWin_heights[i2_iter], haar_sliding_steps[i3_iter], haar_finalROI_sizes[i4_iter], dimension, num_comb, total_comb);
					}
				}
				else{		
					compute_haar(read_image, descriptors, eval, haar_slidingWin_widths[i1_iter], haar_slidingWin_heights[i2_iter], haar_sliding_steps[i3_iter], haar_finalROI_sizes[i4_iter]); //compute and save HAAR descriptors
				if (i == 0){
					saveDescriptorVariablesHAAR(haar_slidingWin_widths[i1_iter], haar_slidingWin_heights[i2_iter], haar_sliding_steps[i3_iter], haar_finalROI_sizes[i4_iter], descriptors.rows, num_comb, total_comb);
				}
				}
			}
			}

			//Load pre-computed feature vectors if there are, store feature vectors, and save Haar feature vectors 

			if (pegasosTraining){

				if (readVectors){
					readVector(filenames[i], descriptor_pegasos,dimension);

					//if (i == 0){

						//dimension = descriptor_pegasos.max_index()+1; //assuming no elements are removed from the vector
					//}

				}

				train_samples.push_back(descriptor_pegasos);	//introduce row vector
				//Haar features can be saved right now, but HOG features would be exported with natural values, i.e. without being normalized
				if (descriptorType != "HOG" && saveVectors && !readVectors){
				//Store computed vector in an external file
				sprintf(buffer, "frame_%06d_%d.yml", i, num_comb);
				
				std::ofstream vector_file(buffer);
				if (!vector_file.good()) {
					std::cerr << "error w/ " << buffer << std::endl;
					exit(EXIT_FAILURE);
				}
				descriptor_pegasos.print(vector_file);
				vector_file.close();
				}
			}
			else{

				//This code must be in this order, because regardless of the actual case (images or precomputed features), in this way the size is known
				if (readVectors){
					loadCsv(filenames[i],descriptors);
				}

				if (i == 0){
					train_features = Mat((int)filenames.size(), descriptors.rows, CV_32FC1);	//reserve size of matrix in first iteration
				}
	
					Mat tmp = descriptors.t();				//transpose to get row vector
					tmp.copyTo(train_features.row(i));
					//float resultado = train_features.at<float>(0, 0);
					//float resultado2 = train_features.at<float>(0, 20);

					//Haar features can be saved right now, but HOG features would be exported with natural values, i.e. without being normalized
					if (descriptorType != "HOG" && saveVectors && !readVectors){
						//Store computed vector in an external file
						sprintf(buffer, "frame_%06d_%d.csv", i, num_comb);
						saveMatToCsv(descriptors, buffer);
					}
				
				
			}
			cout << "Reading frame " << i << "\r";

		}
		cout << "\nSequence read!" << endl;

			// -- Normalization of data (only for HOG descriptors - HAAR descriptors are internally normalized)
			// -- (only if vectors are not previously loaded from external files)
			if (descriptorType == "HOG" && !readVectors){
				cout << "\nNormalizing data..." << endl;

				if (pegasosTraining){

					simple_sparse_vector means;
					simple_sparse_vector sigmas;

					for (uint i = 0; i < dimension; i++){

						float mean; float sigma;
						simple_sparse_vector feature_values;	//A column vector which indicates the values of each selected feature

						feature_values.getCol(train_samples, i);
						feature_values.meanStdDev(train_samples.size(), mean, sigma);

						//means.addElement(i, mean);
						//sigmas.addElement(i, sigma);
						means.addElement(mean);
						sigmas.addElement(sigma);

						for (int j = 0; j < train_samples.size(); j++){

							simple_sparse_vector_iterator it = train_samples[j].my_vec.begin() + i;
							(*it).second = ((*it).second - mean) / sigma;

						}

						cout << "Normalizing feature " << i << "\r";

					}

					cout << "\nSaving mean and sigma parameters in file...";

					string means_filename = "means";
					string sigmas_filename = "sigmas";

					if (total_comb>1){
						means_filename = to_string(num_comb) + "_" + means_filename + ".yml";
						sigmas_filename = to_string(num_comb) + "_" + sigmas_filename + ".yml";
					}
					else{
						means_filename = means_filename + ".yml";
						sigmas_filename = sigmas_filename + ".yml";
					}
					
					std::ofstream means_file(means_filename.c_str());
					if (!means_file.good()) {
						std::cerr << "error w/ " << means_filename << std::endl;
						exit(EXIT_FAILURE);
					}
					means.print(means_file);
					means_file.close();

					std::ofstream sigmas_file(sigmas_filename.c_str());
					if (!sigmas_file.good()) {
						std::cerr << "error w/ " << sigmas_filename << std::endl;
						exit(EXIT_FAILURE);
					}
					sigmas.print(sigmas_file);
					sigmas_file.close();

				}
				else{
					Mat means;
					Mat sigmas;

					for (int i = 0; i < train_features.cols; i++){
						Mat mean; Mat sigma;
						meanStdDev(train_features.col(i), mean, sigma);
						means.push_back(mean);
						if (countNonZero(sigma) < 1){
							sigma = 1;		//to prevent division by zero
						}
						sigmas.push_back(sigma);
						train_features.col(i) = (train_features.col(i) - mean) / sigma;
						cout << "Normalizing feature " << i << "\r";
					}

					Mat meansigma;
					hconcat(means, sigmas, meansigma);
					cout << "\nSaving mean and sigma parameters in file...";
					string meansigma_filename = "meansigma";
					if (total_comb > 1){
						meansigma_filename = to_string(num_comb) + "_" + meansigma_filename + ".csv";
					}
					else{
						meansigma_filename = meansigma_filename + ".csv";
					}

					saveMatToCsv(meansigma, meansigma_filename);
				}
				cout << "Done!" << endl;


				//Store HOG feature vectors once they are normalized
				if (saveVectors){
			
					for (int i = 0; i < filenames.size();i++){
					
						if (pegasosTraining){
						
							//Store computed vector in an external file
							sprintf(buffer, "frame_%06d_%d.yml", i, num_comb);

							std::ofstream vector_file(buffer);
							if (!vector_file.good()) {
								std::cerr << "error w/ " << buffer << std::endl;
								exit(EXIT_FAILURE);
							}
							train_samples[i].print(vector_file);
							vector_file.close();
						
						}
						else{
						
							//Store computed vector in an external file
							sprintf(buffer, "frame_%06d_%d.csv", i, num_comb);
							Mat tmp = (train_features.row(i)).t();

							saveMatToCsv(tmp, buffer); //To verify this: try the same but with meansigma: saveMatToCsv(meansigma[i],"something_numFeature.csv");
						}

						cout << "Storing feature vector of frame " << i << "\r";

					}
					cout << "\nStoring feature vectors done!" << endl;
				}

			}

			//Loop for each configuration of the grid classifiers (assuming they share the same video sequences)
			for (int i5 = 0; i5 < i5_max; i5++){

				//Loop for each configuration of SVM
				for (int i9 = 0; i9 < i9_max; i9++){
				
					//Get the instant of a new combination if it starts here
					if (duration2 == 0.0 && total_comb>1){
						duration2 = static_cast<double>(cv::getTickCount());
					}

				//The training part is not executed if feature vectors are computed exclusively

				if (!onlyComputeFeatures){


				//Set the same iterator of this loop or change it to the iterator of pre-trained models
				int i5_iter;
				int i9_iter;

				if (pegasosTraining && modelsDirectory[0] != ""){
					i5_iter = i7;
					i9_iter = i7;
				}
				else{
					i5_iter = i5;
					i9_iter = i9;
				}

				// First, identification of existing classes

				vector <Mat> classes_activations;	//Group of matrixes where activations are stored
				int numClassifiers;	//Number of positions in the current grid, where in each one there are one or more classifiers

				for (int i = 0; i < activationFolders[i5_iter].size(); i++){
					vector<string> activation_filenames;
					readDirectory(activationFolders[i5_iter][i], activation_filenames);

					for (int j = 0; j < activation_filenames.size(); j++){

						// Obtain the number of classifiers of the grid
						if (i == 0 && j == 0){
							Mat temp_activations;
							loadCsv(activation_filenames[j], temp_activations);
							numClassifiers = temp_activations.cols;
						}

						//Export the list of class names once in all the training session
						if (i7 == 0){

							// Obtain name of class of each activation file
							size_t sepPos = activation_filenames[j].rfind("_");
							size_t extPos = activation_filenames[j].rfind(".csv");
							string class_name = activation_filenames[j].substr(sepPos + 1, extPos - (sepPos + 1));

							// Always push the class name at the beginning
							if (class_names.size() == 0){
								class_names.push_back(class_name);
							}
							else{
								// Only introduce class if it is new 
								for (int k = 0; k < class_names.size(); k++){
									if (class_name.compare(class_names[k]) == 0){
										break;
									}
									else if (k == class_names.size() - 1){
										class_names.push_back(class_name);
									}
								}
							}

						}

					}
				}

				//Export the list of class names once in all the training session
				if (i7 == 0){
					// Put class names in alphabetical order and set number of different activation matrixes to the total number of classes
					sort(class_names.begin(), class_names.end());
					classes_activations.resize(class_names.size());
					// Save class names into a yml file (assumed that classes do not change between the different configurations of activations)
					saveClassNames(class_names);
				}

				// -- Now, load activation files in their correct gaps (activations of classes which do not appear in a certain folder are considered as zeros)
				// Accumulation of number of frames 
				int sum = 0;
				for (int i = 0; i < activationFolders[i5_iter].size(); i++){
					sum = sum + input_sizes[i];
					vector<string> activation_filenames;
					readDirectory(activationFolders[i5_iter][i], activation_filenames);

					for (int j = 0; j < activation_filenames.size(); j++){

						Mat temp_activations;
						loadCsv(activation_filenames[j], temp_activations);

						size_t sepPos = activation_filenames[j].rfind("_");
						size_t extPos = activation_filenames[j].rfind(".csv");
						string class_name = activation_filenames[j].substr(sepPos + 1, extPos - (sepPos + 1));

						for (int k = 0; k < class_names.size(); k++){
							if (class_name.compare(class_names[k]) == 0){
								classes_activations[k].push_back(temp_activations);

							}
						}

					}

					// Before changing the folder, fill with zeros the activations whose classes have not appeared in the current sequence
					for (int j = 0; j < classes_activations.size(); j++){

						if (classes_activations[j].rows<sum){
							Mat filled_activations = Mat(input_sizes[i], numClassifiers, CV_32FC1, Scalar(0));
							classes_activations[j].push_back(filled_activations);
						}
					}
				}

				//for (int i = 0; i < activationFolders.size(); i++){
				//Mat temp_activations;
				//	loadCsv(activationFolders[i], temp_activations);
				//activations.push_back(temp_activations);
				//}			

			// --- Get info from activations matrix (previously computed in GENERATION mode of main application)
			int numTotalSamples = classes_activations[0].rows;

			//int numTotalSamples = activations.rows;
			//int numClassifiers = activations.cols;

			srand(unsigned(std::time(0))); //introduce seed

			//Initialization of the matrixes of statistics and the matrix of best thresholds of each classifier in terms of fscore
			classes_precisions = Mat((int)classes_activations.size(), numClassifiers, CV_64FC1, Scalar(-1));
			classes_recalls = Mat((int)classes_activations.size(), numClassifiers, CV_64FC1, Scalar(-1));
			classes_fscores = Mat((int)classes_activations.size(), numClassifiers, CV_64FC1, Scalar(-1));
			thresholds = Mat((int)classes_activations.size(), numClassifiers, CV_64FC1, Scalar(-1));

			// -- Assign training labels and train
			// --- For each of the SVM classifiers in activations matrix in each class, assign positive and negative samples and train
			for (int k = 0; k < classes_activations.size(); k++){

				vector <int> activatedClassifiers;
				int type;

				// Threading-related variables
				vector <pthread_t> variable_threads;
				vector <struct svm_data> svm_data_arrays;

				pthread_attr_t attr;
				int rc;
				void *status;

				//Specific Pegasos variable
				string import_model_name;

				//Analyze which classifiers have positive samples and which not (especially important to control threading)
				//It is necessary to take into account if a pre-trained model already exists or
				//not, because it would be updated (even if there are only new negative samples) or instead 
				//a new model is created (only when using Pegasos Code)
				for (int i = firstClassifier[k]; i < numClassifiers; i++){

					//if (std::find(excluded_classifiers[k].begin(), excluded_classifiers[k].end(), i) == excluded_classifiers[k].end()){
					
					if (pegasosTraining){
							//Note that modelsDirectory must include slash
							import_model_name = modelsDirectory[i7] + "svmNum" + to_string(i) + "_" + class_names[k] + ".yml";
			
					}

					int num_positives = countNonZero(classes_activations[k].col(i));
					bool existModel = std::ifstream(import_model_name).good();

					if ((!pegasosTraining && num_positives>0) || (pegasosTraining && ((existModel && !balanceDataset) || (num_positives>0 && num_positives<numTotalSamples)) )){

						activatedClassifiers.push_back(i);

						if (useThreads){
						variable_threads.resize(variable_threads.size()+1);
						svm_data_arrays.resize(svm_data_arrays.size()+1);
						}
					
					}
					else{
						type = 102;	//If only negative samples, use ONE_CLASS Classifier -> No train!
					
					}

					//}
				}

				if (useThreads){

				pthread_mutex_init(&mutex_training, NULL);

				/* Initialize and set thread detached attribute */
				pthread_attr_init(&attr);
				pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
				}

				// Assign training labels for classifiers which have positive samples or they are at least updated with more
				//negative samples, and train
				for (int i = 0; i < activatedClassifiers.size();i++){
				
					double precision;
					double recall;
					double f_score;
					double ht;

					vector<vector<int>> samples_pos(2); //(-) (+) it indicates which frames are negative samples and which are positive respectively
					vector<int>batch_sizes(2); //(-) (+)	number of negative samples and number of positive samples in the batch respectively

					int num_positive = countNonZero(classes_activations[k].col(activatedClassifiers[i]));

					//Assign number of samples in the real batch, depending on the situation
					if (pegasosTraining){

						int num_negative = numTotalSamples - num_positive;

						int min_samples = num_positive;	//Determine the number of samples in the minority
						int max_samples = num_negative;

						//Very rare to happen
						if (num_negative < num_positive){
							min_samples = num_negative;
							max_samples = num_positive;
						}

						if (!balanceDataset){
						
							batch_sizes[0] = num_negative;
							batch_sizes[1] = num_positive;
						}

						else{
						//In every moment this can be computed (even just after the input parameters)
						int max_minoritygroup_size = (int) round(division*max_batch_size);

						if (min_samples <= max_minoritygroup_size){
						
							int maxgroup_samples = (int) round((1-division)*min_samples/division);

							if (min_samples == num_positive){
							
								batch_sizes[1] = min_samples;
								if (max_samples > maxgroup_samples){
								batch_sizes[0] = maxgroup_samples;
								}
								else{
									batch_sizes[0] = max_samples;
								}
							}
							else{
								
								batch_sizes[0] = min_samples;
								if (max_samples > maxgroup_samples){
								batch_sizes[1] = maxgroup_samples;
								}
								else{
									batch_sizes[1] = max_samples;
								}
							}
						
						}
						else{
						
							if (min_samples == num_positive){

								batch_sizes[1] = max_minoritygroup_size;
								if (max_samples>max_batch_size - max_minoritygroup_size){

									batch_sizes[0] = max_batch_size - max_minoritygroup_size;
								
								}
								else{
								
									batch_sizes[0] = max_samples;
								
								}

							}
							else{
								batch_sizes[0] = max_minoritygroup_size;

								if (max_samples > max_batch_size - max_minoritygroup_size){
								batch_sizes[1] = max_batch_size - max_minoritygroup_size;
								}
								else{
									batch_sizes[1] = max_samples;
								}

							}
						}

						}

					}
					
					//assign corresponding labels and for online training, which samples are positive and which are negative
					vector<int> class_labels(numTotalSamples, 0); //Binary vector for a determined class
					for (int j = 0; j < numTotalSamples; j++){

						if (classes_activations[k].at<float>(j, activatedClassifiers[i]) == 0){

							if (pegasosTraining){
								class_labels[j] = -1;	//sample is negative for current classifier

								samples_pos[0].push_back(j);
							}

							else{
								class_labels[j] = 0;	//sample is negative for current classifier
							}

						}
						else{
							class_labels[j] = +1;	//sample is positive for current classifier

							if (pegasosTraining){
								samples_pos[1].push_back(j);
							}
						}
					}

					// --- Train classifier
					string class_output_model_name = "svmNum" + to_string(activatedClassifiers[i]) + "_" + class_names[k] + ".yml";
					
					type=100;	//if both positive and negative samples, use C_SVC type for training

					if (total_comb > 1){

						class_output_model_name = to_string(num_comb) + "_" + class_output_model_name;

					}

					if (pegasosTraining){
					import_model_name = modelsDirectory[i7] + class_output_model_name;

					if (!std::ifstream(import_model_name).good()){
						import_model_name = "";
					}

					//In online training, the same original name of the model is used
					if (modelsDirectory[0] != ""){
					
						class_output_model_name = svm_prefix + class_output_model_name;
					
					}

					}


					printf("SVM # %i %s \n",activatedClassifiers[i],class_names[k]);

					if (useThreads){

						svm_data_arrays[i].output_labels = class_labels;
						svm_data_arrays[i].numSVM = activatedClassifiers[i];
						svm_data_arrays[i].svm_name = class_output_model_name;
						svm_data_arrays[i].class_label = k;
						svm_data_arrays[i].regularization = regularization_terms[i9_iter];

						if (pegasosTraining){
						
							svm_data_arrays[i].num_features = dimension;
							svm_data_arrays[i].train_data = train_samples.data();
							svm_data_arrays[i].svm_name_input = import_model_name;
							svm_data_arrays[i].max_iter = max_iters[i9_iter];
							
							svm_data_arrays[i].batch_sizes = batch_sizes;
							svm_data_arrays[i].samples_positions = samples_pos;
							
							rc = pthread_create(&variable_threads[i], &attr, train_svm_thread_pegasos, (void *)
								&svm_data_arrays[i]);
						
						}
						else{

							svm_data_arrays[i].svm_type = type;
							svm_data_arrays[i].input_vectors = train_features;

							rc = pthread_create(&variable_threads[i], &attr, train_svm_thread, (void *)
								&svm_data_arrays[i]);
						
						}

						if (rc) {
							printf("ERROR; return code from pthread_create() is %d\n", rc);
							exit(-1);
						}
					}
					else{

						if (pegasosTraining){
							Mat statistics(15,7,CV_64FC1, Scalar(-1));

							LearnReturnBestAdapted(train_samples, class_labels, dimension, import_model_name, class_output_model_name,precision,recall,f_score,
							regularization_terms[i9_iter],max_iters[i9_iter],ht,batch_sizes,samples_pos,statistics);

							string statistics_name = class_output_model_name.substr(0, class_output_model_name.size() - 4) + ".csv";
							saveMatToCsv(statistics, statistics_name);

						}
						else{
							train_svm(type, train_features, class_labels, class_output_model_name, precision, recall, f_score, ht, regularization_terms[i9_iter]);
							
						}

						classes_precisions.at<double>(k, activatedClassifiers[i]) = precision;
						classes_recalls.at<double>(k, activatedClassifiers[i]) = recall;
						classes_fscores.at<double>(k, activatedClassifiers[i]) = f_score;
						thresholds.at<double>(k, activatedClassifiers[i]) = ht;

					}
				
				}

			if (useThreads){
				/* Free attribute and wait for the other threads */
				pthread_attr_destroy(&attr);
				for (int t = 0; t<variable_threads.size(); t++) {
					rc = pthread_join(variable_threads[t], &status);
					if (rc) {
						printf("ERROR; return code from pthread_join() is %d\n", rc);
						exit(-1);
					}
					//printf("Main: completed join with thread %ld \n",t);
				}

				pthread_mutex_destroy(&mutex_training);
			}

			

			}
			
			//Export statistics about the current combination of parameters
			string precisions_filename = "Precision_" + to_string(num_comb) + ".csv";
			string recalls_filename = "Recall_" + to_string(num_comb) + ".csv";
			string fscores_filename = "F1Score_" + to_string(num_comb) + ".csv";
			string thresholds_filename = "Thresholds_" + to_string(num_comb) + ".csv";

			saveMatToCsv(classes_precisions,precisions_filename);
			saveMatToCsv(classes_recalls, recalls_filename);
			saveMatToCsv(classes_fscores,fscores_filename);
			saveMatToCsv(thresholds, thresholds_filename);
			

		}
		
		//Display elapsed time of the current combination
		if (total_comb > 1){
		// -- Print execution time
		duration2 = static_cast<double>(cv::getTickCount()) - duration2;
		duration2 =duration2 / (cv::getTickFrequency() / 1000);	//duration of processing in ms
		cout << "Combination " << num_comb << " processing elapsed time: " << duration2 << " ms" << endl;
		}
		//original end
		num_comb++;
		duration2 = 0.0;
		}
	}
	}
	}
	}
	}
	}
	}
	}

	// -- Print execution time
	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration = duration / (cv::getTickFrequency() / 1000);	//duration of processing in ms
	cout << "Frame processing elapsed time: " << duration << " ms" << endl;

	cin.ignore();
	if (useThreads){
		pthread_exit(NULL);
	}
	return 0;

}
