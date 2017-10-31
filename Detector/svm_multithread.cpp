
#include "svm_multithread.h"
#include <pthread.h>
#include <opencv2/ml/ml.hpp>

using namespace cv;


/**Predict and determine the confidence in case the prediction is positive*/

//rellenar con el código ejecutado en paralelo
//void *prediction_thread(void *threadarg){

//	double hyperplane;
//	Mat descriptor;
//	Ptr<SVM> svm_threaded;
//	int num_th;
//	int num_class;

//	struct svm_ids *id_data;


//	id_data = (struct svm_ids *) threadarg;

//	hyperplane = id_data->ht;
//	svm_threaded = id_data->svm_pointer;
//	descriptor = id_data->descriptor_vector;
	//num_th = id_data->num_thread;
	//num_class = id_data->class_label;
	//Code to compute predictions

//	float predicted_value = (svm_threaded)->predict(descriptor, true);

//	if (predicted_value < hyperplane){			//if < hyperplane_threshold, positive prediction

	//	if (predicted_value < 0){
			//confidences.at<float>(i) = abs(predicted_value) + hyperplane_threshold;	
		//	float confidence = abs(predicted_value) + hyperplane; //save confidence
			//if (confidence>max_score){
				//max_score = confidence;
				//class_max = j;
				//if (showActivationsAndConfidence){
					//max_predicted = predicted_value;
				//}
			//}
			//class_confidences[j].at<float>(i) = abs(predicted_value) + hyperplane_thresholds[j]; //original version
	//	}
	//	else{
			//confidences.at<float>(i) = hyperplane_threshold - abs(predicted_value);	
		//	float confidence = hyperplane - abs(predicted_value); //save confidence
			//if (confidence>max_score){
				//max_score = confidence;
				//class_max = j;
				//if (showActivationsAndConfidence){
					//max_predicted = predicted_value;
				//}
			//}
			//class_confidences[j].at<float>(i) = hyperplane_thresholds[j] - abs(predicted_value); //original version
	//	}

	
	
	//}
	
	//Sección de congelar y actualizar todo lo que queramos actualizar
	//pthread_mutex_lock(&mutex_classification);
	
	

	//pthread_mutex_unlock(&mutex_classification);
	//Salir
	//pthread_exit((void*)num_th);
	//return 0;

//}