// Distributed under GNU General Public License (see license.txt for details).
//
//  Copyright (c) 2007 Shai Shalev-Shwartz.
//  All Rights Reserved.
//=============================================================================
// File Name: pegasos_optimize.cc
// implements the main optimization function of pegasos
// Adapted by: Andrés Bell Navas (22.03.17)
//=============================================================================

#include "pegasos_optimize.h"

// help function for getting runtime
long get_runtime(void)
{
  clock_t start;
  start = clock();
  return((long)((double)start/(double)CLOCKS_PER_SEC));
}


// ------------------------------------------------------------//
// ---------------- OPTIMIZING --------------------------------//
// ------------------------------------------------------------//
void Learn(// Input variables
	   std::vector<simple_sparse_vector>& Dataset,
	   std::vector<int>& Labels,
	   uint dimension,
	   std::vector<simple_sparse_vector>& testDataset,
	   std::vector<int>& testLabels,
	   double lambda,int max_iter,int exam_per_iter,int num_iter_to_avg,
	   std::string& model_filename,
	   // Output variables
	   long& train_time,long& calc_obj_time,double& obj_value,
	   double& norm_value,double& loss_value,double& zero_one_error,
	   double& test_loss,double& test_error,
	   // additional parameters
	   int eta_rule_type , double eta_constant ,
	   int projection_rule, double projection_constant) {

  uint num_examples = Labels.size();

  long startTime = get_runtime();
  long endTime;
  

  // Initialization of classification vector
  WeightVector W(dimension);
  WeightVector AvgW(dimension);
  double avgScale = (num_iter_to_avg > max_iter) ? max_iter : num_iter_to_avg; 

  // ---------------- Main Loop -------------------
  for (int i = 0; i < max_iter; ++i) {

    // learning rate
    double eta;
    if (eta_rule_type == 0) { // Pegasos eta rule
      eta = 1 / (lambda * (i+2)); 
    } else if (eta_rule_type == 1) { // Norma rule
      eta = eta_constant / sqrt(i+2);
      // solve numerical problems
      W.make_my_a_one();
    } else {
      eta = eta_constant;
    }

    // gradient indices and losses
    std::vector<uint> grad_index;
    std::vector<double> grad_weights;

    // calc sub-gradients
    for (int j=0; j < exam_per_iter; ++j) {

      // choose random example
      uint r = ((int)rand()) % num_examples;

      // calculate prediction
      double prediction = W*Dataset[r];

      // calculate loss
      double cur_loss = 1 - Labels[r]*prediction;
      if (cur_loss < 0.0) cur_loss = 0.0;

      // and add to the gradient
      if (cur_loss > 0.0) {
	grad_index.push_back(r);
	grad_weights.push_back(eta*Labels[r]/exam_per_iter);
      }
    }
 
    // scale w 
    W.scale(1.0 - eta*lambda);

    // and add sub-gradients
    for (uint j=0; j<grad_index.size(); ++j) {
      W.add(Dataset[grad_index[j]],grad_weights[j]);
    }

    // Project if needed
    if (projection_rule == 0) { // Pegasos projection rule
      double norm2 = W.snorm();
      if (norm2 > 1.0/lambda) {
	W.scale(sqrt(1.0/(lambda*norm2)));
      }
    } else if (projection_rule == 1) { // other projection
      double norm2 = W.snorm();
      if (norm2 > (projection_constant*projection_constant)) {
	W.scale(projection_constant/sqrt(norm2));
      }
    } // else -- no projection


    // and update AvgW
    if (max_iter <= num_iter_to_avg + i)
      AvgW.add(W, 1.0/avgScale);
  }


  // update timeline
  endTime = get_runtime();
  train_time = endTime - startTime;
  startTime = get_runtime();

  // Calculate objective value
  norm_value = AvgW.snorm();
  obj_value = norm_value * lambda / 2.0;
  loss_value = 0.0;
  zero_one_error = 0.0;
  for (uint i=0; i < Dataset.size(); ++i) {
    double cur_loss = 1 - Labels[i]*(AvgW * Dataset[i]); 
    if (cur_loss < 0.0) cur_loss = 0.0;
    loss_value += cur_loss/num_examples;
    obj_value += cur_loss/num_examples;
    if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
  }

  endTime = get_runtime();
  calc_obj_time = endTime - startTime;

  // Calculate test_loss and test_error
  test_loss = 0.0;
  test_error = 0.0;
  for (uint i=0; i < testDataset.size(); ++i) {
    double cur_loss = 1 - testLabels[i]*(AvgW * testDataset[i]); 
    if (cur_loss < 0.0) cur_loss = 0.0;
    test_loss += cur_loss;
    if (cur_loss >= 1.0) test_error += 1.0;
  }
  if (testDataset.size() != 0) {
    test_loss /= testDataset.size();
    test_error /= testDataset.size();
  }
  


  // finally, print the model to the model_file
  if (model_filename != "noModelFile") {
    std::ofstream model_file(model_filename.c_str());
    if (!model_file.good()) {
      std::cerr << "error w/ " << model_filename << std::endl;
      exit(EXIT_FAILURE);
    }
    AvgW.print(model_file);
    model_file.close();
  }

}




void LearnAndValidate(// Input variables
		      std::vector<simple_sparse_vector>& Dataset,
		      std::vector<int>& Labels,
		      uint dimension,
		      std::vector<simple_sparse_vector>& testDataset,
		      std::vector<int>& testLabels,
		      double lambda,int max_iter,
		      int exam_per_iter,int num_example_to_validate,
		      std::string& model_filename,
		      // Output variables
		      long& train_time,long& calc_obj_time,
		      double& obj_value,double& norm_value,
		      double& loss_value,double& zero_one_error,
		      double& test_loss,double& test_error,
		      // additional parameters
		      int eta_rule_type , double eta_constant ,
		      int projection_rule, double projection_constant) {

  uint num_examples = Labels.size();

  long startTime = get_runtime();
  long endTime;
  

  // Initialization of classification vector
  WeightVector W(dimension);
  WeightVector BestW(dimension);
  double best_obj = 1.0; // the zero solution

  // create validation indices
  std::vector<uint> validate_indices(num_example_to_validate);
  for (uint i=0; i < validate_indices.size(); ++i)
    validate_indices[i] = ((int)rand()) % num_examples;

  // Choose s random indices 
  int s = 5; // corresponds to confidence of 0.9933
  int block_size = max_iter/s;
  std::vector<int> candidates(s);
  for (int i=0; i<s; ++i) {
    candidates[i] = block_size*i + ((int)rand()) % block_size;
  }
  candidates[s-1] = max_iter-1; // make sure we need all iterations
  int cur_block = 0;

  
  

  // ---------------- Main Loop -------------------
  for (int i = 0; i < max_iter; ++i) {

    // learning rate
    double eta;
    if (eta_rule_type == 0) { // Pegasos eta rule
      eta = 1 / (lambda * (i+2)); 
    } else if (eta_rule_type == 1) { // Norma rule
      eta = eta_constant / sqrt(i+2);
      // solve numerical problems
      if (projection_rule != 2)
	W.make_my_a_one();
    } else {
      eta = eta_constant;
    }

    // gradient indices and losses
    std::vector<uint> grad_index;
    std::vector<double> grad_weights;

    // calc sub-gradients
    for (int j=0; j < exam_per_iter; ++j) {

      // choose random example
      uint r = ((int)rand()) % num_examples;

      // calculate prediction
      double prediction = W*Dataset[r];

      // calculate loss
      double cur_loss = 1 - Labels[r]*prediction;
      if (cur_loss < 0.0) cur_loss = 0.0;

      // and add to the gradient
      if (cur_loss > 0.0) {
	grad_index.push_back(r);
	grad_weights.push_back(eta*Labels[r]/exam_per_iter);
      }
    }
 
    // scale w 
    W.scale(1.0 - eta*lambda);

    // and add sub-gradients
    for (uint j=0; j<grad_index.size(); ++j) {
      W.add(Dataset[grad_index[j]],grad_weights[j]);
    }

    // Project if needed
    if (projection_rule == 0) { // Pegasos projection rule
      double norm2 = W.snorm();
      if (norm2 > 1.0/lambda) {
	W.scale(sqrt(1.0/(lambda*norm2)));
      }
    } else if (projection_rule == 1) { // other projection
      double norm2 = W.snorm();
      if (norm2 > (projection_constant*projection_constant)) {
	W.scale(projection_constant/sqrt(norm2));
      }
    } // else -- no projection


    // and validate
    if (i == candidates[cur_block]) {
      double obj = 0.0;
      for (uint j=0; j < validate_indices.size(); ++j) {
	uint ind = validate_indices[j];
	double cur_loss = 1 - Labels[ind]*(W * Dataset[ind]); 
	if (cur_loss < 0.0) cur_loss = 0.0;
	obj += cur_loss;
      }
      obj /= validate_indices.size();
      obj += lambda/2.0*W.snorm();
      if (obj <= best_obj) {
	//	std::cerr << "obj of " << i << " (candidates[" 
	//		  << cur_block << "]) = " << obj << std::endl;
	BestW = W;
	best_obj = obj;
      }
      cur_block++;
    }
  }


  // update timeline
  endTime = get_runtime();
  train_time = endTime - startTime;
  startTime = get_runtime();

  // Calculate objective value
  norm_value = BestW.snorm();
  obj_value = norm_value * lambda / 2.0;
  loss_value = 0.0;
  zero_one_error = 0.0;
  for (uint i=0; i < Dataset.size(); ++i) {
    double cur_loss = 1 - Labels[i]*(BestW * Dataset[i]); 
    if (cur_loss < 0.0) cur_loss = 0.0;
    loss_value += cur_loss/num_examples;
    obj_value += cur_loss/num_examples;
    if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
  }

  endTime = get_runtime();
  calc_obj_time = endTime - startTime;

  // Calculate test_loss and test_error
  test_loss = 0.0;
  test_error = 0.0;
  for (uint i=0; i < testDataset.size(); ++i) {
    double cur_loss = 1 - testLabels[i]*(BestW * testDataset[i]); 
    if (cur_loss < 0.0) cur_loss = 0.0;
    test_loss += cur_loss;
    if (cur_loss >= 1.0) test_error += 1.0;
  }
  if (testDataset.size() != 0) {
    test_loss /= testDataset.size();
    test_error /= testDataset.size();
  }
  


  // finally, print the model to the model_file
  if (model_filename != "noModelFile") {
    std::ofstream model_file(model_filename.c_str());
    if (!model_file.good()) {
      std::cerr << "error w/ " << model_filename << std::endl;
      exit(EXIT_FAILURE);
    }
    BestW.print(model_file);
    model_file.close();
  }

}





void LearnReturnLast(// Input variables
		      std::vector<simple_sparse_vector>& Dataset,
		      std::vector<int>& Labels,
		      uint dimension,
		      std::vector<simple_sparse_vector>& testDataset,
		      std::vector<int>& testLabels,
		      double lambda,int max_iter,
		      int exam_per_iter,
		      std::string& model_filename,
		      // Output variables
		      long& train_time,long& calc_obj_time,
		      double& obj_value,double& norm_value,
		      double& loss_value,double& zero_one_error,
		      double& test_loss,double& test_error,
		      // additional parameters
		      int eta_rule_type , double eta_constant ,
		      int projection_rule, double projection_constant) {

  uint num_examples = Labels.size();

  long startTime = get_runtime();
  long endTime;
  

  // Initialization of classification vector
  WeightVector W(dimension);

  // ---------------- Main Loop -------------------
  for (int i = 0; i < max_iter; ++i) {

    // learning rate
    double eta;
    if (eta_rule_type == 0) { // Pegasos eta rule
      eta = 1 / (lambda * (i+2)); 
    } else if (eta_rule_type == 1) { // Norma rule
      eta = eta_constant / sqrt(i+2);
      // solve numerical problems
      //if (projection_rule != 2)
	W.make_my_a_one();
    } else {
      eta = eta_constant;
    }

    // gradient indices and losses
    std::vector<uint> grad_index;
    std::vector<double> grad_weights;

    // calc sub-gradients
    for (int j=0; j < exam_per_iter; ++j) {

      // choose random example
      uint r = ((int)rand()) % num_examples;

      // calculate prediction
      double prediction = W*Dataset[r];

      // calculate loss
      double cur_loss = 1 - Labels[r]*prediction;
      if (cur_loss < 0.0) cur_loss = 0.0;

      // and add to the gradient
      if (cur_loss > 0.0) {
	grad_index.push_back(r);
	grad_weights.push_back(eta*Labels[r]/exam_per_iter);
      }
    }
 
    // scale w 
    W.scale(1.0 - eta*lambda);

    // and add sub-gradients
    for (uint j=0; j<grad_index.size(); ++j) {
      W.add(Dataset[grad_index[j]],grad_weights[j]);
    }

    // Project if needed
    if (projection_rule == 0) { // Pegasos projection rule
      double norm2 = W.snorm();
      if (norm2 > 1.0/lambda) {
	W.scale(sqrt(1.0/(lambda*norm2)));
      }
    } else if (projection_rule == 1) { // other projection
      double norm2 = W.snorm();
      if (norm2 > (projection_constant*projection_constant)) {
	W.scale(projection_constant/sqrt(norm2));
      }
    } // else -- no projection

  }


  // update timeline
  endTime = get_runtime();
  train_time = endTime - startTime;
  startTime = get_runtime();

  // Calculate objective value
  norm_value = W.snorm();
  obj_value = norm_value * lambda / 2.0;
  loss_value = 0.0;
  zero_one_error = 0.0;
  for (uint i=0; i < Dataset.size(); ++i) {
    double cur_loss = 1 - Labels[i]*(W * Dataset[i]); 
    if (cur_loss < 0.0) cur_loss = 0.0;
    loss_value += cur_loss/num_examples;
    obj_value += cur_loss/num_examples;
    if (cur_loss >= 1.0) zero_one_error += 1.0/num_examples;
  }

  endTime = get_runtime();
  calc_obj_time = endTime - startTime;

  // Calculate test_loss and test_error
  test_loss = 0.0;
  test_error = 0.0;
  for (uint i=0; i < testDataset.size(); ++i) {
    double cur_loss = 1 - testLabels[i]*(W * testDataset[i]); 
    if (cur_loss < 0.0) cur_loss = 0.0;
    test_loss += cur_loss;
    if (cur_loss >= 1.0) test_error += 1.0;
  }
  if (testDataset.size() != 0) {
    test_loss /= testDataset.size();
    test_error /= testDataset.size();
  }
  


  // finally, print the model to the model_file
  if (model_filename != "noModelFile") {
    std::ofstream model_file(model_filename.c_str());
    if (!model_file.good()) {
      std::cerr << "error w/ " << model_filename << std::endl;
      exit(EXIT_FAILURE);
    }
    W.print(model_file);
    model_file.close();
  }

}


//New version of the function which must introduce online training and must be simplified according to SVM_training_main
void LearnReturnBestAdapted(// Input variables
	std::vector<simple_sparse_vector>& Dataset,
	std::vector<int>& Labels,
	uint dimension,
	//std::vector<simple_sparse_vector>& testDataset,
	//std::vector<int>& testLabels,
	//double lambda, int max_iter,
	//int exam_per_iter,
	std::string& input_filename,
	// Output variables
	std::string& output_filename,
	double &precision,
	double &recall,
	double &f_score,
	double lambda,
	int max_iter, double &threshold,
	vector<int> &batch_size,
	vector<vector<int>> &samples_pos,
	Mat &statistics
	//long& train_time, long& calc_obj_time,
	//double& obj_value, double& norm_value,
	//double& loss_value, double& zero_one_error,
	//double& test_loss, double& test_error,
	) {

	std::clog << "Start training...\n";

	bool evaluate = true;

	double max_th_lambda = 3;	//Boundary in reduction of lambdaEff
	double max_lambda = 4;	//Minimum value of lambdaEff when it reaches the boundary

	double improve_th = 0.00005;	//If improvement in the cost function is less than this value, the training ends
								//assuming that now it is optimized

	double class_th = 1;		//Threshold to use in the loss function, a value >1 makes more possibilities to have a larger margin
	double decision_th = 7;

	int step_0 = (int)round(batch_size[0]*1/4);
	int step_1 = (int)round(batch_size[1]*1/4);

	//Start program

	int eta_rule_type = 0; double eta_constant = 0.0;
	int projection_rule = 0; double projection_constant = 0.0;

	//uint num_examples = Labels.size();

	// Initialization of classification vector
	WeightVector W(dimension); //Use this default constructor for first training; use other constructor for model importation
	WeightVector BestW(dimension);
	double best_obj; 
	double best_loss;
	int best_iter=-1;

	if (input_filename != "") {

		// read weight vector
		std::ifstream model_file_import(input_filename.c_str());
		if (!model_file_import.good()) {
			std::cerr << "error w/ " << input_filename << std::endl;
			exit(EXIT_FAILURE);
		}
		W=WeightVector(dimension, model_file_import);
		model_file_import.close();

	}

	double current_obj_value=0.0;
	double current_loss_value=0.0;

	//double TP;
	//double FP;
	//double FN;

	double lambdaEff = lambda; //Initialization of real lambda, which will be descending

	int it0_begin = 0;
	int it1_begin = 0;

	// ---------------- Main Loop -------------------
	for (int i = 0; i < max_iter; ++i) {

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
			//
			//	if (Labels[j] == +1){
			//		TP += 1.0;
			//	}
			//	else{
			//		FP += 1.0;
			//	}
			//}
			//else{
			//
			//	if (Labels[j] == +1){
			//		FN += 1.0;
			//	}
			//
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
		//std::cout << "Dimension " << BestW.dimension() << std::endl;
		//std::cout << "Norma " << BestW.snorm() << std::endl;
		//std::cout << "contenido " << BestW[0] << std::endl;
		//std::cout << "Iteracion del mejor " << best_iter << std::endl;

		//std::cout << "Lambda: " << lambdaEff << std::endl;

		if (i>0 && (((current_obj_value - new_obj_value) >=0) && ((current_obj_value - new_obj_value)<improve_th))){

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
			//	lambdaEff = lambdaEff + (lambda / 200); // 
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
			double cur_loss = class_th - Labels[r] * (prediction);
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

		if (i == max_iter - 1){

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

		//std::cout << "Time elapsed: " << train_time << std::endl;

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

		statistics.at<double>(1, 0) = best_iter;

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

					double fscore = (double)(2 * TPs[j]) / (double)((2 * TPs[j]) + FPs[j] + FNs[j]);
					double new_precision = (double)TPs[j] / (double)(TPs[j] + FPs[j]);
					double new_recall = (double)TPs[j] / (double)(TPs[j] + FNs[j]);

					if (j == 0){

						max_fscore = fscore;
						min_FP = FPs[j];
						precision = new_precision;
						recall = new_recall;
					}

					statistics.at<double>(j, 2) = new_precision;
					statistics.at<double>(j, 3) = new_recall;
					statistics.at<double>(j, 4) = fscore;
					statistics.at<double>(j, 5) = FPs[j];

					if (TPs[j] + FNs[j] == 0){

						if (FPs[j]<min_FP){

							min_FP = FPs[j];
							best_ht = ht[j];
							max_fscore = fscore;
							precision = new_precision;
							recall = new_recall;

						}

					}

					else{
						if (fscore > max_fscore){
							max_fscore = fscore;
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

		f_score = max_fscore;
		threshold = best_ht;

	}
	else{
		precision = -1.0;
		recall = -1.0;
		f_score = -1.0;
		threshold = 0.0;
	}
	
	


	//endTime = get_runtime();	//optional
	//calc_obj_time = endTime - startTime; //optional

	// Calculate test_loss and test_error (optional)
	//test_loss = 0.0;
	//test_error = 0.0;
	//for (uint i = 0; i < testDataset.size(); ++i) {
	//	double cur_loss = 1 - testLabels[i] * (W * testDataset[i]);
	//	if (cur_loss < 0.0) cur_loss = 0.0;
	//	test_loss += cur_loss;
	//	if (cur_loss >= 1.0) test_error += 1.0;
//	}
	//if (testDataset.size() != 0) {
		//test_loss /= testDataset.size();
		//test_error /= testDataset.size();
	//}

	std::clog << "...[done]" << std::endl;

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

}



// ------------------------------------------------------------//
// ---------------- READING DATA ------------------------------//
// ------------------------------------------------------------//
void ReadData(// input
	      std::string& data_filename,
	      // output
	      std::vector<simple_sparse_vector> & Dataset,
	      std::vector<int> & Labels,
	      uint& dimension,
	      long& readingTime) {
  
  dimension = 0;

  // Start a timer
  long startTime = get_runtime();

  // OPEN DATA FILE
  // =========================
  std::ifstream data_file(data_filename.c_str());
  if (!data_file.good()) {
    std::cerr << "error w/ " << data_filename << std::endl;
    exit(EXIT_FAILURE);
  }

  
  // Read SVM-Light data file
  // ========================
  int num_examples = 0;
  std::string buf;
  while (getline(data_file,buf)) {
    // ignore lines which begin with #
    if (buf[0] == '#') continue;
    // Erase what comes after #
    size_t pos = buf.find('#');
    if (pos < buf.size()) {
      buf.erase(pos);
    }
    // replace ':' with white space
    int n=0;
    for (size_t pos=0; pos < buf.size(); ++pos)
      if (buf[pos] == ':') {
	n++; buf[pos] = ' ';
      }
    // read from the string
    std::istringstream is(buf);
    int label = 0;
    is >> label;
    if (label != 1 && label != -1) {
      std::cerr << "Error reading SVM-light format. Abort." << std::endl;
      exit(EXIT_FAILURE);
    }
    Labels.push_back(label);
    simple_sparse_vector instance(is,n);
    Dataset.push_back(instance);
    num_examples++;
    uint cur_max_ind = instance.max_index() + 1;
    if (cur_max_ind > dimension) dimension = cur_max_ind;
  }

  data_file.close();


#ifdef nodef
  std::cerr << "num_examples = " << num_examples 
	    << " dimension = " << dimension
	    << " Dataset.size = " << Dataset.size() 
	    << " Labels.size = " << Labels.size() << std::endl;
#endif
    
  
  // update timeline
  readingTime = get_runtime() - startTime;
  
}



// -------------------------------------------------------------//
// ---------------------- Experiments mode ---------------------//
// -------------------------------------------------------------//

  class ExperimentStruct {
  public:
    ExperimentStruct() { }
    void Load(std::istringstream& is) {
      is >> lambda >> max_iter >> exam_per_iter >> num_iter_to_avg
	 >> eta_rule >> eta_constant >> projection_rule 
	 >> projection_constant;
    }
    void Print() {
      std::cout << lambda << "\t\t" << max_iter << "\t\t" << exam_per_iter << "\t\t" 
		<< num_iter_to_avg << "\t\t" << eta_rule << "\t\t" << eta_constant 
		<< "\t\t" << projection_rule << "\t\t" << projection_constant << "\t\t";
    }
    void PrintHead() {
      std::cerr << "lambda\t\tT\t\tk\t\tnumValid\t\te_r\t\te_c\t\tp_r\t\tp_c\t\t";
    }
    double lambda, eta_constant, projection_constant;
    uint max_iter,exam_per_iter,num_iter_to_avg;
    int eta_rule,projection_rule;
  };

  class ResultStruct {
  public:
    ResultStruct() : trainTime(0), calc_obj_time(0),
		     norm_value(0.0),loss_value(0.0),
		     zero_one_error(0.0),obj_value(0.0),
		     test_loss(0.0), test_error(0.0) { }
    void Print() {
      std::cout << trainTime << "\t\t" 
		<< calc_obj_time << "\t\t" 
		<< norm_value  << "\t\t" 
		<< loss_value << "\t\t"  
		<< zero_one_error  << "\t\t" 
		<< obj_value << "\t\t"
		<< test_loss << "\t\t"
		<< test_error << "\t\t";

    }
    void PrintHead() {
      std::cerr << "tTime\t\tcoTime\t\t||w||\t\tL\t\tL0-1\t\tobj_value\t\ttest_L\t\ttestL0-1\t\t";
    }

    long trainTime, calc_obj_time;
    double norm_value,loss_value,zero_one_error,obj_value,test_loss,test_error;
  };


void run_experiments(std::string& experiments_filename,
		     std::vector<simple_sparse_vector>& Dataset,
		     std::vector<int>& Labels,
		     uint dimension,
		     std::vector<simple_sparse_vector>& testDataset,
		     std::vector<int>& testLabels) {

  // open the experiments file
  std::ifstream exp_file(experiments_filename.c_str());
  if (!exp_file.good()) {
    std::cerr << "error w/ " << experiments_filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // read the entire experiment specification

  
  uint num_experiments = 0;
  std::vector<ExperimentStruct> v;
  std::string buf;
  while (getline(exp_file,buf)) {
    // read from the string
    std::istringstream is(buf);
    ExperimentStruct tmp; tmp.Load(is);
    v.push_back(tmp);
    num_experiments++;
  }
  exp_file.close();

  // run all the experiments
  std::vector<ResultStruct> res(num_experiments);
  std::string lala = "noModelFile";
  for (uint i=0; i<num_experiments; ++i) {
    LearnReturnLast(Dataset,Labels,dimension,testDataset,testLabels,
		     v[i].lambda,v[i].max_iter,v[i].exam_per_iter,
		     //v[i].num_iter_to_avg,
		     lala,
		     res[i].trainTime,res[i].calc_obj_time,res[i].obj_value,
		     res[i].norm_value,
		     res[i].loss_value,res[i].zero_one_error,
		     res[i].test_loss,res[i].test_error,
		     v[i].eta_rule,v[i].eta_constant,
		     v[i].projection_rule,v[i].projection_constant);
  }

  // print results
  v[0].PrintHead(); res[0].PrintHead(); 
  std::cout << std::endl;
  for (uint i=0; i<num_experiments; ++i) {
    v[i].Print();
    res[i].Print();
    std::cout << std::endl;
  }

  std::cerr << std::endl;

}
