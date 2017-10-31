// Distributed under GNU General Public License (see license.txt for details).
//
//  Copyright (c) 2007 Shai Shalev-Shwartz.
//  All Rights Reserved.
//=============================================================================
// File Name: pegasos_optimize.h
// header for the main optimization function of pegasos
// Adapted by: Andrés Bell Navas (22.03.17)
//=============================================================================

#ifndef _SHAI_PEGASOS_OPTIMIZE_H
#define _SHAI_PEGASOS_OPTIMIZE_H

//*****************************************************************************
// Included Files
//*****************************************************************************
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <ctime>
#include <cmath>
#include "simple_sparse_vec_hash.h"
#include "WeightVector.h"
#include <opencv2/core/core.hpp>

#include <algorithm>

using namespace cv;

// function from Joachims that measures CPU time
long get_runtime(void);

// main optimization function
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
	   int eta_rule_type, double eta_constant,
	   int projection_rule, double projection_constant);


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
		      int projection_rule, double projection_constant);


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
		      int projection_rule, double projection_constant);

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
	Mat &statistics);//,
	//long& train_time, long& calc_obj_time,
	//double& obj_value, double& norm_value,
	//double& loss_value, double& zero_one_error,
	//double& test_loss, double& test_error,
	// additional parameters
	//int eta_rule_type, double eta_constant,
	//int projection_rule, double projection_constant);



// function for reading the data
void ReadData(// input
	      std::string& data_filename,
	      // output
	      std::vector<simple_sparse_vector> & Dataset,
	      std::vector<int> & Labels,
	      uint& dimension,
	      long& readingTime);


// function for running experiments
void run_experiments(std::string& experiments_file,
		     std::vector<simple_sparse_vector>& Dataset,
		     std::vector<int>& Labels,
		     uint dimension,
		     std::vector<simple_sparse_vector>& testDataset,
		     std::vector<int>& testLabels);


#endif
