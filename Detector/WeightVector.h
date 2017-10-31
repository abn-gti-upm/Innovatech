// Distributed under GNU General Public License (see license.txt for details).
//
//  Copyright (c) 2007 Shai Shalev-Shwartz.
//  All Rights Reserved.
//==============================================================================
// File Name: WeightVector.h
// Written by: Shai Shalev-Shwartz (28.01.07)
// efficiently implements a weight vector for sparse data
// Adapted by: Andrés Bell Navas (22.03.17)
//==============================================================================

#ifndef _SHAI_WeightVector_H
#define _SHAI_WeightVector_H

//*****************************************************************************
// Included Files
//*****************************************************************************
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sstream>

#include "simple_sparse_vec_hash.h"


class WeightVector {
  
 private:
  
	// WeightVector(const WeightVector&);// disallowed
  
 public:

  WeightVector() :d(0), my_a(1.0), my_snorm(0.0), my_v(NULL){
  
  };

  //WeightVector(const WeightVector&);// disallowed

  // Construct a Weight vector with dimension d
  WeightVector(uint dd) : d(dd), my_a(1.0), my_snorm(0.0), my_v(NULL)  {
    my_v = new double[d];
    for (uint i=0; i<d; ++i) my_v[i] = 0.0;
  }

  // Construct a Weight vector with dimension d from a file
  //WeightVector(uint dd,std::ifstream &model_file) : 
 //   d(dd), my_a(1.0), my_snorm(0.0), my_v(NULL) {
 //   my_v = new double[d];
  //  for (uint i=0; i<d; ++i) my_v[i] = 0.0;
  //  unsigned int ind = 0;
  //  model_file >> ind;
  //  while (model_file.good()) {
    //  char c;
   //   model_file >> c;
    //  double val;
   //   model_file >> val;
    //  my_v[ind] = val;
    //  my_snorm += val*val;
    //  model_file >> ind;
    //}
 // }

  WeightVector(uint dd,std::ifstream &model_file) : 
     d(dd), my_a(1.0), my_snorm(0.0), my_v(new double[d]) {
     //my_v = new double[d];
  //  for (uint i=0; i<d; ++i) my_v[i] = 0.0;

    std::string buf;

	getline(model_file, buf);

	size_t separator_pos = buf.find(" ");
	size_t colon_pos = buf.find(":");

	uint ind = 0;

	while (colon_pos!=std::string::npos){

		double val = stod(buf.substr(colon_pos+1,separator_pos-colon_pos-1));
		my_v[ind] = val;
		my_snorm += val*val;
		separator_pos = buf.find(" ", separator_pos + 1);
		colon_pos = buf.find(":", colon_pos + 1);
		ind++;

	}

    //unsigned int ind = 0;
    //model_file >> ind;
    //while (model_file.good()) {
   // char c;
	//model_file >> c;
  
   // double val;
   // model_file >> val;
   // my_v[ind] = val;
    //my_snorm += val*val;
   // model_file >> ind;
  //}
   }
    
  
  // destructor
  ~WeightVector() { delete[] my_v;  }

  // this *= s
  void scale(double s) { 
    my_snorm *= (s*s); 
    if (s != 0.0) {
      my_a *= s; 
    } else {
      my_a = 1.0;
      for (uint i=0; i<d; ++i) my_v[i] = 0.0;
    }
  }
  
  // this += s*x
  void add(simple_sparse_vector& x, double s);

  // this += s*x
  void add(WeightVector& x, double s);

  void print(std::ostream& os);

  double operator[](uint i) {
    if (i < d) return (my_v[i] * my_a);
    return 0.0;
  }

  uint dimension() {return d;}

  // ||this||^2
  double snorm() {
    return my_snorm;
  }

  // copy other
  void operator= (const WeightVector& other) {
    if (d != other.d) {
      std::cerr << "Assigning WeightVector of size "
		<< other.d << " to a WeightVector of size "
		<< d << " is not permitted" << std::endl;
      exit(EXIT_FAILURE);
    }
    my_a = other.my_a; my_snorm = other.my_snorm;
    for (uint i=0; i<d; ++i) my_v[i] = other.my_v[i];
  }

  // make_my_a_one
  // use it for forcing my_a to be 1.0
  void make_my_a_one() {
    for (uint i=0; i<d; ++i) {
      my_v[i] *= my_a;
    }
    my_a = 1.0;
  }

 //Remakes Weightvector with new dimension 
  void setNewDimension(uint dimension);

 private:  
  // The internal representation of w is as w = a*v where:
  uint d;
  double my_a; 
  double my_snorm;
  double* my_v;
};


//-----------------------------------------------------------------------------
/** Operator * for vector-vector multiplication
    @param u A reference to a simple_sparse_vector
    @param v A reference to a WeightVector
    @return The product (double)
*/
double operator* (simple_sparse_vector& u, WeightVector& v);

//-----------------------------------------------------------------------------
/** Operator * for vector-vector multiplication
    @param v A reference to a WeightVector
    @param u A reference to a simple_sparse_vector
    @return The product (double)
*/
double operator* (WeightVector& v, simple_sparse_vector& u);




#endif
