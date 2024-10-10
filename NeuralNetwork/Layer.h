#pragma once
#include"Eigen/Dense"
#include"Activation.h"

class Layer
{
public:
	Eigen::MatrixXd*weights;
	Eigen::VectorXd*Bias;
public:
	Activation *activation;
public:
	Layer(int input_count, int output_count, Activation* func) 
	{
		
		this->activation = func;
		this->weights = new Eigen::MatrixXd(output_count, input_count);
		this->Bias = new Eigen::VectorXd(output_count);
		this->weights->setRandom();
		this->Bias->setRandom();
		this->inputCount = input_count;
		this->outputCount = output_count;
	}

public:
	int inputCount, outputCount;
public:
	Eigen::VectorXd getOutPut(Eigen::VectorXd input);
};

