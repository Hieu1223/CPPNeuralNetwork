#pragma once
#include<iostream>
#include<vector>
#include"Layer.h"
#include"Dataset.h"
#include"Eigen/Dense"
#include<thread>
#include<filesystem>

class Model
{	
public:
	std::vector<Layer> layers;
	double(*cost)(Eigen::VectorXd& vec);
	double learningRate = 0.001f;
	double momentumMultiplier = 0.9f;
	std::vector<Eigen::VectorXd> BiasMomentum;
	std::vector<Eigen::MatrixXd> WeightMomentum;
public:
	Model& layer(Layer& layer);
public:
	Eigen::VectorXd getOutput(Eigen::VectorXd& input);
	double backpropagation(const Eigen::VectorXd& input, const Eigen::VectorXd& expected_output, int data_size, std::vector<Eigen::VectorXd>& BiasGradient, std::vector<Eigen::MatrixXd>& WeightGradient);
	double train(Dataset&data,int batchSize= 10);
	double trainInBatch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& input, int start, int stop, std::vector<Eigen::VectorXd>&Bias, std::vector<Eigen::MatrixXd>& Weights);
	double batchingTraining(Dataset& data, int batchSize = 10);
public:
	void BackUp(std::filesystem::path path_to_file);
	void LoadBackUp(std::filesystem::path path_to_file);
};

