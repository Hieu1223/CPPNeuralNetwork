#pragma once
#include"Layer.h"
class Linear : public Layer
{
public:
	Eigen::MatrixXf weight;
	Eigen::MatrixXf bias;
	Eigen::MatrixXf input;
	Eigen::MatrixXf activation;
	int inFeature;
	int outFeature;
public:
	Linear(int inFeature, int outFeature);
public:
	Eigen::ArrayXf forward(const Eigen::ArrayXf&input) override;
	Eigen::ArrayXf backward(const Eigen::ArrayXf&gradient) override;
};

