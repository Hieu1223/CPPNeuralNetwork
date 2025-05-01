#pragma once
#include"Layer.h"
class Convolution : public Layer
{
private: 
	int kernel_size;
	int kernel_depth;
	int in_channel;
	int padding;
	int stride;
private:
	Eigen::ArrayXf kernel;
	Eigen::ArrayXf input;
	Eigen::ArrayXf activation;

public:
	Convolution(int kernel_size,int kernel_depth,int in_channel ,int padding, int stride);
public:
	Eigen::ArrayXf forward(const Eigen::ArrayXf& input) override;
	Eigen::ArrayXf backward(const Eigen::ArrayXf& gradient) override;
};

