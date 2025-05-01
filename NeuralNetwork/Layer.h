#pragma once
#include"Eigen/Dense"
#include"Activation.h"

class Layer
{
public:
	virtual Eigen::ArrayXf forward(const Eigen::ArrayXf& input) { static_assert("Unimplemented forward"); return Eigen::ArrayXf(); }
	virtual Eigen::ArrayXf backward(const Eigen::ArrayXf& gradient) { static_assert("Unimplemented forward"); return Eigen::ArrayXf();}
};

