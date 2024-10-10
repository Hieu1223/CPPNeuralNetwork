#pragma once
#include"Eigen/Dense"
class Activation
{
public:
	Eigen::VectorXd activation(Eigen::VectorXd& vec);
	Eigen::VectorXd activationN(Eigen::VectorXd vec);
	Eigen::VectorXd derivative(Eigen::VectorXd& vec);
};

