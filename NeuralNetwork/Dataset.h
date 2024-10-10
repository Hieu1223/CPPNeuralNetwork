#pragma once
#include<vector>
#include"Eigen/Dense"
class Dataset
{
public:
	virtual const std::vector < std::pair<Eigen::VectorXd, Eigen::VectorXd>>&getAll();
};

