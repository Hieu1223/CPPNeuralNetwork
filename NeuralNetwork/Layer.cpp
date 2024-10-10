#include "Layer.h"



Eigen::VectorXd Layer::getOutPut(Eigen::VectorXd input)
{
	return  *weights * input + *Bias;
}
