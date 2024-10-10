#include "Activation.h"

Eigen::VectorXd Activation::activation(Eigen::VectorXd& vec)
{
    Eigen::ArrayXd temp = (-vec.array()).exp();
    Eigen::ArrayXd one = Eigen::ArrayXd::Ones(vec.size());
    return (one / (temp + one)).matrix();
}
Eigen::VectorXd Activation::activationN(Eigen::VectorXd vec)
{
    Eigen::ArrayXd temp = (-vec.array()).exp();
    Eigen::ArrayXd one = Eigen::ArrayXd::Ones(vec.size());
    return (one / (temp + one)).matrix();
}

Eigen::VectorXd Activation::derivative(Eigen::VectorXd& vec)
{
    auto temp = activation(vec);
    return (temp.array() * (-temp + Eigen::VectorXd::Ones(vec.size())).array()).matrix();
}
