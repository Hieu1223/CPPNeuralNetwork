#include "Linear.h"


Linear::Linear(int inFeature, int outFeature)
{
    this->inFeature = inFeature;
    this->outFeature = outFeature;
	this->bias = Eigen::MatrixXf::Random(outFeature,1);
	this->weight = Eigen::MatrixXf::Random(outFeature, inFeature);
}

Eigen::ArrayXf Linear::forward(const Eigen::ArrayXf& input) {
    assert(input.size() == inFeature && "input length must equal inFeature");

    // Map the ArrayXf to a column vector of size (inFeature × 1)
    Eigen::Map<const Eigen::VectorXf>  x_vec(input.data(), inFeature);

    // Do the mat‐vec multiply, then add bias
    Eigen::VectorXf out = weight * x_vec + bias.col(0);

    // Store activation as an ArrayXf if you need elementwise ops later
    activation = out.array();
    this->input = input;
    return activation;
}

Eigen::ArrayXf Linear::backward(const Eigen::ArrayXf& gradient)
{
    // Apply ReLU derivative: 1 where activation > 0, else 0
    Eigen::ArrayXf relu_derivative = (this->activation.array() > 0).cast<float>();
    Eigen::ArrayXf delta = gradient * relu_derivative;

    // Compute gradients
    Eigen::MatrixXf grad_weight = delta.matrix() * input.transpose().matrix();
    Eigen::MatrixXf grad_bias = delta.matrix();

    // SGD update
    float learning_rate = 0.01;  // Fixed learning rate
    this->weight -= learning_rate * grad_weight;
    this->bias -= learning_rate * grad_bias;

    // Gradient w.r.t. input
    Eigen::ArrayXf grad_input = (this->weight.transpose() * delta.matrix()).array();
    return grad_input;
}