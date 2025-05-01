// main.cpp
#include <iostream>
#include "Eigen/Dense"
#include "Convolution.h"
#include"VisionTransformer.h"

#include <random>

// Cross‐entropy loss for logits -> one‐hot target
float cross_entropy_loss(const Eigen::ArrayXf& logits, int target_label) {
    // apply softmax
    Eigen::ArrayXf exps = (logits - logits.maxCoeff()).exp();
    Eigen::ArrayXf probs = exps / exps.sum();
    // loss = -log p_target
    return -std::log(std::max(1e-7f, probs(target_label)));
}

// Gradient of cross‐entropy w.r.t. logits
Eigen::ArrayXf grad_cross_entropy(const Eigen::ArrayXf& logits, int target_label) {
    Eigen::ArrayXf exps = (logits - logits.maxCoeff()).exp();
    Eigen::ArrayXf probs = exps / exps.sum();
    probs(target_label) -= 1.0f;  // dL/dz = p - y_onehot
    return probs;
}

int main() {
    // --- Hyperparameters ---
    const int IMG_H = 32, IMG_W = 32, IN_CH = 3;
    const int PATCH = 4, DIM = 128, DEPTH = 6, NUM_CLASSES = 10;
    const float LR = 0.01f;

    // 1) Instantiate Vision Transformer
    VisionTransformer vit(IMG_H, IMG_W, IN_CH, PATCH, DIM, DEPTH, NUM_CLASSES);

    // 2) Create one random image and random label
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);
    Eigen::ArrayXf img(IMG_H * IMG_W * IN_CH);
    for (int i = 0; i < img.size(); ++i) img[i] = udist(rng);

    std::uniform_int_distribution<int> ldist(0, NUM_CLASSES - 1);
    int label = ldist(rng);
    std::cout << "Target label: " << label << std::endl;

    for(int i = 0; i< 100; i++)
    {
        // 3) Forward pass
        Eigen::ArrayXf logits = vit.forward(img);

        // 4) Compute loss
        float loss = cross_entropy_loss(logits, label);
        std::cout << "Initial loss: " << loss << std::endl;

        // 5) Backward pass
        Eigen::ArrayXf grad_logits = grad_cross_entropy(logits, label);
        Eigen::ArrayXf grad_img = vit.backward(grad_logits);
        Eigen::ArrayXf logits2 = vit.forward(img);
        float loss2 = cross_entropy_loss(logits2, label);
        std::cout << "Loss after one update: " << loss2 << std::endl;
    }

    return 0;
}
