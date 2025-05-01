#include "Convolution.h"
#include <cassert>
#include <cmath>

Convolution::Convolution(int kernel_size, int kernel_depth, int in_channel, int padding, int stride)
    : kernel_size(kernel_size),
    kernel_depth(kernel_depth),
    in_channel(in_channel),
    padding(padding),
    stride(stride),
    kernel(kernel_depth* in_channel* kernel_size* kernel_size)
{
    kernel.setRandom();  // initialize weights
}

Eigen::ArrayXf Convolution::forward(const Eigen::ArrayXf& input)
{
    // 1) store input for backward
    this->input = input;

    // 2) infer H, W
    const int total_spatial = input.size() / in_channel;
    const int H = static_cast<int>(std::round(std::sqrt(total_spatial)));
    const int W = H;
    assert(H * W * in_channel == input.size());

    // 3) compute output dims
    const int OH = (H - kernel_size + 2 * padding) / stride + 1;
    const int OW = (W - kernel_size + 2 * padding) / stride + 1;

    // 4) allocate activation (we'll flatten it)
    activation.resize(kernel_depth * OH * OW);

    // 5) convolution
    for (int k = 0; k < kernel_depth; ++k) {
        // pointer to this filter’s weights
        const int K2C = in_channel * kernel_size * kernel_size;
        const float* wptr = kernel.data() + k * K2C;

        for (int oy = 0; oy < OH; ++oy) {
            for (int ox = 0; ox < OW; ++ox) {
                float sum = 0.f;
                // slide window
                for (int c = 0; c < in_channel; ++c) {
                    const int channel_off = c * H * W;
                    const int filt_off = c * kernel_size * kernel_size;
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        const int in_y = oy * stride - padding + ky;
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            const int in_x = ox * stride - padding + kx;
                            float val = 0.f;
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                val = input[channel_off + in_y * W + in_x];
                            }
                            sum += wptr[filt_off + ky * kernel_size + kx] * val;
                        }
                    }
                }
                activation[k * OH * OW + oy * OW + ox] = sum;
            }
        }
    }

    return activation;
}

Eigen::ArrayXf Convolution::backward(const Eigen::ArrayXf& gradient)
{
    // 1) infer H, W again
    const int total_spatial = input.size() / in_channel;
    const int H = static_cast<int>(std::round(std::sqrt(total_spatial)));
    const int W = H;
    const int OH = (H - kernel_size + 2 * padding) / stride + 1;
    const int OW = OH;

    // 2) prepare gradients
    Eigen::ArrayXf grad_input = Eigen::ArrayXf::Zero(input.size());
    Eigen::ArrayXf grad_kernel = Eigen::ArrayXf::Zero(kernel.size());

    const float lr = 0.01f;

    // 3) accumulate gradients
    for (int k = 0; k < kernel_depth; ++k) {
        const int K2C = in_channel * kernel_size * kernel_size;
        float* wgrad_ptr = grad_kernel.data() + k * K2C;

        for (int oy = 0; oy < OH; ++oy) {
            for (int ox = 0; ox < OW; ++ox) {
                const float g = gradient[k * OH * OW + oy * OW + ox];
                // for each position in the patch
                for (int c = 0; c < in_channel; ++c) {
                    const int channel_off = c * H * W;
                    const int filt_off = c * kernel_size * kernel_size;
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        const int in_y = oy * stride - padding + ky;
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            const int in_x = ox * stride - padding + kx;
                            float inp_val = 0.f;
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                inp_val = input[channel_off + in_y * W + in_x];
                                // grad wrt input
                                grad_input[channel_off + in_y * W + in_x] +=
                                    kernel[k * K2C + filt_off + ky * kernel_size + kx] * g;
                            }
                            // grad wrt kernel
                            wgrad_ptr[filt_off + ky * kernel_size + kx] += inp_val * g;
                        }
                    }
                }
            }
        }
    }

    // 4) SGD update
    kernel -= lr * grad_kernel;

    return grad_input;
}
