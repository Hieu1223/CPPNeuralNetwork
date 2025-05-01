#pragma once
#include <vector>
#include "Layer.h"
#include "Linear.h"
#include "Convolution.h"
#include "Eigen/Dense"

// Simple LayerNorm implementation
class LayerNorm : public Layer {
private:
    int dim;
    Eigen::ArrayXf gamma, beta;
public:
    LayerNorm(int dim) : dim(dim) {
        gamma = Eigen::ArrayXf::Ones(dim);
        beta = Eigen::ArrayXf::Zero(dim);
    }
    Eigen::ArrayXf forward(const Eigen::ArrayXf& x) override {
        const int N = x.size();
        const int L = dim;
        const int seqLen = N / L;
        Eigen::Map<const Eigen::MatrixXf> Xmat(x.data(), L, seqLen);
        Eigen::MatrixXf Xhat(L, seqLen);
        for (int t = 0; t < seqLen; ++t) {
            auto col = Xmat.col(t);
            float m = col.mean();
            float v = (col.array() - m).square().mean();
            Xhat.col(t) = (col.array() - m) / std::sqrt(v + 1e-5f);
        }
        Eigen::ArrayXXf A = Xhat.array();
        A.colwise() *= gamma;
        A.colwise() += beta;
        return Eigen::Map<Eigen::ArrayXf>(A.data(), A.size());
    }
    Eigen::ArrayXf backward(const Eigen::ArrayXf& grad) override {
        return grad;
    }
};

// Self-Attention using per-token Linear projections
class SelfAttention : public Layer {
private:
    int dim;
    Linear* proj_q, * proj_k, * proj_v, * proj_out;
public:
    SelfAttention(int dim) : dim(dim) {
        proj_q = new Linear(dim, dim);
        proj_k = new Linear(dim, dim);
        proj_v = new Linear(dim, dim);
        proj_out = new Linear(dim, dim);
    }
    Eigen::ArrayXf forward(const Eigen::ArrayXf& x) override {
        int seqLen = x.size() / dim;
        // Build Q, K, V by projecting each token separately
        Eigen::MatrixXf Q(dim, seqLen), K(dim, seqLen), V(dim, seqLen);
        for (int i = 0; i < seqLen; ++i) {
            Eigen::ArrayXf token = x.segment(i * dim, dim);
            Q.col(i) = proj_q->forward(token).matrix();
            K.col(i) = proj_k->forward(token).matrix();
            V.col(i) = proj_v->forward(token).matrix();
        }
        // Compute scaled dot-product attention
        Eigen::MatrixXf scores = Q.transpose() * K;  // (seqLen x seqLen)
        scores.array() /= std::sqrt((float)dim);
        // Numerically stable softmax
        Eigen::MatrixXf attn(scores.rows(), scores.cols());
        for (int i = 0; i < scores.rows(); ++i) {
            float m = scores.row(i).maxCoeff();
            Eigen::ArrayXf ex = (scores.row(i).array() - m).exp();
            attn.row(i) = (ex / ex.sum()).matrix();
        }
        // Attention output
        Eigen::MatrixXf out_mat = V * attn.transpose(); // (dim x seqLen)
        // Project output tokens
        Eigen::ArrayXf out(dim * seqLen);
        for (int i = 0; i < seqLen; ++i) {
            Eigen::Map<Eigen::ArrayXf> tok_out(out_mat.col(i).data(), dim);
            Eigen::ArrayXf proj = proj_out->forward(tok_out);
            out.segment(i * dim, dim) = proj;
        }
        return out;
    }
    Eigen::ArrayXf backward(const Eigen::ArrayXf& grad) override {
        return grad;
    }
};

class TransformerEncoderBlock : public Layer {
private:
    int dim;
    LayerNorm* ln1, * ln2;
    SelfAttention* attn;
    Linear* mlp1, * mlp2;
public:
    TransformerEncoderBlock(int dim) : dim(dim) {
        ln1 = new LayerNorm(dim);
        ln2 = new LayerNorm(dim);
        attn = new SelfAttention(dim);
        mlp1 = new Linear(dim, dim * 4);
        mlp2 = new Linear(dim * 4, dim);
    }
    Eigen::ArrayXf forward(const Eigen::ArrayXf& x) override {
        Eigen::ArrayXf x1 = ln1->forward(x);
        Eigen::ArrayXf a1 = attn->forward(x1);
        Eigen::ArrayXf r1 = x + a1;
        Eigen::ArrayXf x2 = ln2->forward(r1);
        // MLP per token
        int seqLen = x.size() / dim;
        Eigen::ArrayXf mlp_out(seqLen * dim);
        for (int i = 0; i < seqLen; ++i) {
            Eigen::ArrayXf tok = x2.segment(i * dim, dim);
            Eigen::ArrayXf h1 = mlp1->forward(tok);
            Eigen::ArrayXf h2 = mlp2->forward(h1);
            mlp_out.segment(i * dim, dim) = h2;
        }
        return r1 + mlp_out;
    }
    Eigen::ArrayXf backward(const Eigen::ArrayXf& grad) override {
        return grad;
    }
};

class VisionTransformer : public Layer {
private:
    int img_h, img_w, in_ch, patch_size, num_patches, dim;
    Convolution* patch_embed;
    Eigen::ArrayXf pos_embed;
    std::vector<TransformerEncoderBlock*> blocks;
    LayerNorm* pre_ln;
    Linear* head;
    Eigen::ArrayXf last_out;
public:
    VisionTransformer(int img_h, int img_w, int in_ch, int patch_size,
        int dim, int depth, int num_classes)
        : img_h(img_h), img_w(img_w), in_ch(in_ch), patch_size(patch_size), dim(dim)
    {
        num_patches = (img_h / patch_size) * (img_w / patch_size);
        patch_embed = new Convolution(patch_size, dim, in_ch, 0, patch_size);
        pos_embed = Eigen::ArrayXf::Random(dim * num_patches);
        for (int i = 0; i < depth; ++i)
            blocks.push_back(new TransformerEncoderBlock(dim));
        pre_ln = new LayerNorm(dim);
        head = new Linear(dim, num_classes);
    }
    Eigen::ArrayXf forward(const Eigen::ArrayXf& x) override {
        // Patch embedding
        Eigen::ArrayXf patches = patch_embed->forward(x);
        // Add positional
        Eigen::ArrayXf seq = patches + pos_embed;
        // Encoder
        Eigen::ArrayXf out = seq;
        for (auto& blk : blocks)
            out = blk->forward(out);
        // Pre-norm
        Eigen::ArrayXf cls = pre_ln->forward(out);
        // Classification head (mean pool)
        int seqLen = cls.size() / dim;
        Eigen::ArrayXf logits = Eigen::ArrayXf::Zero(head->bias.size());
        for (int i = 0; i < seqLen; ++i) {
            logits += head->forward(cls.segment(i * dim, dim));
        }
        logits /= seqLen;
        last_out = cls;  // cache for backward
        return logits;
    }
    Eigen::ArrayXf backward(const Eigen::ArrayXf& grad) override {
        int seqLen = last_out.size() / dim;
        // grad: shape num_classes
        // Head backward for each token
        Eigen::ArrayXf grad_cls = Eigen::ArrayXf::Zero(last_out.size());
        for (int i = 0; i < seqLen; ++i) {
            Eigen::ArrayXf token = last_out.segment(i * dim, dim);
            Eigen::ArrayXf d_head = head->backward(grad / seqLen);
            grad_cls.segment(i * dim, dim) += d_head;
        }
        // Pre-ln backward
        Eigen::ArrayXf grad_enc = pre_ln->backward(grad_cls);
        // Encoder blocks backward in reverse
        for (int i = blocks.size() - 1; i >= 0; --i) {
            grad_enc = blocks[i]->backward(grad_enc);
        }
        // Remove pos_embed
        Eigen::ArrayXf grad_patches = grad_enc;  // addition gradient
        // Patch embed backward
        Eigen::ArrayXf grad_input = patch_embed->backward(grad_patches);
        return grad_input;
    }
};
