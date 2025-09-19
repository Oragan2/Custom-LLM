#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <map>
#include <cuda_runtime.h>
#include "transformer.hpp"

// GPU functions

// CPU functions
// Helper
std::vector<std::vector<float>> initialize_matrix(int rows, int cols)
{
    float limit = 0.1f;
    std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            mat[i][j] = dist(gen);

    return mat;
}

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B)
{
    int a_rows = A.size();
    int a_cols = A[0].size();
    int b_rows = B.size();
    int b_cols = B[0].size();

    if (a_cols != b_rows)
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");

    std::vector<std::vector<float>> C(a_rows, std::vector<float>(b_cols, 0.0f));

    for (int i = 0; i < a_rows; i++)
        for (int j = 0; j < b_cols; j++)
            for (int k = 0; k < a_cols; k++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& mat)
{
    std::vector<std::vector<float>> result = mat;
    for (auto& row : result)
    {
        float max_val = *std::max_element(row.begin(), row.end());
        float sum = 0.0f;
        for (auto& val : row)
        {
            val = std::exp(val - max_val);
            sum += val;
        }
        for (auto& val : row)
            val /= sum;
    }
    return result;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            transposed[j][i] = mat[i][j];
    return transposed;
}

MultiHeadAttention::MultiHeadAttention(int h, int d) : num_heads(h), hidden_dim(d)
{
    head_dim = hidden_dim / num_heads;
    W_Q = initialize_matrix(hidden_dim, hidden_dim);
    W_K = initialize_matrix(hidden_dim, hidden_dim);
    W_V = initialize_matrix(hidden_dim, hidden_dim);
    W_O = initialize_matrix(hidden_dim, hidden_dim);
}

std::vector<std::vector<float>> MultiHeadAttention::forward(const std::vector<std::vector<float>>& X) {
    // 1) projections
    std::vector<std::vector<float>> Q = matmul(X, W_Q);
    std::vector<std::vector<float>> K = matmul(X, W_K);
    std::vector<std::vector<float>> V = matmul(X, W_V);

    int seq_len = X.size();
    int head_dim = hidden_dim / num_heads;

    // 2) split heads
    std::vector<std::vector<float>> concat(seq_len, std::vector<float>(hidden_dim, 0.0f));
    for (int h = 0; h < num_heads; h++) {
        // slice per head
        std::vector<std::vector<float>> Qh(seq_len, std::vector<float>(head_dim));
        std::vector<std::vector<float>> Kh(seq_len, std::vector<float>(head_dim));
        std::vector<std::vector<float>> Vh(seq_len, std::vector<float>(head_dim));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                Qh[i][j] = Q[i][h * head_dim + j];
                Kh[i][j] = K[i][h * head_dim + j];
                Vh[i][j] = V[i][h * head_dim + j];
            }
        }
        // 3) attention: softmax((QK^T)/sqrt(dk)) * V
        auto scores = matmul(Qh, transpose(Kh));
        for (auto& row : scores) for (auto& v : row) v /= std::sqrt((float)head_dim);
        auto attn = softmax(scores);
        auto head_out = matmul(attn, Vh);

        // 4) copy into concat
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                concat[i][h * head_dim + j] = head_out[i][j];
            }
        }
    }

    // 5) final projection
    return matmul(concat, W_O);
}

LLM::LLM(int vocab_size, int max_seq_len, int hidden_dim, int num_head)
    : embeding(vocab_size, std::vector<float>(hidden_dim, 0.0f)),
      positionalEncoding(sinusoidalEncoding(max_seq_len, hidden_dim)),
      transformer(num_head, hidden_dim){}

// temp on the CPU for testing the logic
std::vector<std::vector<float>> LLM::tokenEmbeding(std::vector<int>& tokens)
{
    size_t seq_len = tokens.size();
    size_t hidden_dim = embeding[0].size();

    std::vector<std::vector<float>> x(seq_len, std::vector<float>(hidden_dim, 0.0f));

    for (size_t i = 0; i < seq_len; i++)
    {
        int token_id = tokens[i];
        for (size_t d = 0; d < hidden_dim; d++)
        {
            x[i][d] = embeding[token_id][d] + positionalEncoding[i][d];
        }
    }
    return x;
}

// temp on the CPU for testing logic
std::vector<std::vector<float>> LLM::forwardPass(std::string text) {
    std::vector<int> tokens = {286,5012,3795};
    // TODO : make a real tokenizer
    auto X = tokenEmbeding(tokens);
    return transformer.forward(X);
}

// temp will be changed later
std::vector<std::vector<float>> LLM::sinusoidalEncoding(int seq_len, int dim)
{
    std::vector<std::vector<float>> pe(seq_len, std::vector<float>(dim));

    for (int pos = 0; pos < seq_len; pos++)
    {
        for (int i = 0; i < dim; i++)
        {
            double angle = pos / std::pow(10000.0, (2 * (i / 2)) / (double)dim);
            if (i % 2 == 0)
                pe[pos][i] = std::sin(angle);
            else
                pe[pos][i] = std::cos(angle);
        }
    }
    return pe;
}