#ifndef RESNET_CUH
#define RESNET_CUH

#include "../neural_network.h"

#include <cuda_runtime.h>
#include <cudnn.h>


// Model configuration
typedef struct {
    int input_channels;
    int input_height;
    int input_width;
    int num_actions;
    int num_residual_blocks;
    int num_filters;
    float learning_rate;
    float weight_decay;
} ModelConfig;

// ResNet model
typedef struct {
    // Input block
    cudnnFilterDescriptor_t input_conv_filter;
    cudnnTensorDescriptor_t input_bn_mean, input_bn_var;
    float *input_conv_weights, *input_bn_scale, *input_bn_bias;

    // Residual blocks
    cudnnFilterDescriptor_t *res_conv_filters;
    cudnnTensorDescriptor_t *res_bn_means, *res_bn_vars;
    float **res_conv_weights, **res_bn_scales, **res_bn_biases;

    // Output block
    cudnnFilterDescriptor_t value_conv_filter, policy_conv_filter;
    cudnnTensorDescriptor_t value_bn_mean, value_bn_var, policy_bn_mean, policy_bn_var;
    float *value_conv_weights, *value_bn_scale, *value_bn_bias;
    float *policy_conv_weights, *policy_bn_scale, *policy_bn_bias;
    float *value_fc1_weights, *value_fc1_bias, *value_fc2_weights, *value_fc2_bias;
    float *policy_fc_weights, *policy_fc_bias;

    // CUDNN handles
    cudnnHandle_t cudnn_handle;
    
    // Model configuration
    ModelConfig config;
} ResNetModel;

// Function prototypes
INeuralNet* create_resnet_model(const IGame* game);

#endif // RESNET_CUH