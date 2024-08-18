#include "resnet.cuh"

#include <curand.h>



static void resnet_init(INeuralNet* self, const IGame* game) {
    ResNetModel* model = (ResNetModel*)malloc(sizeof(ResNetModel));
    self->impl = model;

    // Initialize model configuration based on game
    init_model_config(model, game);

    // Initialize cuDNN
    cudnnCreate(&model->cudnn_handle);

    // Initialize input block
    init_input_block(model);

    // Initialize residual blocks
    init_residual_blocks(model);

    // Initialize output block
    init_output_block(model);

    // Initialize weights with small random values
    init_weights(model);
}

static void init_model_config(ResNetModel* model, const IGame* game) {
    // Set up model configuration based on game parameters
    int rows, cols;
    game->get_board_size(game, &rows, &cols);
    model->config.input_channels = 3;  // Assuming 3 channels for player 1, player 2, and turn
    model->config.input_height = rows;
    model->config.input_width = cols;
    model->config.num_actions = game->get_action_size(game);
    model->config.num_residual_blocks = 19;  // AlphaZero used 19 residual blocks
    model->config.num_filters = 256;
    model->config.learning_rate = 0.001;
    model->config.weight_decay = 0.0001;
}

static void init_input_block(ResNetModel* model) {
    // Create and initialize convolution filter descriptor
    cudnnCreateFilterDescriptor(&model->input_conv_filter);
    cudnnSetFilter4dDescriptor(model->input_conv_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               model->config.num_filters, model->config.input_channels,
                               3, 3);  // 3x3 convolution

    // Allocate memory for convolution weights
    cudaMalloc(&model->input_conv_weights, sizeof(float) * model->config.num_filters * model->config.input_channels * 3 * 3);

    // Create and initialize batch normalization descriptors
    cudnnCreateTensorDescriptor(&model->input_bn_mean);
    cudnnCreateTensorDescriptor(&model->input_bn_var);
    cudnnSetTensor4dDescriptor(model->input_bn_mean, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, model->config.num_filters, 1, 1);
    cudnnSetTensor4dDescriptor(model->input_bn_var, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, model->config.num_filters, 1, 1);

    // Allocate memory for batch normalization parameters
    cudaMalloc(&model->input_bn_scale, sizeof(float) * model->config.num_filters);
    cudaMalloc(&model->input_bn_bias, sizeof(float) * model->config.num_filters);
}

static void init_residual_blocks(ResNetModel* model) {
    // Allocate arrays for residual block parameters
    model->res_conv_filters = malloc(sizeof(cudnnFilterDescriptor_t) * model->config.num_residual_blocks * 2);
    model->res_bn_means = malloc(sizeof(cudnnTensorDescriptor_t) * model->config.num_residual_blocks * 2);
    model->res_bn_vars = malloc(sizeof(cudnnTensorDescriptor_t) * model->config.num_residual_blocks * 2);
    model->res_conv_weights = malloc(sizeof(float*) * model->config.num_residual_blocks * 2);
    model->res_bn_scales = malloc(sizeof(float*) * model->config.num_residual_blocks * 2);
    model->res_bn_biases = malloc(sizeof(float*) * model->config.num_residual_blocks * 2);

    for (int i = 0; i < model->config.num_residual_blocks; i++) {
        for (int j = 0; j < 2; j++) {
            int idx = i * 2 + j;
            // Create and initialize convolution filter descriptor
            cudnnCreateFilterDescriptor(&model->res_conv_filters[idx]);
            cudnnSetFilter4dDescriptor(model->res_conv_filters[idx], CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                       model->config.num_filters, model->config.num_filters,
                                       3, 3);  // 3x3 convolution

            // Allocate memory for convolution weights
            cudaMalloc(&model->res_conv_weights[idx], sizeof(float) * model->config.num_filters * model->config.num_filters * 3 * 3);

            // Create and initialize batch normalization descriptors
            cudnnCreateTensorDescriptor(&model->res_bn_means[idx]);
            cudnnCreateTensorDescriptor(&model->res_bn_vars[idx]);
            cudnnSetTensor4dDescriptor(model->res_bn_means[idx], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       1, model->config.num_filters, 1, 1);
            cudnnSetTensor4dDescriptor(model->res_bn_vars[idx], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       1, model->config.num_filters, 1, 1);

            // Allocate memory for batch normalization parameters
            cudaMalloc(&model->res_bn_scales[idx], sizeof(float) * model->config.num_filters);
            cudaMalloc(&model->res_bn_biases[idx], sizeof(float) * model->config.num_filters);
        }
    }
}

static void init_output_block(ResNetModel* model) {
    // Initialize value head
    cudnnCreateFilterDescriptor(&model->value_conv_filter);
    cudnnSetFilter4dDescriptor(model->value_conv_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               1, model->config.num_filters, 1, 1);  // 1x1 convolution
    cudaMalloc(&model->value_conv_weights, sizeof(float) * model->config.num_filters);

    cudnnCreateTensorDescriptor(&model->value_bn_mean);
    cudnnCreateTensorDescriptor(&model->value_bn_var);
    cudnnSetTensor4dDescriptor(model->value_bn_mean, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 1, 1, 1);
    cudnnSetTensor4dDescriptor(model->value_bn_var, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 1, 1, 1);

    cudaMalloc(&model->value_bn_scale, sizeof(float));
    cudaMalloc(&model->value_bn_bias, sizeof(float));

    // Allocate memory for fully connected layers in value head
    int fc1_size = model->config.input_height * model->config.input_width;
    cudaMalloc(&model->value_fc1_weights, sizeof(float) * fc1_size * 256);
    cudaMalloc(&model->value_fc1_bias, sizeof(float) * 256);
    cudaMalloc(&model->value_fc2_weights, sizeof(float) * 256);
    cudaMalloc(&model->value_fc2_bias, sizeof(float));

    // Initialize policy head
    cudnnCreateFilterDescriptor(&model->policy_conv_filter);
    cudnnSetFilter4dDescriptor(model->policy_conv_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               2, model->config.num_filters, 1, 1);  // 1x1 convolution
    cudaMalloc(&model->policy_conv_weights, sizeof(float) * 2 * model->config.num_filters);

    cudnnCreateTensorDescriptor(&model->policy_bn_mean);
    cudnnCreateTensorDescriptor(&model->policy_bn_var);
    cudnnSetTensor4dDescriptor(model->policy_bn_mean, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 2, 1, 1);
    cudnnSetTensor4dDescriptor(model->policy_bn_var, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 2, 1, 1);

    cudaMalloc(&model->policy_bn_scale, sizeof(float) * 2);
    cudaMalloc(&model->policy_bn_bias, sizeof(float) * 2);

    // Allocate memory for fully connected layer in policy head
    int policy_fc_size = 2 * model->config.input_height * model->config.input_width;
    cudaMalloc(&model->policy_fc_weights, sizeof(float) * policy_fc_size * model->config.num_actions);
    cudaMalloc(&model->policy_fc_bias, sizeof(float) * model->config.num_actions);
}

static void init_weights(ResNetModel* model) {
    // Initialize weights with small random values
    // You can use cuRAND for this purpose
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    // Initialize input block weights
    curandGenerateNormal(gen, model->input_conv_weights, model->config.num_filters * model->config.input_channels * 3 * 3, 0, 0.1);
    curandGenerateNormal(gen, model->input_bn_scale, model->config.num_filters, 1, 0.1);
    curandGenerateNormal(gen, model->input_bn_bias, model->config.num_filters, 0, 0.1);

    // Initialize residual block weights
    for (int i = 0; i < model->config.num_residual_blocks * 2; i++) {
        curandGenerateNormal(gen, model->res_conv_weights[i], model->config.num_filters * model->config.num_filters * 3 * 3, 0, 0.1);
        curandGenerateNormal(gen, model->res_bn_scales[i], model->config.num_filters, 1, 0.1);
        curandGenerateNormal(gen, model->res_bn_biases[i], model->config.num_filters, 0, 0.1);
    }

    // Initialize output block weights
    curandGenerateNormal(gen, model->value_conv_weights, model->config.num_filters, 0, 0.1);
    curandGenerateNormal(gen, model->value_bn_scale, 1, 1, 0.1);
    curandGenerateNormal(gen, model->value_bn_bias, 1, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc1_weights, model->config.input_height * model->config.input_width * 256, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc1_bias, 256, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc2_weights, 256, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc2_bias, 1, 0, 0.1);

    curandGenerateNormal(gen, model->policy_conv_weights, 2 * model->config.num_filters, 0, 0.1);
    curandGenerateNormal(gen, model->policy_bn_scale, 2, 1, 0.1);
    curandGenerateNormal(gen, model->policy_bn_bias, 2, 0, 0.1);
    curandGenerateNormal(gen, model->policy_fc_weights, 2 * model->config.input_height * model->config.input_width * model->config.num_actions, 0, 0.1);
    curandGenerateNormal(gen, model->policy_fc_bias, model->config.num_actions, 0, 0.1);

    curandDestroyGenerator(gen);
}