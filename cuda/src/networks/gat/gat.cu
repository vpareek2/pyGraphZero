#include "gat.cuh"

#include <curand.h>
#include <torch/serialize.h>

static void gat_init(INeuralNet* self, const IGame* game) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Initialize model configuration
    init_model_config(model, game);

    // Initialize cuDNN
    cudnnCreate(&model->cudnn_handle);

    // Initialize input block
    init_input_block(model);

    // Initialize GAT layers
    init_gat_layers(model);

    // Initialize output block
    init_output_block(model);

    // Initialize weights
    init_weights(model);

    // Initialize PyTorch optimizer
    std::vector<torch::Tensor> params;
    // Add all weights and biases to params vector
    // Input block
    params.push_back(torch::from_blob(model->input_weights, {model->config.input_features, model->config.hidden_features}, torch::kCUDA));
    params.push_back(torch::from_blob(model->input_bias, {model->config.hidden_features}, torch::kCUDA));
    // GAT layers
    for (int i = 0; i < model->config.num_layers; i++) {
        params.push_back(torch::from_blob(model->layer_weights[i], {model->config.hidden_features, model->config.hidden_features}, torch::kCUDA));
        params.push_back(torch::from_blob(model->layer_biases[i], {model->config.hidden_features}, torch::kCUDA));
        params.push_back(torch::from_blob(model->attention_weights[i], {model->config.num_heads, 2, model->config.hidden_features}, torch::kCUDA));
    }
    // Output block
    params.push_back(torch::from_blob(model->value_weights, {model->config.hidden_features}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_bias, {1}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_weights, {model->config.hidden_features, model->config.num_actions}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_bias, {model->config.num_actions}, torch::kCUDA));

    model->optimizer = new torch::optim::Adam(params, torch::optim::AdamOptions(model->config.learning_rate).weight_decay(model->config.weight_decay));

    // Allocate workspace for cuDNN
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(model->cudnn_handle, model->input_descriptor, model->layer_descriptors[0],
                                            /* convolution descriptor */, model->layer_descriptors[0],
                                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_size);
    cudaMalloc(&model->workspace, workspace_size);
    model->workspace_size = workspace_size;
}

INeuralNet* create_gat_model(const IGame* game) {
    GATWrapper* wrapper = (GATWrapper*)malloc(sizeof(GATWrapper));
    wrapper->base.impl = wrapper;
    wrapper->base.init = gat_init;
    wrapper->base.train = gat_train;
    wrapper->base.predict = gat_predict;
    wrapper->base.save_checkpoint = gat_save_checkpoint;
    wrapper->base.load_checkpoint = gat_load_checkpoint;
    wrapper->base.destroy = gat_destroy;

    gat_init(&wrapper->base, game);

    return &wrapper->base;
}

static void gat_save_checkpoint(INeuralNet* self, const char* folder, const char* filename) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    char filepath[MAX_FILENAME_LENGTH];
    snprintf(filepath, MAX_FILENAME_LENGTH, "%s/%s", folder, filename);

    torch::serialize::OutputArchive archive;

    // Save model configuration
    archive.write("config", torch::from_blob(&model->config, {sizeof(ModelConfig)}, torch::kByte));

    // Save input block weights
    archive.write("input_weights", torch::from_blob(model->input_weights, 
        {model->config.input_features, model->config.hidden_features}, torch::kFloat32));
    archive.write("input_bias", torch::from_blob(model->input_bias, 
        {model->config.hidden_features}, torch::kFloat32));

    // Save GAT layer weights
    for (int i = 0; i < model->config.num_layers; i++) {
        char key[50];
        snprintf(key, sizeof(key), "layer_weights_%d", i);
        archive.write(key, torch::from_blob(model->layer_weights[i], 
            {model->config.hidden_features, model->config.hidden_features}, torch::kFloat32));
        
        snprintf(key, sizeof(key), "layer_biases_%d", i);
        archive.write(key, torch::from_blob(model->layer_biases[i], 
            {model->config.hidden_features}, torch::kFloat32));
        
        snprintf(key, sizeof(key), "attention_weights_%d", i);
        archive.write(key, torch::from_blob(model->attention_weights[i], 
            {model->config.num_heads, 2, model->config.hidden_features}, torch::kFloat32));
    }

    // Save output block weights
    archive.write("value_weights", torch::from_blob(model->value_weights, 
        {model->config.hidden_features}, torch::kFloat32));
    archive.write("value_bias", torch::from_blob(model->value_bias, {1}, torch::kFloat32));
    archive.write("policy_weights", torch::from_blob(model->policy_weights, 
        {model->config.hidden_features, model->config.num_actions}, torch::kFloat32));
    archive.write("policy_bias", torch::from_blob(model->policy_bias, 
        {model->config.num_actions}, torch::kFloat32));

    // Save optimizer state
    archive.write("optimizer", model->optimizer->state_dict());

    torch::serialize::save_to_file(archive, filepath);
}

static void gat_load_checkpoint(INeuralNet* self, const char* folder, const char* filename) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    char filepath[MAX_FILENAME_LENGTH];
    snprintf(filepath, MAX_FILENAME_LENGTH, "%s/%s", folder, filename);

    torch::serialize::InputArchive archive;
    torch::serialize::load_from_file(archive, filepath);

    // Load model configuration
    torch::Tensor config_tensor;
    archive.read("config", config_tensor);
    memcpy(&model->config, config_tensor.data_ptr(), sizeof(ModelConfig));

    // Reallocate memory if necessary (in case the loaded model has different dimensions)
    // This part is omitted for brevity, but you should implement it in a production environment

    // Load input block weights
    torch::Tensor input_weights, input_bias;
    archive.read("input_weights", input_weights);
    archive.read("input_bias", input_bias);
    cudaMemcpy(model->input_weights, input_weights.data_ptr(), 
        input_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(model->input_bias, input_bias.data_ptr(), 
        input_bias.numel() * sizeof(float), cudaMemcpyDeviceToDevice);

    // Load GAT layer weights
    for (int i = 0; i < model->config.num_layers; i++) {
        char key[50];
        torch::Tensor layer_weights, layer_biases, attention_weights;
        
        snprintf(key, sizeof(key), "layer_weights_%d", i);
        archive.read(key, layer_weights);
        cudaMemcpy(model->layer_weights[i], layer_weights.data_ptr(), 
            layer_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        
        snprintf(key, sizeof(key), "layer_biases_%d", i);
        archive.read(key, layer_biases);
        cudaMemcpy(model->layer_biases[i], layer_biases.data_ptr(), 
            layer_biases.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
        
        snprintf(key, sizeof(key), "attention_weights_%d", i);
        archive.read(key, attention_weights);
        cudaMemcpy(model->attention_weights[i], attention_weights.data_ptr(), 
            attention_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Load output block weights
    torch::Tensor value_weights, value_bias, policy_weights, policy_bias;
    archive.read("value_weights", value_weights);
    archive.read("value_bias", value_bias);
    archive.read("policy_weights", policy_weights);
    archive.read("policy_bias", policy_bias);
    cudaMemcpy(model->value_weights, value_weights.data_ptr(), 
        value_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(model->value_bias, value_bias.data_ptr(), 
        value_bias.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(model->policy_weights, policy_weights.data_ptr(), 
        policy_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(model->policy_bias, policy_bias.data_ptr(), 
        policy_bias.numel() * sizeof(float), cudaMemcpyDeviceToDevice);

    // Load optimizer state
    torch::serialize::OutputArchive optimizer_archive;
    archive.read("optimizer", optimizer_archive);
    model->optimizer->load_state_dict(optimizer_archive);

    // Recreate cuDNN descriptors
    cudnnSetTensor4dDescriptor(model->input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, model->config.max_nodes, model->config.input_features);

    for (int i = 0; i < model->config.num_layers; i++) {
        cudnnSetTensor4dDescriptor(model->layer_descriptors[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   model->config.batch_size, model->config.num_heads, model->config.max_nodes, model->config.hidden_features);
    }

    cudnnSetTensor4dDescriptor(model->value_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.hidden_features);
    cudnnSetTensor4dDescriptor(model->policy_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.num_actions);
}

static void gat_destroy(INeuralNet* self) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Free input block resources
    cudnnDestroyTensorDescriptor(model->input_descriptor);
    cudaFree(model->input_weights);
    cudaFree(model->input_bias);

    // Free GAT layer resources
    for (int i = 0; i < model->config.num_layers; i++) {
        cudnnDestroyTensorDescriptor(model->layer_descriptors[i]);
        cudaFree(model->layer_weights[i]);
        cudaFree(model->layer_biases[i]);
        cudaFree(model->attention_weights[i]);
    }
    free(model->layer_descriptors);
    free(model->layer_weights);
    free(model->layer_biases);
    free(model->attention_weights);

    // Free output block resources
    cudnnDestroyTensorDescriptor(model->value_descriptor);
    cudnnDestroyTensorDescriptor(model->policy_descriptor);
    cudaFree(model->value_weights);
    cudaFree(model->value_bias);
    cudaFree(model->policy_weights);
    cudaFree(model->policy_bias);

    // Free cuDNN workspace
    cudaFree(model->workspace);

    // Destroy cuDNN handle
    cudnnDestroy(model->cudnn_handle);

    // Delete PyTorch optimizer
    delete model->optimizer;

    // Free the wrapper itself
    free(wrapper);
}

/*************************************************************************************************************************************************************
 * INIT HELPER FUNCTIONS
**************************************************************************************************************************************************************/


static void init_model_config(GATModel* model, const IGame* game) {
    int board_size = game->get_board_size(game);
    model->config.input_features = board_size * board_size;  // Assuming square board
    model->config.hidden_features = 256;  // You can adjust this
    model->config.output_features = 256;  // You can adjust this
    model->config.num_heads = 8;  // Typical value, can be adjusted
    model->config.num_layers = 3;  // You can adjust this
    model->config.num_actions = game->get_action_size(game);
    model->config.max_nodes = board_size * board_size;
    model->config.max_edges = model->config.max_nodes * model->config.max_nodes;  // Fully connected graph
    model->config.learning_rate = 0.001f;
    model->config.weight_decay = 0.0001f;
    model->config.dropout = 0.1f;
    model->config.alpha = 0.2f;  // LeakyReLU angle
    model->config.batch_size = 64;  // You can adjust this
    model->config.epochs = 10;  // You can adjust this
}

static void init_input_block(GATModel* model) {
    cudnnCreateTensorDescriptor(&model->input_descriptor);
    cudnnSetTensor4dDescriptor(model->input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, model->config.max_nodes, model->config.input_features);

    cudaMalloc(&model->input_weights, sizeof(float) * model->config.input_features * model->config.hidden_features);
    cudaMalloc(&model->input_bias, sizeof(float) * model->config.hidden_features);
}

static void init_gat_layers(GATModel* model) {
    model->layer_descriptors = (cudnnTensorDescriptor_t*)malloc(model->config.num_layers * sizeof(cudnnTensorDescriptor_t));
    model->layer_weights = (float**)malloc(model->config.num_layers * sizeof(float*));
    model->layer_biases = (float**)malloc(model->config.num_layers * sizeof(float*));
    model->attention_weights = (float**)malloc(model->config.num_layers * sizeof(float*));

    for (int i = 0; i < model->config.num_layers; i++) {
        cudnnCreateTensorDescriptor(&model->layer_descriptors[i]);
        cudnnSetTensor4dDescriptor(model->layer_descriptors[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   model->config.batch_size, model->config.num_heads, model->config.max_nodes, model->config.hidden_features);

        cudaMalloc(&model->layer_weights[i], sizeof(float) * model->config.hidden_features * model->config.hidden_features);
        cudaMalloc(&model->layer_biases[i], sizeof(float) * model->config.hidden_features);
        cudaMalloc(&model->attention_weights[i], sizeof(float) * model->config.num_heads * 2 * model->config.hidden_features);
    }
}

static void init_output_block(GATModel* model) {
    cudnnCreateTensorDescriptor(&model->value_descriptor);
    cudnnCreateTensorDescriptor(&model->policy_descriptor);

    cudnnSetTensor4dDescriptor(model->value_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.hidden_features);
    cudnnSetTensor4dDescriptor(model->policy_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.num_actions);

    cudaMalloc(&model->value_weights, sizeof(float) * model->config.hidden_features);
    cudaMalloc(&model->value_bias, sizeof(float));
    cudaMalloc(&model->policy_weights, sizeof(float) * model->config.hidden_features * model->config.num_actions);
    cudaMalloc(&model->policy_bias, sizeof(float) * model->config.num_actions);
}

static void init_weights(GATModel* model) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    // Initialize input weights
    curandGenerateNormal(gen, model->input_weights, model->config.input_features * model->config.hidden_features, 0, 0.1);
    curandGenerateNormal(gen, model->input_bias, model->config.hidden_features, 0, 0.1);

    // Initialize GAT layer weights
    for (int i = 0; i < model->config.num_layers; i++) {
        curandGenerateNormal(gen, model->layer_weights[i], model->config.hidden_features * model->config.hidden_features, 0, 0.1);
        curandGenerateNormal(gen, model->layer_biases[i], model->config.hidden_features, 0, 0.1);
        curandGenerateNormal(gen, model->attention_weights[i], model->config.num_heads * 2 * model->config.hidden_features, 0, 0.1);
    }

    // Initialize output weights
    curandGenerateNormal(gen, model->value_weights, model->config.hidden_features, 0, 0.1);
    curandGenerateNormal(gen, model->value_bias, 1, 0, 0.1);
    curandGenerateNormal(gen, model->policy_weights, model->config.hidden_features * model->config.num_actions, 0, 0.1);
    curandGenerateNormal(gen, model->policy_bias, model->config.num_actions, 0, 0.1);

    curandDestroyGenerator(gen);
}