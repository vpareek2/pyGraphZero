#include "gat.cuh"

#include <curand.h>
#include <chrono>
#include <torch/serialize.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
static void gat_init(INeuralNet* self, const IGame* game) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Initialize model configuration
    init_model_config(model, game);

    // Initialize cuDNN
    CUDNN_CHECK(cudnnCreate(&model->cudnn_handle));

    // Initialize input block
    CUDA_CHECK(init_input_block(model));

    // Initialize GAT layers
    CUDA_CHECK(init_gat_layers(model));

    // Initialize output block
    CUDA_CHECK(init_output_block(model));

    // Initialize weights
    CUDA_CHECK(init_weights(model));

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
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(model->cudnn_handle, model->input_descriptor, model->layer_descriptors[0],
        /* convolution descriptor */, model->layer_descriptors[0],
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_size));
    CUDA_CHECK(cudaMalloc(&model->workspace, workspace_size));
    model->workspace_size = workspace_size;
}

static void gat_train(INeuralNet* self, TrainingExample* examples, int num_examples) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Allocate memory for batch data
    float *batch_boards, *batch_pis, *batch_vs;
    CUDA_CHECK(cudaMalloc(&batch_boards, model->config.batch_size * MAX_NODES * MAX_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch_pis, model->config.batch_size * model->config.num_actions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch_vs, model->config.batch_size * sizeof(float)));

    // Allocate memory for output
    float *out_pi, *out_v;
    CUDA_CHECK(cudaMalloc(&out_pi, model->config.batch_size * model->config.num_actions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_v, model->config.batch_size * sizeof(float)));

    for (int epoch = 0; epoch < model->config.epochs; epoch++) {
        printf("EPOCH ::: %d\n", epoch + 1);

        float pi_loss_sum = 0.0f;
        float v_loss_sum = 0.0f;
        int batch_count = num_examples / model->config.batch_size;

        for (int batch = 0; batch < batch_count; batch++) {
            // Prepare batch data
            prepare_batch(examples, num_examples, model->config.batch_size, batch_boards, batch_pis, batch_vs);

            // Forward pass
            auto start_time = std::chrono::high_resolution_clock::now();
            forward_gat(model, batch_boards, &out_pi, &out_v);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            printf("Forward pass took %lld microseconds\n", duration.count());

            // Compute losses
            auto [pi_loss, v_loss] = compute_losses(batch_pis, batch_vs, out_pi, out_v, 
                                                    model->config.batch_size, model->config.num_actions);
            pi_loss_sum += pi_loss;
            v_loss_sum += v_loss;

            // Backward pass
            start_time = std::chrono::high_resolution_clock::now();
            backward_gat(model, batch_boards, batch_pis, batch_vs, out_pi, out_v);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            printf("Backward pass took %lld microseconds\n", duration.count());

            // Update weights using Adam optimizer
            start_time = std::chrono::high_resolution_clock::now();
            model->optimizer->step();
            model->optimizer->zero_grad();
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            printf("Optimizer step took %lld microseconds\n", duration.count());
        }

        // Print epoch results
        printf("Average Policy Loss: %f, Average Value Loss: %f\n", 
               pi_loss_sum / batch_count, v_loss_sum / batch_count);
    }

    // Clean up
    CUDA_CHECK(cudaFree(batch_boards));
    CUDA_CHECK(cudaFree(batch_pis));
    CUDA_CHECK(cudaFree(batch_vs));
    CUDA_CHECK(cudaFree(out_pi));
    CUDA_CHECK(cudaFree(out_v));
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

/*************************************************************************************************************************************************************
 * CUDA ERROR CHECKING
**************************************************************************************************************************************************************/

#define CUDNN_CHECK(call) { cudnnStatus_t status = call; if (status != CUDNN_STATUS_SUCCESS) { fprintf(stderr, "CUDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); exit(1); } }
#define CUDA_CHECK(call) { cudaError_t status = call; if (status != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); exit(1); } }
#define CURAND_CHECK(call) { curandStatus_t status = call; if (status != CURAND_STATUS_SUCCESS) { fprintf(stderr, "CURAND error at %s:%d: %d\n", __FILE__, __LINE__, status); exit(1); } }

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
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->input_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, model->config.max_nodes, model->config.input_features));

    CUDA_CHECK(cudaMalloc(&model->input_weights, sizeof(float) * model->config.input_features * model->config.hidden_features));
    CUDA_CHECK(cudaMalloc(&model->input_bias, sizeof(float) * model->config.hidden_features));
}

static void init_gat_layers(GATModel* model) {
    model->layer_descriptors = (cudnnTensorDescriptor_t*)malloc(model->config.num_layers * sizeof(cudnnTensorDescriptor_t));
    model->layer_weights = (float**)malloc(model->config.num_layers * sizeof(float*));
    model->layer_biases = (float**)malloc(model->config.num_layers * sizeof(float*));
    model->attention_weights = (float**)malloc(model->config.num_layers * sizeof(float*));

    if (!model->layer_descriptors || !model->layer_weights || !model->layer_biases || !model->attention_weights) {
        fprintf(stderr, "Memory allocation failed in init_gat_layers\n");
        exit(1);
    }

    for (int i = 0; i < model->config.num_layers; i++) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->layer_descriptors[i]));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->layer_descriptors[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   model->config.batch_size, model->config.num_heads, model->config.max_nodes, model->config.hidden_features));

        CUDA_CHECK(cudaMalloc(&model->layer_weights[i], sizeof(float) * model->config.hidden_features * model->config.hidden_features));
        CUDA_CHECK(cudaMalloc(&model->layer_biases[i], sizeof(float) * model->config.hidden_features));
        CUDA_CHECK(cudaMalloc(&model->attention_weights[i], sizeof(float) * model->config.num_heads * 2 * model->config.hidden_features));
    }
}

static void init_output_block(GATModel* model) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->value_descriptor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->policy_descriptor));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->value_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.hidden_features));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->policy_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.num_actions));

    CUDA_CHECK(cudaMalloc(&model->value_weights, sizeof(float) * model->config.hidden_features));
    CUDA_CHECK(cudaMalloc(&model->value_bias, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->policy_weights, sizeof(float) * model->config.hidden_features * model->config.num_actions));
    CUDA_CHECK(cudaMalloc(&model->policy_bias, sizeof(float) * model->config.num_actions));
}

static void init_weights(GATModel* model) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    // Initialize input weights
    CURAND_CHECK(curandGenerateNormal(gen, model->input_weights, model->config.input_features * model->config.hidden_features, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->input_bias, model->config.hidden_features, 0, 0.1));

    // Initialize GAT layer weights
    for (int i = 0; i < model->config.num_layers; i++) {
        CURAND_CHECK(curandGenerateNormal(gen, model->layer_weights[i], model->config.hidden_features * model->config.hidden_features, 0, 0.1));
        CURAND_CHECK(curandGenerateNormal(gen, model->layer_biases[i], model->config.hidden_features, 0, 0.1));
        CURAND_CHECK(curandGenerateNormal(gen, model->attention_weights[i], model->config.num_heads * 2 * model->config.hidden_features, 0, 0.1));
    }

    // Initialize output weights
    CURAND_CHECK(curandGenerateNormal(gen, model->value_weights, model->config.hidden_features, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->value_bias, 1, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->policy_weights, model->config.hidden_features * model->config.num_actions, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->policy_bias, model->config.num_actions, 0, 0.1));

    CURAND_CHECK(curandDestroyGenerator(gen));
}

/*************************************************************************************************************************************************************
 * TRAIN HELPER FUNCTIONS
**************************************************************************************************************************************************************/

// Helper function to prepare batch data
static void prepare_batch(TrainingExample* examples, int num_examples, int batch_size,
                          float* batch_boards, float* batch_pis, float* batch_vs) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use cuRAND for random selection
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, batch_size * sizeof(int)));
    CURAND_CHECK(curandGenerate(gen, (unsigned int*)d_indices, batch_size));

    // Custom CUDA kernel to prepare batch
    dim3 grid((batch_size + 255) / 256);
    dim3 block(256);
    prepare_batch_kernel<<<grid, block>>>(examples, num_examples, d_indices, batch_boards, batch_pis, batch_vs, batch_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_indices));
    CURAND_CHECK(curandDestroyGenerator(gen));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Batch preparation took %lld microseconds\n", duration.count());
}

// CUDA kernel for batch preparation
__global__ void prepare_batch_kernel(TrainingExample* examples, int num_examples, int* indices,
                                     float* batch_boards, float* batch_pis, float* batch_vs, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int example_idx = indices[idx] % num_examples;
        memcpy(batch_boards + idx * MAX_NODES * MAX_FEATURES, examples[example_idx].board, MAX_NODES * MAX_FEATURES * sizeof(float));
        memcpy(batch_pis + idx * NUM_ACTIONS, examples[example_idx].pi, NUM_ACTIONS * sizeof(float));
        batch_vs[idx] = examples[example_idx].v;
    }
}

// Helper function for forward pass
static void forward_gat(GATModel* model, float* batch_boards, float** out_pi, float** out_v) {
    auto start_time = std::chrono::high_resolution_clock::now();

    cudnnHandle_t cudnn = model->cudnn_handle;
    cublasHandle_t cublas;
    CUDA_CHECK(cublasCreate(&cublas));

    float alpha = 1.0f, beta = 0.0f;
    
    // Input layer
    CUDA_CHECK(cudnnConvolutionForward(cudnn, &alpha, model->input_descriptor, batch_boards,
                            model->input_filter, model->input_weights,
                            model->conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                            model->workspace, model->workspace_size, &beta, model->layer_descriptors[0], model->layer_outputs[0]));

    // GAT layers
    for (int i = 0; i < model->config.num_layers; i++) {
        // Attention mechanism (custom CUDA kernel)
        dim3 grid((model->config.max_nodes + 255) / 256, model->config.num_heads);
        dim3 block(256);
        compute_attention<<<grid, block>>>(model->layer_outputs[i], model->attention_weights[i], 
                                           model->attention_scores[i], model->config.max_nodes, 
                                           model->config.hidden_features, model->config.num_heads);
        CUDA_CHECK(cudaGetLastError());

        // Apply attention (custom CUDA kernel)
        apply_attention<<<grid, block>>>(model->layer_outputs[i], model->attention_scores[i], 
                                         model->layer_outputs[i+1], model->config.max_nodes, 
                                         model->config.hidden_features, model->config.num_heads);
        CUDA_CHECK(cudaGetLastError());

        // Linear transformation
        CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    model->config.hidden_features, model->config.max_nodes, model->config.hidden_features,
                    &alpha, model->layer_weights[i], model->config.hidden_features,
                    model->layer_outputs[i+1], model->config.hidden_features,
                    &beta, model->layer_outputs[i+1], model->config.hidden_features));

        // Add bias and apply activation (custom CUDA kernel)
        add_bias_activate<<<grid, block>>>(model->layer_outputs[i+1], model->layer_biases[i], 
                                           model->config.max_nodes, model->config.hidden_features);
        CUDA_CHECK(cudaGetLastError());
    }

    // Output layer
    CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                model->config.num_actions, model->config.batch_size, model->config.hidden_features,
                &alpha, model->policy_weights, model->config.num_actions,
                model->layer_outputs[model->config.num_layers], model->config.hidden_features,
                &beta, *out_pi, model->config.num_actions));

    CUDA_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                1, model->config.hidden_features,
                &alpha, model->value_weights, 1,
                model->layer_outputs[model->config.num_layers], 1,
                &beta, *out_v, 1));

    // Apply softmax to policy output (custom CUDA kernel)
    dim3 grid_policy((model->config.num_actions + 255) / 256, model->config.batch_size);
    dim3 block_policy(256);
    softmax<<<grid_policy, block_policy>>>(*out_pi, model->config.num_actions);
    CUDA_CHECK(cudaGetLastError());

    // Apply tanh to value output (custom CUDA kernel)
    dim3 grid_value((model->config.batch_size + 255) / 256);
    dim3 block_value(256);
    tanh_activate<<<grid_value, block_value>>>(*out_v, model->config.batch_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cublasDestroy(cublas));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Forward pass took %lld microseconds\n", duration.count());
}

// Helper function to compute losses
static std::pair<float, float> compute_losses(float* target_pi, float* target_v, float* out_pi, float* out_v, int batch_size, int action_size) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use PyTorch for loss computation
    auto target_pi_tensor = torch::from_blob(target_pi, {batch_size, action_size}, torch::kFloat32);
    auto target_v_tensor = torch::from_blob(target_v, {batch_size}, torch::kFloat32);
    auto out_pi_tensor = torch::from_blob(out_pi, {batch_size, action_size}, torch::kFloat32);
    auto out_v_tensor = torch::from_blob(out_v, {batch_size}, torch::kFloat32);

    auto pi_loss = torch::nn::functional::kl_div(out_pi_tensor.log(), target_pi_tensor, torch::Reduction::Sum);
    auto v_loss = torch::mse_loss(out_v_tensor, target_v_tensor, torch::Reduction::Sum);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Loss computation took %lld microseconds\n", duration.count());

    return {pi_loss.item<float>() / batch_size, v_loss.item<float>() / batch_size};
}

static void backward_gat(GATModel* model, float* batch_boards, float* target_pi, float* target_v, float* out_pi, float* out_v) {
    auto start_time = std::chrono::high_resolution_clock::now();

    cudnnHandle_t cudnn = model->cudnn_handle;
    cublasHandle_t cublas;
    CUDA_CHECK(cublasCreate(&cublas));

    float alpha = 1.0f, beta = 0.0f;

    // Compute gradients for policy and value heads
    float* d_policy, *d_value;
    CUDA_CHECK(cudaMalloc(&d_policy, model->config.batch_size * model->config.num_actions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_value, model->config.batch_size * sizeof(float)));

    // Policy loss gradient
    softmax_cross_entropy_gradient<<<(model->config.num_actions + 255) / 256, 256>>>(
        out_pi, target_pi, d_policy, model->config.batch_size, model->config.num_actions);
    CUDA_CHECK(cudaGetLastError());

    // Value loss gradient
    mse_gradient<<<(model->config.batch_size + 255) / 256, 256>>>(
        out_v, target_v, d_value, model->config.batch_size);
    CUDA_CHECK(cudaGetLastError());

    // Backpropagate through output layer
    CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                model->config.hidden_features, model->config.batch_size, model->config.num_actions,
                &alpha, model->policy_weights, model->config.num_actions,
                d_policy, model->config.num_actions,
                &beta, model->d_last_layer, model->config.hidden_features));

    CUDA_CHECK(cublasSger(cublas, model->config.hidden_features, model->config.batch_size,
               &alpha, model->value_weights, 1,
               d_value, 1,
               model->d_last_layer, model->config.hidden_features));

    // Backpropagate through GAT layers
    for (int i = model->config.num_layers - 1; i >= 0; i--) {
        // Backpropagate through attention mechanism
        backward_attention<<<(model->config.max_nodes + 255) / 256, 256>>>(
            model->layer_outputs[i], model->attention_scores[i], model->d_last_layer,
            model->d_attention_weights[i], model->d_layer_outputs[i],
            model->config.max_nodes, model->config.hidden_features, model->config.num_heads);
        CUDA_CHECK(cudaGetLastError());

        // Backpropagate through linear transformation
        CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    model->config.hidden_features, model->config.hidden_features, model->config.max_nodes,
                    &alpha, model->d_layer_outputs[i], model->config.hidden_features,
                    model->layer_weights[i], model->config.hidden_features,
                    &beta, model->d_last_layer, model->config.hidden_features));

        CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    model->config.hidden_features, model->config.max_nodes, model->config.hidden_features,
                    &alpha, model->layer_weights[i], model->config.hidden_features,
                    model->d_layer_outputs[i], model->config.hidden_features,
                    &beta, model->d_layer_weights[i], model->config.hidden_features));

        // Compute gradient for biases
        CUDA_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                    model->config.hidden_features, model->config.max_nodes,
                    &alpha, model->d_layer_outputs[i], model->config.hidden_features,
                    model->ones, 1,
                    &beta, model->d_layer_biases[i], 1));

        // Backpropagate activation function
        backward_activation<<<(model->config.max_nodes * model->config.hidden_features + 255) / 256, 256>>>(
            model->layer_outputs[i], model->d_last_layer,
            model->config.max_nodes * model->config.hidden_features);
        CUDA_CHECK(cudaGetLastError());
    }

    // Backpropagate through input layer
    CUDA_CHECK(cudnnConvolutionBackwardFilter(cudnn, &alpha,
                                   model->input_descriptor, batch_boards,
                                   model->layer_descriptors[0], model->d_last_layer,
                                   model->conv_descriptor, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                   model->workspace, model->workspace_size,
                                   &beta, model->input_filter, model->d_input_weights));

    CUDA_CHECK(cudnnConvolutionBackwardData(cudnn, &alpha,
                                 model->input_filter, model->input_weights,
                                 model->layer_descriptors[0], model->d_last_layer,
                                 model->conv_descriptor, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                 model->workspace, model->workspace_size,
                                 &beta, model->input_descriptor, model->d_input_data));

    // Clean up
    CUDA_CHECK(cudaFree(d_policy));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cublasDestroy(cublas));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Backward pass took %lld microseconds\n", duration.count());
}

// Custom CUDA kernel implementations

__global__ void compute_attention(float* inputs, float* weights, float* scores, int num_nodes, int hidden_size, int num_heads) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;

    if (node_idx < num_nodes) {
        for (int i = 0; i < num_nodes; i++) {
            float score = 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                int input_idx = node_idx * hidden_size + j;
                int weight_idx = head_idx * 2 * hidden_size + j;
                score += inputs[input_idx] * weights[weight_idx];
                score += inputs[i * hidden_size + j] * weights[weight_idx + hidden_size];
            }
            scores[head_idx * num_nodes * num_nodes + node_idx * num_nodes + i] = score;
        }
    }
}

__global__ void apply_attention(float* inputs, float* scores, float* outputs, int num_nodes, int hidden_size, int num_heads) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int feature_idx = threadIdx.y;

    if (node_idx < num_nodes && feature_idx < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < num_nodes; i++) {
            int score_idx = head_idx * num_nodes * num_nodes + node_idx * num_nodes + i;
            int input_idx = i * hidden_size + feature_idx;
            sum += scores[score_idx] * inputs[input_idx];
        }
        outputs[head_idx * num_nodes * hidden_size + node_idx * hidden_size + feature_idx] = sum;
    }
}

__global__ void add_bias_activate(float* outputs, float* biases, int num_nodes, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes * hidden_size) {
        int feature_idx = idx % hidden_size;
        float x = outputs[idx] + biases[feature_idx];
        outputs[idx] = x > 0.0f ? x : 0.2f * x;  // LeakyReLU activation with alpha = 0.2
    }
}

__global__ void softmax(float* inputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float max_val = inputs[idx * size];
        for (int i = 1; i < size; i++) {
            max_val = max(max_val, inputs[idx * size + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            inputs[idx * size + i] = exp(inputs[idx * size + i] - max_val);
            sum += inputs[idx * size + i];
        }
        
        for (int i = 0; i < size; i++) {
            inputs[idx * size + i] /= sum;
        }
    }
}

__global__ void tanh_activate(float* inputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        inputs[idx] = tanh(inputs[idx]);
    }
}

__global__ void softmax_cross_entropy_gradient(float* output, float* target, float* gradient, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        gradient[idx] = output[idx] - target[idx];
    }
}

__global__ void mse_gradient(float* output, float* target, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradient[idx] = 2 * (output[idx] - target[idx]);
    }
}

__global__ void backward_attention(float* inputs, float* scores, float* grad_output, float* grad_weights, float* grad_inputs, int num_nodes, int hidden_size, int num_heads) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;

    if (node_idx < num_nodes) {
        for (int i = 0; i < num_nodes; i++) {
            float grad_score = 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                int grad_idx = head_idx * num_nodes * hidden_size + i * hidden_size + j;
                int input_idx = node_idx * hidden_size + j;
                grad_score += grad_output[grad_idx] * inputs[input_idx];
                atomicAdd(&grad_inputs[input_idx], grad_output[grad_idx] * scores[head_idx * num_nodes * num_nodes + i * num_nodes + node_idx]);
            }
            atomicAdd(&grad_weights[head_idx * 2 * hidden_size + node_idx], grad_score);
            atomicAdd(&grad_weights[head_idx * 2 * hidden_size + hidden_size + i], grad_score);
        }
    }
}

__global__ void backward_activation(float* inputs, float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] *= (inputs[idx] > 0.0f) ? 1.0f : 0.2f;  // LeakyReLU gradient
    }
}