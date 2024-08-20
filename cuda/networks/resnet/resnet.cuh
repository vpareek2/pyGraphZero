#ifndef GAT_CUH
#define GAT_CUH

#include "../neural_network.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <torch/torch.h>
#include <curand.h>

// Model configuration
typedef struct {
    int input_features;
    int hidden_features;
    int output_features;
    int num_heads;
    int num_layers;
    int num_actions;
    int max_nodes;
    int max_edges;
    float learning_rate;
    float weight_decay;
    float dropout;
    float alpha;  // LeakyReLU angle
    int batch_size;
    int epochs;
    int board_size;  // Added based on usage in gat.cu
} ModelConfig;

// GAT model
typedef struct {
    // Input block
    cudnnTensorDescriptor_t input_descriptor;
    float *input_weights, *input_bias;
    cudnnFilterDescriptor_t input_filter;  // Added based on usage in gat.cu

    // GAT layers
    cudnnTensorDescriptor_t *layer_descriptors;
    float **layer_weights, **layer_biases;
    float **attention_weights;
    float **layer_outputs;  // Added based on usage in gat.cu
    float **attention_scores;  // Added based on usage in gat.cu

    // Output block
    cudnnTensorDescriptor_t value_descriptor, policy_descriptor;
    float *value_weights, *value_bias;
    float *policy_weights, *policy_bias;

    // CUDNN handles
    cudnnHandle_t cudnn_handle;

    // Model configuration
    ModelConfig config;

    // PyTorch optimizer
    torch::optim::Adam* optimizer;

    // Gradient fields
    float *d_input_weights;
    float **d_layer_weights, **d_attention_weights;
    float *d_value_weights, *d_policy_weights;
    float *d_last_layer;  // Added based on usage in gat.cu
    float **d_layer_outputs;  // Added based on usage in gat.cu
    float **d_layer_biases;  // Added based on usage in gat.cu

    // Workspace for cuDNN
    void *workspace;
    size_t workspace_size;

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;  // Added based on usage in gat.cu

    // Additional fields
    float *ones;  // Added based on usage in gat.cu
} GATModel;

typedef struct {
    INeuralNet base;
    GATModel model;
} GATWrapper;

// Function prototypes
INeuralNet* create_gat_model(const IGame* game);
static void gat_init(INeuralNet* self, const IGame* game);
static void gat_train(INeuralNet* self, TrainingExample* examples, int num_examples);
static void gat_predict(INeuralNet* self, const float* board, float* pi, float* v);
static void gat_save_checkpoint(INeuralNet* self, const char* folder, const char* filename);
static void gat_load_checkpoint(INeuralNet* self, const char* folder, const char* filename);
static void gat_destroy(INeuralNet* self);

// Helper function prototypes
static void init_model_config(GATModel* model, const IGame* game);
static void init_input_block(GATModel* model);
static void init_gat_layers(GATModel* model);
static void init_output_block(GATModel* model);
static void init_weights(GATModel* model);
static void prepare_batch(TrainingExample* examples, int num_examples, int batch_size,
                          float* batch_boards, float* batch_pis, float* batch_vs);
static void forward_gat(GATModel* model, float* batch_boards, float** out_pi, float** out_v);
static std::pair<float, float> compute_losses(float* target_pi, float* target_v, float* out_pi, float* out_v, int batch_size, int action_size);
static void backward_gat(GATModel* model, float* batch_boards, float* target_pi, float* target_v, float* out_pi, float* out_v);

// CUDA kernel declarations
__global__ void prepare_batch_kernel(TrainingExample* examples, int num_examples, int* indices,
                                     float* batch_boards, float* batch_pis, float* batch_vs, int batch_size);
__global__ void compute_attention(float* inputs, float* weights, float* scores, int num_nodes, int hidden_size, int num_heads);
__global__ void apply_attention(float* inputs, float* scores, float* outputs, int num_nodes, int hidden_size, int num_heads);
__global__ void add_bias_activate(float* outputs, float* biases, int num_nodes, int hidden_size);
__global__ void softmax(float* inputs, int size);
__global__ void tanh_activate(float* inputs, int size);
__global__ void softmax_cross_entropy_gradient(float* output, float* target, float* gradient, int batch_size, int num_classes);
__global__ void mse_gradient(float* output, float* target, float* gradient, int size);
__global__ void backward_attention(float* inputs, float* scores, float* grad_output, float* grad_weights, float* grad_inputs, int num_nodes, int hidden_size, int num_heads);
__global__ void backward_activation(float* inputs, float* grad_output, int size);

#endif // GAT_CUH