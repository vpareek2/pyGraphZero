#ifndef GAT_CUH
#define GAT_CUH

#include "../neural_network.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// Model configuration
typedef struct {
    int input_features;
    int hidden_features;
    int output_features;
    int num_heads;
    int num_layers;
    float dropout;
    float alpha;  // LeakyReLU angle
    int max_nodes;
    int max_edges;
    float learning_rate;
    float weight_decay;
} GATConfig;

// GAT Layer
typedef struct {
    float* W;  // [num_heads x in_features x out_features]
    float* a;  // [num_heads x 2 x out_features]
    float* attention_scores;
    float* attention_probs;
    float* node_features_proj;
} GATLayer;

// GAT model
typedef struct {
    GATConfig config;
    GATLayer* layers;
    
    // CUDA handles
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    
    // CUDA memory
    void* d_workspace;
    size_t workspace_size;

    // PyTorch optimizer (if using PyTorch for optimization)
    void* optimizer;
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
static void init_gat_config(GATModel* model, const IGame* game);
static void init_gat_layers(GATModel* model);
static void init_weights(GATModel* model);
static void prepare_batch(TrainingExample* examples, int num_examples, int batch_size, float** batch_boards, float** batch_pis, float** batch_vs);
static void forward_gat(GATModel* model, float* batch_boards, float** out_pi, float** out_v);
static std::pair<float, float> compute_losses(float* target_pi, float* target_v, float* out_pi, float* out_v, int batch_size, int action_size);
static void backward_gat(GATModel* model, float* batch_boards, float* target_pi, float* target_v, float* out_pi, float* out_v);
static void adam_update(void* optimizer);

// CUDA kernel declarations
__global__ void compute_attention_scores(GATLayer* layer, float* node_features, int num_nodes, int num_edges);
__global__ void apply_attention(GATLayer* layer, float* node_features, int* edge_index, int num_nodes, int num_edges);

#endif // GAT_CUH