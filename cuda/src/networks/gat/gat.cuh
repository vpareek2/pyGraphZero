#ifndef GAT_CUH
#define GAT_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// Define constants
#define MAX_NODES 10000
#define MAX_EDGES 100000

// Structure to represent a graph
typedef struct {
    int num_nodes;
    int num_edges;
    int num_features;
    float* node_features;  // [num_nodes x num_features]
    int* edge_index;       // [2 x num_edges]
} Graph;

// Structure to represent a GAT layer
typedef struct {
    int in_features;
    int out_features;
    int num_heads;
    float dropout;
    float alpha;  // LeakyReLU angle
    
    // Weights and biases
    float* W;  // [num_heads x in_features x out_features]
    float* a;  // [num_heads x 2 x out_features]
    
    // Intermediate results
    float* attention_scores;     // [num_edges x num_heads]
    float* attention_probs;      // [num_edges x num_heads]
    float* node_features_proj;   // [num_nodes x num_heads x out_features]
    
    // CUDA handles
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
} GATLayer;

// Structure to represent the entire GAT model
typedef struct {
    int num_layers;
    GATLayer* layers;
    
    // CUDA memory
    void* d_workspace;
    size_t workspace_size;
} GATModel;

// Function prototypes

// Initialization
__host__ void initGraph(Graph* graph, int num_nodes, int num_edges, int num_features);
__host__ void initGATLayer(GATLayer* layer, int in_features, int out_features, int num_heads, float dropout, float alpha);
__host__ void initGATModel(GATModel* model, int num_layers, int* layer_sizes, int num_heads, float dropout, float alpha);

// Forward pass
__global__ void computeNodeFeatureProjection(GATLayer* layer, Graph* graph);
__global__ void computeAttentionScores(GATLayer* layer, Graph* graph);
__global__ void applyAttention(GATLayer* layer, Graph* graph);
__host__ void forwardGATLayer(GATLayer* layer, Graph* graph);
__host__ void forwardGATModel(GATModel* model, Graph* graph);

// Backward pass
__global__ void backpropAttention(GATLayer* layer, Graph* graph, float* grad_output);
__global__ void backpropNodeFeatures(GATLayer* layer, Graph* graph, float* grad_output);
__host__ void backwardGATLayer(GATLayer* layer, Graph* graph, float* grad_output);
__host__ void backwardGATModel(GATModel* model, Graph* graph, float* grad_output);

// Utility functions
__host__ void updateWeights(GATModel* model, float learning_rate);
__device__ float leakyReLU(float x, float alpha);

// Memory management
__host__ void freeGraph(Graph* graph);
__host__ void freeGATLayer(GATLayer* layer);
__host__ void freeGATModel(GATModel* model);

#endif // GAT_CUH