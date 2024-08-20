#ifndef SELF_PLAY_CUH
#define SELF_PLAY_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../games/game.h"
#include "../networks/neural_network.h"
#include "../mcts/mcts.cuh"

#define MAX_BATCH_SIZE 1024
#define MAX_NUM_GAMES 10000

// To make sure that the cuda stuff doesnt throw an
#define CUDA_CHECK(call) { cudaError_t status = call; if (status != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); exit(1); } }

typedef struct {
    int numIters;
    int numEps;
    int numGames;
    int batchSize;
    int numMCTSSims;
    float tempThreshold;
    float updateThreshold;
    int maxlenOfQueue;
    int numItersForTrainExamplesHistory;
    int arenaCompare;
    char checkpoint[256];
} SelfPlayConfig;

typedef struct {
    int board[MAX_BOARD_SIZE];
    float pi[MAX_BOARD_SIZE];
    float v;
} TrainingExample;

typedef struct {
    IGame* game;
    INeuralNet* nnet;
    INeuralNet* pnet;
    MCTSState* mcts;
    SelfPlayConfig config;
    TrainingExample** trainExamplesHistory;
    int historySize;
    int skipFirstSelfPlay;

    // GPU resources
    curandState* d_rng_states;
    int* d_boards;
    float* d_pis;
    float* d_vs;
    MCTSNode* d_mcts_nodes;
} SelfPlayPipeline;

// Function prototypes

// Initialization and cleanup
SelfPlayPipeline* create_self_play_pipeline(IGame* game, INeuralNet* nnet, SelfPlayConfig config);
void destroy_self_play_pipeline(SelfPlayPipeline* pipeline);

// Main self-play and learning functions
void execute_self_play(SelfPlayPipeline* pipeline);
void learn(SelfPlayPipeline* pipeline);

// CUDA kernel function prototypes
__global__ void init_rng(curandState* states, unsigned long seed);
__global__ void parallel_self_play_kernel(MCTSNode* nodes, int* boards, float* pis, float* vs, curandState* rng_states, IGame* game, int num_games, int num_mcts_sims);

// Helper functions
void save_train_examples(SelfPlayPipeline* pipeline, int iteration);
void load_train_examples(SelfPlayPipeline* pipeline);
int pit_against_previous_version(SelfPlayPipeline* pipeline);
void get_checkpoint_file(SelfPlayPipeline* pipeline, int iteration, char* filename);

#endif // SELF_PLAY_CUH