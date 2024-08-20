#ifndef SELF_PLAY_CUH
#define SELF_PLAY_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../games/game.cuh"
#include "../networks/neural_network.h"
#include "../mcts/mcts.cuh"

#define MAX_BATCH_SIZE 1024
#define MAX_NUM_GAMES 10000
#define MAX_GAME_LENGTH 1000  // Maximum number of moves in a game
#define TERMINAL_STATE -1  // Special value to indicate end of game in board state

// CUDA error checking macro
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
    TrainingExample* d_examples;
} SelfPlayPipeline;

// Function prototypes

// Initialization and cleanup
SelfPlayPipeline* create_self_play_pipeline(IGame* game, INeuralNet* nnet, SelfPlayConfig config);
void destroy_self_play_pipeline(SelfPlayPipeline* pipeline);

// Main self-play and learning functions
void execute_self_play(SelfPlayPipeline* pipeline);
void learn(SelfPlayPipeline* pipeline);

// CUDA kernel function prototypes
__global__ void init_rng(curandState* states, unsigned long seed, int num_states);
__global__ void parallel_self_play_kernel(
    MCTSNode* roots, int* boards, float* pis, float* vs, int* players,
    curandState* rng_states, IGame* game, INeuralNet* nnet,
    int num_games, int num_mcts_sims, int temp_threshold, TrainingExample* examples
);

// Helper functions
void add_to_training_history(SelfPlayPipeline* pipeline, TrainingExample* examples, int num_examples);
void save_train_examples(SelfPlayPipeline* pipeline, int iteration);
void load_train_examples(SelfPlayPipeline* pipeline);
int pit_against_previous_version(SelfPlayPipeline* pipeline);
void get_checkpoint_file(SelfPlayPipeline* pipeline, int iteration, char* filename);

// MCTS and action selection helpers
__device__ void mcts_simulate(MCTSNode* node, int* board, int player, curandState* rng_state, IGame* game, INeuralNet* nnet);
__device__ void mcts_get_policy(MCTSNode* node, float* policy, float temperature);
__device__ int select_action(float* policy, int action_size, curandState* rng_state);
__device__ MCTSNode* mcts_move_to_child(MCTSNode* node, int action);
__device__ void mcts_expand(MCTSNode* node, int* board, int player, IGame* game);
__device__ MCTSNode* mcts_select_uct(MCTSNode* node);



#endif // SELF_PLAY_CUH