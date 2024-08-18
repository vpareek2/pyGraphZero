#ifndef MCTS_CUH
#define MCTS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "game.h"

#define MAX_BOARD_SIZE 64  // 8x8 board for chess
#define MAX_CHILDREN 218   // Maximum possible moves in chess, most likely highest scale
#define C_PUCT 1.0         // This can be tuned based on performance
#define NUM_SIMULATIONS 1600
#define THREADS_PER_BLOCK 256

typedef struct MCTSNode {
    int board[MAX_BOARD_SIZE];
    MCTSNode* children[MAX_CHILDREN];
    int num_children;
    float P[MAX_CHILDREN];
    float Q[MAX_CHILDREN];
    int N[MAX_CHILDREN];
    int visit_count;
    float value_sum;
    int player;
    int action;
} MCTSNode;

typedef struct MCTSState {
    IGame* game;
    MCTSNode* root;
} MCTSState;

// CPU functions
MCTSState* mcts_init(IGame* game);
void mcts_free(MCTSState* state);
int mcts_select_action(MCTSState* state, float temperature);
void mcts_update_with_move(MCTSState* state, int action);

// CUDA kernel functions
__global__ void mcts_simulate_kernel(MCTSNode* nodes, int* board, int player, curandState* rng_states);
__device__ float mcts_simulate(MCTSNode* node, int* board, int player, curandState* rng_state);
__device__ MCTSNode* mcts_select(MCTSNode* node);
__device__ void mcts_expand(MCTSNode* node, int* board, int player, IGame* game);
__device__ float mcts_evaluate(int* board, int player, IGame* game);
__device__ void mcts_backpropagate(MCTSNode* node, float value);

// CPU helper functions
void mcts_get_valid_moves(const IGame* game, const int* board, int player, bool* valid_moves);
void mcts_apply_move(const IGame* game, int* board, int player, int action);

// CUDA helper functions
__global__ void init_rng(curandState* states, unsigned long seed);


#endif // MCTS_CUH