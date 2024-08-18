#ifndef MCTS_CUH
#define MCTS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../games/tictactoe/tictactoe.h"

#define BOARD_SIZE 3
#define NUM_SQUARES (BOARD_SIZE * BOARD_SIZE)

// Node structure for the MCTS tree
typedef struct Node {
    int board[NUM_SQUARES];
    struct Node** children;
    int num_children;
    struct Node* parent;
    int visits;
    float score;
    int player;
} Node;

// Function declarations for MCTS
Node* create_node(const int board[NUM_SQUARES], int player, Node* parent);
void free_node(Node* node);
Node* select_node(Node* root, const TicTacToeGame* game);
void expand_node(Node* node, const TicTacToeGame* game);
float simulate(const TicTacToeGame* game, int board[NUM_SQUARES], int player);
void backpropagate(Node* node, float score);
Node* mcts_search(const TicTacToeGame* game, int board[NUM_SQUARES], int player, int num_iterations);

// CUDA kernel declarations
__global__ void init_rand_kernel(curandState* states, unsigned long seed);
__global__ void mcts_simulation_kernel(curandState* states, Node* nodes, int num_nodes, TicTacToeGame* game, float* results);

// CUDA-related function declarations
void cuda_mcts_simulate(Node** leaf_nodes, int num_leaves, const TicTacToeGame* game, float* scores);

#endif // MCTS_CUH