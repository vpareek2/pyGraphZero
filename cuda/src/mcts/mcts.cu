#include "mcts.cuh"

#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

// CPU functions implementation

MCTSState* mcts_init(IGame* game) {
    MCTSState* state = new MCTSState;
    state->game = game;
    state->root = new MCTSNode;
    
    // Initialize root node
    memset(state->root->board, 0, sizeof(int) * MAX_BOARD_SIZE);
    game->get_init_board(game, state->root->board);
    state->root->num_children = 0;
    state->root->visit_count = 0;
    state->root->value_sum = 0;
    state->root->player = 1;  // Assume player 1 starts
    state->root->action = -1;  // Root has no action

    return state;
}

void mcts_free(MCTSState* state) {
    // Recursive function to free the MCTS tree
    std::function<void(MCTSNode*)> free_node = [&](MCTSNode* node) {
        for (int i = 0; i < node->num_children; i++) {
            free_node(node->children[i]);
        }
        delete node;
    };

    free_node(state->root);
    delete state;
}

int mcts_select_action(MCTSState* state, float temperature) {
    // Perform simulations
    mcts_run_simulations(state);

    // Select action based on visit counts and temperature
    float sum = 0;
    float probs[MAX_CHILDREN];
    for (int i = 0; i < state->root->num_children; i++) {
        if (temperature == 0) {
            probs[i] = (i == std::max_element(state->root->N, state->root->N + state->root->num_children) - state->root->N);
        } else {
            probs[i] = std::pow(state->root->N[i], 1 / temperature);
        }
        sum += probs[i];
    }

    // Normalize probabilities
    for (int i = 0; i < state->root->num_children; i++) {
        probs[i] /= sum;
    }

    // Choose action based on probabilities
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probs, probs + state->root->num_children);
    return d(gen);
}

void mcts_update_with_move(MCTSState* state, int action) {
    // Find the child node corresponding to the action
    MCTSNode* child = nullptr;
    for (int i = 0; i < state->root->num_children; i++) {
        if (state->root->children[i]->action == action) {
            child = state->root->children[i];
            break;
        }
    }

    if (child != nullptr) {
        // Make the child the new root
        MCTSNode* old_root = state->root;
        state->root = child;
        child->parent = nullptr;

        // Delete all other children of the old root
        for (int i = 0; i < old_root->num_children; i++) {
            if (old_root->children[i] != child) {
                mcts_free_node(old_root->children[i]);
            }
        }
        delete old_root;
    } else {
        // If the move wasn't in the tree, create a new root node
        MCTSNode* new_root = new MCTSNode;
        state->game->get_next_state(state->game, state->root->board, state->root->player, action, new_root->board, &new_root->player);
        new_root->action = action;
        new_root->num_children = 0;
        new_root->visit_count = 0;
        new_root->value_sum = 0;

        mcts_free_node(state->root);
        state->root = new_root;
    }
}

// CPU helper functions

void mcts_get_valid_moves(const IGame* game, const int* board, int player, bool* valid_moves) {
    game->get_valid_moves(game, board, player, valid_moves);
}

void mcts_apply_move(const IGame* game, int* board, int player, int action) {
    int next_player;
    game->get_next_state(game, board, player, action, board, &next_player);
}

// CUDA kernel functions

__global__ void mcts_simulate_kernel(MCTSNode* nodes, int* board, int player, curandState* rng_states, IGame* game) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < NUM_SIMULATIONS) {
        float value = mcts_simulate(&nodes[0], board, player, &rng_states[tid], game);
        mcts_backpropagate(&nodes[0], value);
    }
}

__device__ float mcts_simulate(MCTSNode* node, int* board, int player, curandState* rng_state, IGame* game) {
    if (game->get_game_ended(game, board, player) != 0) {
        return -game->get_game_ended(game, board, player);
    }

    if (node->num_children == 0) {
        mcts_expand(node, board, player, game);
        return -mcts_evaluate(board, player, game);
    }

    MCTSNode* best_child = mcts_select(node);
    
    int next_board[MAX_BOARD_SIZE];
    int next_player;
    game->get_next_state(game, board, player, best_child->action, next_board, &next_player);
    
    float value = -mcts_simulate(best_child, next_board, next_player, rng_state, game);

    atomicAdd(&node->visit_count, 1);
    atomicAdd(&node->value_sum, value);

    return value;
}

__device__ MCTSNode* mcts_select(MCTSNode* node) {
    float best_value = -INFINITY;
    MCTSNode* best_child = nullptr;

    for (int i = 0; i < node->num_children; i++) {
        MCTSNode* child = &node->children[i];
        float q_value = child->value_sum / (child->visit_count + 1e-8);
        float u_value = C_PUCT * sqrtf(__logf(node->visit_count + 1) / (child->visit_count + 1e-8));
        float uct_value = q_value + u_value;

        if (uct_value > best_value) {
            best_value = uct_value;
            best_child = child;
        }
    }

    return best_child;
}

__device__ void mcts_expand(MCTSNode* node, int* board, int player, IGame* game) {
    bool valid_moves[MAX_BOARD_SIZE];
    game->get_valid_moves(game, board, player, valid_moves);

    int num_children = 0;
    for (int i = 0; i < game->get_action_size(game); i++) {
        if (valid_moves[i]) {
            MCTSNode* child = &node->children[num_children];
            child->action = i;
            child->player = -player;
            child->num_children = 0;
            child->visit_count = 0;
            child->value_sum = 0;
            num_children++;
        }
    }
    node->num_children = num_children;
}

__device__ float mcts_evaluate(int* board, int player, IGame* game) {
    return game->evaluate(game, board, player);
}

__device__ void mcts_backpropagate(MCTSNode* node, float value) {
    while (node != nullptr) {
        atomicAdd(&node->visit_count, 1);
        atomicAdd(&node->value_sum, value);
        value = -value;
        node = node->parent;
    }
}

// CUDA helper functions

__global__ void init_rng(curandState* states, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < NUM_SIMULATIONS) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Function to run GPU simulations
void mcts_run_simulations(MCTSState* state) {
    // Allocate GPU memory
    MCTSNode* d_nodes;
    int* d_board;
    curandState* d_rng_states;
    IGame* d_game;

    cudaMalloc(&d_nodes, sizeof(MCTSNode) * MAX_CHILDREN);  // Assuming worst case: all children are expanded
    cudaMalloc(&d_board, sizeof(int) * MAX_BOARD_SIZE);
    cudaMalloc(&d_rng_states, sizeof(curandState) * NUM_SIMULATIONS);
    cudaMalloc(&d_game, sizeof(IGame));

    // Copy data to GPU
    cudaMemcpy(d_nodes, state->root, sizeof(MCTSNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_board, state->root->board, sizeof(int) * MAX_BOARD_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_game, state->game, sizeof(IGame), cudaMemcpyHostToDevice);

    // Initialize RNG states
    init_rng<<<(NUM_SIMULATIONS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_rng_states, time(NULL));

    // Run simulations
    mcts_simulate_kernel<<<(NUM_SIMULATIONS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_nodes, d_board, state->root->player, d_rng_states, d_game);

    // Copy results back to CPU
    cudaMemcpy(state->root, d_nodes, sizeof(MCTSNode), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_nodes);
    cudaFree(d_board);
    cudaFree(d_rng_states);
    cudaFree(d_game);
}

// Helper function to free a node and its children
void mcts_free_node(MCTSNode* node) {
    for (int i = 0; i < node->num_children; i++) {
        mcts_free_node(node->children[i]);
    }
    delete node;
}