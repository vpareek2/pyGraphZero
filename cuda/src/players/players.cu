#include "players.cuh"
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ int random_action(const IGame* game, const int* board, int player, curandState* state) {
    int action_size = game->get_action_size(game);
    bool valid_moves[MAX_BOARD_SIZE];
    game->get_valid_moves_cuda(game, board, player, valid_moves);

    int valid_actions[MAX_BOARD_SIZE];
    int num_valid = 0;
    for (int i = 0; i < action_size; i++) {
        if (valid_moves[i]) {
            valid_actions[num_valid++] = i;
        }
    }

    if (num_valid == 0) return -1;  // No valid moves

    int random_index = curand(state) % num_valid;
    return valid_actions[random_index];
}

__global__ void random_action_kernel(const IGame* game, const int* board, int player, int* result) {
    curandState state;
    curand_init(clock64(), threadIdx.x, 0, &state);
    *result = random_action(game, board, player, &state);
}

int random_player_get_action(const IGame* game, const int* board, int player) {
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));

    random_action_kernel<<<1, 1>>>(game, board, player, d_result);
    
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_result);
    return result;
}

RandomPlayer* create_random_player() {
    RandomPlayer* player = (RandomPlayer*)malloc(sizeof(RandomPlayer));
    player->base.get_action = random_player_get_action;
    return player;
}

int mcts_player_get_action(const IGame* game, const int* board, int player, MCTSState* mcts_state, float temperature) {
    // Update the MCTS state with the current board
    mcts_update_with_move(mcts_state, -1);  // -1 to reset to root

    // Run MCTS simulations
    for (int i = 0; i < mcts_state->num_simulations; i++) {
        mcts_search(mcts_state);
    }

    // Select action based on visit counts and temperature
    return mcts_select_action(mcts_state, temperature);
}

MCTSPlayer* create_mcts_player(IGame* game, int num_simulations, float temperature) {
    MCTSPlayer* player = (MCTSPlayer*)malloc(sizeof(MCTSPlayer));
    player->base.get_action = (int (*)(const IGame*, const int*, int))mcts_player_get_action;
    player->mcts_state = mcts_init(game);
    player->mcts_state->num_simulations = num_simulations;
    player->temperature = temperature;
    return player;
}

int nnet_player_get_action(const IGame* game, const int* board, int player, INeuralNet* net, MCTSState* mcts_state, float temperature) {
    // Update the MCTS state with the current board
    mcts_update_with_move(mcts_state, -1);  // -1 to reset to root

    // Run MCTS simulations using the neural network
    for (int i = 0; i < mcts_state->num_simulations; i++) {
        mcts_search_with_nn(mcts_state, net);
    }

    // Select action based on visit counts and temperature
    return mcts_select_action(mcts_state, temperature);
}

NNetPlayer* create_nnet_player(IGame* game, INeuralNet* net, int num_simulations, float temperature) {
    NNetPlayer* player = (NNetPlayer*)malloc(sizeof(NNetPlayer));
    player->base.get_action = (int (*)(const IGame*, const int*, int))nnet_player_get_action;
    player->net = net;
    player->mcts_state = mcts_init(game);
    player->mcts_state->num_simulations = num_simulations;
    player->temperature = temperature;
    return player;
}

void destroy_random_player(RandomPlayer* player) {
    free(player);
}

void destroy_mcts_player(MCTSPlayer* player) {
    mcts_free(player->mcts_state);
    free(player);
}

void destroy_nnet_player(NNetPlayer* player) {
    mcts_free(player->mcts_state);
    // Note: We don't free the neural network here, as it might be shared among multiple players
    free(player);
}