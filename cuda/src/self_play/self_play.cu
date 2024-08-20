#include "self_play.cuh"

#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

SelfPlayPipeline* create_self_play_pipeline(IGame* game, INeuralNet* nnet, SelfPlayConfig config) {
    SelfPlayPipeline* pipeline = (SelfPlayPipeline*)malloc(sizeof(SelfPlayPipeline));
    if (!pipeline) {
        fprintf(stderr, "Failed to allocate memory for SelfPlayPipeline\n");
        return NULL;
    }

    pipeline->game = game;
    pipeline->nnet = nnet;
    pipeline->config = config;

    // Initialize MCTS
    pipeline->mcts = mcts_init(game);
    if (!pipeline->mcts) {
        fprintf(stderr, "Failed to initialize MCTS\n");
        free(pipeline);
        return NULL;
    }

    // Allocate GPU resources
    CUDA_CHECK(cudaMalloc(&pipeline->d_rng_states, config.numGames * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&pipeline->d_boards, config.numGames * MAX_BOARD_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pipeline->d_pis, config.numGames * MAX_BOARD_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pipeline->d_vs, config.numGames * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pipeline->d_mcts_nodes, config.numGames * sizeof(MCTSNode)));

    // Initialize RNG states
    init_rng<<<(config.numGames + 255) / 256, 256>>>(pipeline->d_rng_states, time(NULL));
    CUDA_CHECK(cudaGetLastError());

    // Allocate CPU resources for training examples history
    pipeline->trainExamplesHistory = (TrainingExample**)malloc(config.numItersForTrainExamplesHistory * sizeof(TrainingExample*));
    if (!pipeline->trainExamplesHistory) {
        fprintf(stderr, "Failed to allocate memory for trainExamplesHistory\n");
        destroy_self_play_pipeline(pipeline);
        return NULL;
    }
    pipeline->historySize = 0;
    pipeline->skipFirstSelfPlay = false;

    return pipeline;
}

void destroy_self_play_pipeline(SelfPlayPipeline* pipeline) {
    if (!pipeline) return;

    mcts_free(pipeline->mcts);

    CUDA_CHECK(cudaFree(pipeline->d_rng_states));
    CUDA_CHECK(cudaFree(pipeline->d_boards));
    CUDA_CHECK(cudaFree(pipeline->d_pis));
    CUDA_CHECK(cudaFree(pipeline->d_vs));
    CUDA_CHECK(cudaFree(pipeline->d_mcts_nodes));

    for (int i = 0; i < pipeline->historySize; i++) {
        free(pipeline->trainExamplesHistory[i]);
    }
    free(pipeline->trainExamplesHistory);

    free(pipeline);
}


void execute_self_play(SelfPlayPipeline* pipeline) {
    int numGames = pipeline->config.numGames;
    int numMCTSSims = pipeline->config.numMCTSSims;
    int tempThreshold = pipeline->config.tempThreshold;
    
    // Initialize boards on GPU
    thrust::host_vector<int> h_init_board(MAX_BOARD_SIZE);
    pipeline->game->get_init_board(pipeline->game, h_init_board.data());
    thrust::device_vector<int> d_boards(numGames * MAX_BOARD_SIZE, 0);
    for (int i = 0; i < numGames; ++i) {
        thrust::copy(h_init_board.begin(), h_init_board.end(), d_boards.begin() + i * MAX_BOARD_SIZE);
    }

    // Initialize MCTS nodes
    thrust::device_vector<MCTSNode> d_mcts_roots(numGames);

    // Initialize other necessary arrays
    thrust::device_vector<float> d_pis(numGames * MAX_BOARD_SIZE);
    thrust::device_vector<float> d_vs(numGames);
    thrust::device_vector<int> d_players(numGames, 1);  // Start with player 1 for all games

    // Launch parallel self-play kernel
    dim3 grid((numGames + 255) / 256, 1, 1);
    dim3 block(256, 1, 1);
    
    parallel_self_play_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_mcts_roots.data()),
        thrust::raw_pointer_cast(d_boards.data()),
        thrust::raw_pointer_cast(d_pis.data()),
        thrust::raw_pointer_cast(d_vs.data()),
        thrust::raw_pointer_cast(d_players.data()),
        pipeline->d_rng_states,
        pipeline->game,
        pipeline->nnet,
        numGames,
        numMCTSSims,
        tempThreshold
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to CPU and process
    thrust::host_vector<TrainingExample> h_examples(numGames * MAX_GAME_LENGTH);
    CUDA_CHECK(cudaMemcpy(h_examples.data(), pipeline->d_examples, numGames * MAX_GAME_LENGTH * sizeof(TrainingExample), cudaMemcpyDeviceToHost));

    // Process and store examples
    int totalExamples = 0;
    for (int i = 0; i < numGames; i++) {
        for (int j = 0; j < MAX_GAME_LENGTH; j++) {
            if (h_examples[i * MAX_GAME_LENGTH + j].board[0] == TERMINAL_STATE) {
                break;
            }
            totalExamples++;
        }
    }

    // Add examples to the training history
    add_to_training_history(pipeline, h_examples.data(), totalExamples);
}

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void parallel_self_play_kernel(
    MCTSNode* roots, int* boards, float* pis, float* vs, int* players,
    curandState* rng_states, IGame* game, INeuralNet* nnet,
    int num_games, int num_mcts_sims, int temp_threshold
) {
    int game_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (game_idx >= num_games) return;

    curandState* rng_state = &rng_states[game_idx];
    MCTSNode* root = &roots[game_idx];
    int* board = &boards[game_idx * MAX_BOARD_SIZE];
    float* pi = &pis[game_idx * MAX_BOARD_SIZE];
    int player = players[game_idx];
    int moves = 0;

    while (true) {
        // Perform MCTS simulations
        for (int i = 0; i < num_mcts_sims; i++) {
            mcts_simulate(root, board, player, rng_state, game, nnet);
        }

        // Compute policy from visit counts
        float temp = (moves < temp_threshold) ? 1.0f : 1e-3f;
        mcts_get_policy(root, pi, temp);

        // Store the current state as a training example (assuming d_examples is accessible)
        TrainingExample* example = &d_examples[game_idx * MAX_GAME_LENGTH + moves];
        memcpy(example->board, board, MAX_BOARD_SIZE * sizeof(int));
        memcpy(example->pi, pi, MAX_BOARD_SIZE * sizeof(float));

        // Select action
        int action = select_action(pi, game->get_action_size(game), rng_state);

        // Apply action
        int next_board[MAX_BOARD_SIZE];
        int next_player;
        game->get_next_state_cuda(game, board, player, action, next_board, &next_player);

        // Check if game has ended
        float reward = game->get_game_ended_cuda(game, next_board, next_player);
        if (reward != 0) {
            // Game has ended, update all examples with the reward
            for (int i = 0; i <= moves; i++) {
                TrainingExample* ex = &d_examples[game_idx * MAX_GAME_LENGTH + i];
                ex->v = reward * (i % 2 == 0 ? 1 : -1);
            }
            vs[game_idx] = reward;
            break;
        }

        // Move to next state
        memcpy(board, next_board, MAX_BOARD_SIZE * sizeof(int));
        player = next_player;
        root = mcts_move_to_child(root, action);
        moves++;

        if (moves >= MAX_GAME_LENGTH - 1) {
            // Force end of game if it's taking too long
            for (int i = 0; i <= moves; i++) {
                TrainingExample* ex = &d_examples[game_idx * MAX_GAME_LENGTH + i];
                ex->v = 0.0f;  // Draw
            }
            vs[game_idx] = 0.0f;
            break;
        }
    }

    players[game_idx] = player;  // Update final player state
}