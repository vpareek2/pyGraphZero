#include "self_play.cuh"

#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

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
    int* init_board = (int*)malloc(MAX_BOARD_SIZE * sizeof(int));
    if (!init_board) {
        fprintf(stderr, "Failed to allocate memory for init_board\n");
        return;
    }
    pipeline->game->get_init_board(pipeline->game, init_board);
    CUDA_CHECK(cudaMemcpy(pipeline->d_boards, init_board, numGames * MAX_BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    free(init_board);

    // Initialize game states, MCTS nodes, etc.
    CUDA_CHECK(cudaMalloc(&pipeline->d_game_states, numGames * sizeof(GameState)));
    CUDA_CHECK(cudaMalloc(&pipeline->d_mcts_roots, numGames * sizeof(MCTSNode)));

    // Launch parallel self-play kernel
    dim3 grid((numGames + 255) / 256);
    dim3 block(256);
    
    parallel_self_play_kernel<<<grid, block>>>(
        pipeline->d_mcts_roots,
        pipeline->d_boards,
        pipeline->d_pis,
        pipeline->d_vs,
        pipeline->d_rng_states,
        pipeline->game,
        pipeline->nnet,
        numGames,
        numMCTSSims,
        tempThreshold
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate host memory for results
    TrainingExample* examples = (TrainingExample*)malloc(numGames * MAX_GAME_LENGTH * sizeof(TrainingExample));
    if (!examples) {
        fprintf(stderr, "Failed to allocate memory for examples\n");
        return;
    }

    // Copy results back to CPU
    CUDA_CHECK(cudaMemcpy(examples, pipeline->d_examples, numGames * MAX_GAME_LENGTH * sizeof(TrainingExample), cudaMemcpyDeviceToHost));

    // Process and store examples
    int totalExamples = 0;
    for (int i = 0; i < numGames; i++) {
        for (int j = 0; j < MAX_GAME_LENGTH; j++) {
            if (examples[i * MAX_GAME_LENGTH + j].board[0] == TERMINAL_STATE) {
                break;
            }
            totalExamples++;
        }
    }

    // Add examples to the training history
    add_to_training_history(pipeline, examples, totalExamples);

    // Clean up
    free(examples);
    CUDA_CHECK(cudaFree(pipeline->d_game_states));
    CUDA_CHECK(cudaFree(pipeline->d_mcts_roots));
}

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// In self_play.cu
__global__ void self_play_kernel(IGame* game, GATModel* model, MCTSNode* roots, TrainingExample* examples, int num_games, int num_mcts_sims, float cpuct, curandState* rng_states) {
    int game_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (game_idx >= num_games) return;

    curandState* rng_state = &rng_states[game_idx];
    MCTSNode* root = &roots[game_idx];
    TrainingExample* game_examples = &examples[game_idx * MAX_GAME_LENGTH];
    int example_count = 0;

    int board[MAX_BOARD_SIZE];
    game->get_init_board(game, board);
    int player = 1;

    while (true) {
        // Perform MCTS simulations
        for (int i = 0; i < num_mcts_sims; i++) {
            mcts_simulate(root, board, player, rng_state, game, model, cpuct);
        }

        // Compute policy from visit counts
        float policy[MAX_BOARD_SIZE];
        mcts_get_policy(root, policy, 1.0f); // Temperature 1.0 for exploration

        // Store the current state as a training example
        memcpy(game_examples[example_count].board, board, sizeof(board));
        memcpy(game_examples[example_count].pi, policy, sizeof(policy));
        example_count++;

        // Select action
        int action = select_action(policy, game->get_action_size(game), rng_state);

        // Apply action
        int next_board[MAX_BOARD_SIZE];
        int next_player;
        game->get_next_state_cuda(game, board, player, action, next_board, &next_player);

        // Check if game has ended
        float reward = game->get_game_ended_cuda(game, next_board, next_player);
        if (reward != 0) {
            // Game has ended, update all examples with the reward
            for (int i = 0; i < example_count; i++) {
                game_examples[i].v = reward * (i % 2 == 0 ? 1 : -1);
            }
            break;
        }

        // Move to next state
        memcpy(board, next_board, sizeof(board));
        player = next_player;
        root = mcts_move_to_child(root, action);
    }
}

// Implement other helper functions as needed