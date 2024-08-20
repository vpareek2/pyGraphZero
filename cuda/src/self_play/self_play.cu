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

    // Initialize d_examples to nullptr
    pipeline->d_examples = nullptr;

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

    // Free memory allocated for d_examples
    if (pipeline->d_examples) {
        CUDA_CHECK(cudaFree(pipeline->d_examples));
    }

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

    // Allocate memory for examples if not already allocated
    if (pipeline->d_examples == nullptr) {
        CUDA_CHECK(cudaMalloc(&pipeline->d_examples, numGames * MAX_GAME_LENGTH * sizeof(TrainingExample)));
    }

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
        tempThreshold,
        pipeline->d_examples
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
    int num_games, int num_mcts_sims, int temp_threshold,
    TrainingExample* examples
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

        // Store the current state as a training example
        TrainingExample* example = &examples[game_idx * MAX_GAME_LENGTH + moves];
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
                TrainingExample* ex = &examples[game_idx * MAX_GAME_LENGTH + i];
                ex->v = reward * ((i % 2 == 0) ? 1 : -1);
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
                TrainingExample* ex = &examples[game_idx * MAX_GAME_LENGTH + i];
                ex->v = 0.0f;  // Draw
            }
            vs[game_idx] = 0.0f;
            break;
        }
    }

    players[game_idx] = player;  // Update final player state
}

// Helper functions

void add_to_training_history(SelfPlayPipeline* pipeline, TrainingExample* examples, int num_examples) {
    // If the history is full, remove the oldest entry
    if (pipeline->historySize >= pipeline->config.numItersForTrainExamplesHistory) {
        free(pipeline->trainExamplesHistory[0]);
        memmove(pipeline->trainExamplesHistory, pipeline->trainExamplesHistory + 1,
                (pipeline->historySize - 1) * sizeof(TrainingExample*));
        pipeline->historySize--;
    }

    // Allocate memory for the new examples
    TrainingExample* new_examples = (TrainingExample*)malloc(num_examples * sizeof(TrainingExample));
    if (new_examples == nullptr) {
        fprintf(stderr, "Failed to allocate memory for new training examples\n");
        return;
    }

    // Copy the examples
    memcpy(new_examples, examples, num_examples * sizeof(TrainingExample));

    // Add the new examples to the history
    pipeline->trainExamplesHistory[pipeline->historySize] = new_examples;
    pipeline->historySize++;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void save_train_examples(SelfPlayPipeline* pipeline, int iteration) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/train_examples_%d.bin", pipeline->config.checkpoint, iteration);

    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        return;
    }

    // Write the number of iterations in the history
    fwrite(&pipeline->historySize, sizeof(int), 1, file);

    // Write each iteration's examples
    for (int i = 0; i < pipeline->historySize; i++) {
        // Write the number of examples in this iteration
        int num_examples = pipeline->config.numEps * MAX_GAME_LENGTH;
        fwrite(&num_examples, sizeof(int), 1, file);

        // Write the examples
        fwrite(pipeline->trainExamplesHistory[i], sizeof(TrainingExample), num_examples, file);
    }

    fclose(file);
    printf("Training examples saved to %s\n", filename);
}

void load_train_examples(SelfPlayPipeline* pipeline) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/train_examples_latest.bin", pipeline->config.checkpoint);

    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s for reading\n", filename);
        return;
    }

    // Read the number of iterations in the history
    int loaded_history_size;
    fread(&loaded_history_size, sizeof(int), 1, file);

    // Clear existing history
    for (int i = 0; i < pipeline->historySize; i++) {
        free(pipeline->trainExamplesHistory[i]);
    }
    pipeline->historySize = 0;

    // Read each iteration's examples
    for (int i = 0; i < loaded_history_size; i++) {
        // Read the number of examples in this iteration
        int num_examples;
        fread(&num_examples, sizeof(int), 1, file);

        // Allocate memory for the examples
        TrainingExample* examples = (TrainingExample*)malloc(num_examples * sizeof(TrainingExample));
        if (examples == NULL) {
            fprintf(stderr, "Error: Unable to allocate memory for loading examples\n");
            fclose(file);
            return;
        }

        // Read the examples
        fread(examples, sizeof(TrainingExample), num_examples, file);

        // Add to history
        pipeline->trainExamplesHistory[pipeline->historySize] = examples;
        pipeline->historySize++;
    }

    fclose(file);
    printf("Training examples loaded from %s\n", filename);
}

// MCTS and action selection helpers

__device__ void mcts_simulate(MCTSNode* node, int* board, int player, curandState* rng_state, IGame* game, INeuralNet* nnet) {
    if (game->get_game_ended_cuda(game, board, player) != 0) {
        // Game has ended, backpropagate the result
        mcts_backpropagate(node, -game->get_game_ended_cuda(game, board, player));
        return;
    }

    if (node->num_children == 0) {
        // Expand the node
        mcts_expand(node, board, player, game);

        // Evaluate the position using the neural network
        float value;
        float policy[MAX_BOARD_SIZE];
        nnet->predict(nnet, board, policy, &value);

        // Update node with the evaluation results
        for (int i = 0; i < node->num_children; i++) {
            node->P[i] = policy[node->children[i]->action];
        }

        // Backpropagate the value
        mcts_backpropagate(node, value);
    } else {
        // Select the best child according to the UCT formula
        MCTSNode* best_child = mcts_select_uct(node);

        // Recursively simulate from the best child
        int next_board[MAX_BOARD_SIZE];
        int next_player;
        game->get_next_state_cuda(game, board, player, best_child->action, next_board, &next_player);
        mcts_simulate(best_child, next_board, next_player, rng_state, game, nnet);
    }
}

__device__ void mcts_get_policy(MCTSNode* node, float* policy, float temperature) {
    int action_size = node->num_children;
    float sum = 0.0f;

    for (int i = 0; i < action_size; i++) {
        if (temperature == 0.0f) {
            policy[i] = (i == argmax(node->N, action_size)) ? 1.0f : 0.0f;
        } else {
            policy[i] = __powf(node->N[i], 1.0f / temperature);
        }
        sum += policy[i];
    }

    // Normalize the policy
    for (int i = 0; i < action_size; i++) {
        policy[i] /= sum;
    }
}

// Helper function to find the index of the maximum value
__device__ int argmax(float* arr, int size) {
    int max_idx = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

__device__ int select_action(float* policy, int action_size, curandState* rng_state) {
    float r = curand_uniform(rng_state);
    float sum = 0.0f;
    for (int i = 0; i < action_size; i++) {
        sum += policy[i];
        if (r <= sum) {
            return i;
        }
    }
    // In case of rounding errors, return the last action
    return action_size - 1;
}

__device__ MCTSNode* mcts_move_to_child(MCTSNode* node, int action) {
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]->action == action) {
            return node->children[i];
        }
    }
    // This should never happen if the action is valid
    return nullptr;
}