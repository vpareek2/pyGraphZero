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
    cudaMalloc(&pipeline->d_rng_states, config.numGames * sizeof(curandState));
    cudaMalloc(&pipeline->d_boards, config.numGames * MAX_BOARD_SIZE * sizeof(int));
    cudaMalloc(&pipeline->d_pis, config.numGames * MAX_BOARD_SIZE * sizeof(float));
    cudaMalloc(&pipeline->d_vs, config.numGames * sizeof(float));
    cudaMalloc(&pipeline->d_mcts_nodes, config.numGames * sizeof(MCTSNode));

    // Initialize RNG states
    init_rng<<<(config.numGames + 255) / 256, 256>>>(pipeline->d_rng_states, time(NULL));

    // Allocate CPU resources for training examples history
    pipeline->trainExamplesHistory = (TrainingExample**)malloc(config.numItersForTrainExamplesHistory * sizeof(TrainingExample*));
    pipeline->historySize = 0;
    pipeline->skipFirstSelfPlay = false;

    return pipeline;
}

void destroy_self_play_pipeline(SelfPlayPipeline* pipeline) {
    if (!pipeline) return;

    mcts_free(pipeline->mcts);

    cudaFree(pipeline->d_rng_states);
    cudaFree(pipeline->d_boards);
    cudaFree(pipeline->d_pis);
    cudaFree(pipeline->d_vs);
    cudaFree(pipeline->d_mcts_nodes);

    for (int i = 0; i < pipeline->historySize; i++) {
        free(pipeline->trainExamplesHistory[i]);
    }
    free(pipeline->trainExamplesHistory);

    free(pipeline);
}

void execute_self_play(SelfPlayPipeline* pipeline) {
    int numGames = pipeline->config.numGames;
    int numMCTSSims = pipeline->config.numMCTSSims;

    // Initialize boards on GPU
    int* init_board = (int*)malloc(MAX_BOARD_SIZE * sizeof(int));
    pipeline->game->get_init_board(pipeline->game, init_board);
    cudaMemcpy(pipeline->d_boards, init_board, numGames * MAX_BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    free(init_board);

    // Launch parallel self-play kernel
    dim3 grid((numGames + 255) / 256);
    dim3 block(256);
    parallel_self_play_kernel<<<grid, block>>>(
        pipeline->d_mcts_nodes, 
        pipeline->d_boards, 
        pipeline->d_pis, 
        pipeline->d_vs, 
        pipeline->d_rng_states, 
        pipeline->game, 
        numGames, 
        numMCTSSims
    );
    cudaDeviceSynchronize();

    // Copy results back to CPU and process
    TrainingExample* examples = (TrainingExample*)malloc(numGames * sizeof(TrainingExample));
    cudaMemcpy(examples, pipeline->d_boards, numGames * sizeof(TrainingExample), cudaMemcpyDeviceToHost);

    // Process and store examples
    // (You'll need to implement this part based on your specific requirements)

    free(examples);
}

__global__ void init_rng(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void parallel_self_play_kernel(MCTSNode* nodes, int* boards, float* pis, float* vs, curandState* rng_states, IGame* game, int num_games, int num_mcts_sims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_games) return;

    MCTSNode* root = &nodes[idx];
    int* board = &boards[idx * MAX_BOARD_SIZE];
    float* pi = &pis[idx * MAX_BOARD_SIZE];
    curandState* rng_state = &rng_states[idx];

    // Perform MCTS simulations
    for (int i = 0; i < num_mcts_sims; i++) {
        mcts_simulate(root, board, 1, rng_state, game);
    }

    // Get action probabilities
    for (int i = 0; i < root->num_children; i++) {
        pi[root->children[i]->action] = (float)root->N[i] / root->visit_count;
    }

    // Select action and update board
    int action = mcts_select_action(root, 0); // temperature = 0 for deterministic selection
    game->get_next_state(game, board, 1, action, board, NULL);

    // Evaluate final state
    vs[idx] = mcts_evaluate(board, 1, game);
}

// Implement other helper functions as needed