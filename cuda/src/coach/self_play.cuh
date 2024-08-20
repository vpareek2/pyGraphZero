#ifndef SELF_PLAY_CUH
#define SELF_PLAY_CUH

#include <cuda_runtime.h>
#include "game.h"
#include "neural_network.h"
#include "mcts.cuh"

#define MAX_HISTORY_SIZE 20
#define MAX_EPISODE_LENGTH 1000

typedef struct {
    int numIters;
    int numEps;
    int tempThreshold;
    float updateThreshold;
    int maxlenOfQueue;
    int numItersForTrainExamplesHistory;
    int arenaCompare;
    char checkpoint[MAX_FILENAME_LENGTH];
} TrainingConfig;

typedef struct {
    IGame* game;
    INeuralNet* nnet;
    INeuralNet* pnet;
    MCTSState* mcts;
    TrainingConfig config;
    TrainingExample** trainExamplesHistory;
    int historySize;
    int skipFirstSelfPlay;
} TrainingPipeline;

// Function prototypes
TrainingPipeline* create_training_pipeline(IGame* game, INeuralNet* nnet, TrainingConfig config);
void destroy_training_pipeline(TrainingPipeline* pipeline);

// Main training loop
void train(TrainingPipeline* pipeline);

// Helper functions
void execute_episode(TrainingPipeline* pipeline, TrainingExample** examples, int* num_examples);
void learn_iteration(TrainingPipeline* pipeline, int iteration);
int pit_against_previous_version(TrainingPipeline* pipeline);

// Utility functions
void save_train_examples(TrainingPipeline* pipeline, int iteration);
void load_train_examples(TrainingPipeline* pipeline);
void get_checkpoint_file(TrainingPipeline* pipeline, int iteration, char* filename);

// CUDA kernel function prototypes
__global__ void parallel_self_play(MCTSState* mcts_states, TrainingExample* examples, int num_episodes, int max_moves, curandState* rng_states);
__global__ void parallel_arena_play(MCTSState* mcts_states_1, MCTSState* mcts_states_2, int* results, int num_games, curandState* rng_states);

#endif // SELF_PLAY_CUH