#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "games/connect4/connect4.cuh"
#include "games/tictactoe/tictactoe.cuh"
#include "networks/gat/gat.cuh"
#include "self_play/self_play.cuh"
#include "utils/cuda_utils.cuh"

#define CUDA_CHECK(call) { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(1); \
    } \
}

int main(int argc, char* argv[]) {
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));

    // Parse command-line arguments
    const char* game_type = "tictactoe";
    if (argc > 1) {
        game_type = argv[1];
    }

    // Create game instance
    IGame* game = NULL;
    if (strcmp(game_type, "tictactoe") == 0) {
        game = (IGame*)create_tictactoe_game();
    } else if (strcmp(game_type, "connect4") == 0) {
        game = (IGame*)create_connect4_game();
    } else {
        fprintf(stderr, "Unsupported game type: %s\n", game_type);
        return 1;
    }

    if (!game) {
        fprintf(stderr, "Failed to create game instance\n");
        return 1;
    }

    // Create neural network instance
    INeuralNet* nnet = create_gat_model(game);
    if (!nnet) {
        fprintf(stderr, "Failed to create GAT neural network instance\n");
        game->destroy(game);
        return 1;
    }

    // Initialize self-play configuration
    SelfPlayConfig config = {
        .numIters = 1000,
        .numEps = 100,
        .numGames = 100,
        .tempThreshold = 15,
        .updateThreshold = 0.6f,
        .maxlenOfQueue = 200000,
        .numMCTSSims = 25,
        .arenaCompare = 40,
        .cpuct = 1.0f,
        .checkpoint = "./checkpoints",
        .load_model = false,
        .numItersForTrainExamplesHistory = 20
    };

    // Create self-play pipeline
    SelfPlayPipeline* pipeline = create_self_play_pipeline(game, nnet, config);
    if (!pipeline) {
        fprintf(stderr, "Failed to create self-play pipeline\n");
        nnet->destroy(nnet);
        game->destroy(game);
        return 1;
    }

    // Main training loop
    for (int i = 1; i <= config.numIters; i++) {
        printf("Starting iteration %d\n", i);

        // Execute self-play
        execute_self_play(pipeline);

        // Train neural network
        learn(pipeline);

        // Optionally save checkpoint
        if (i % 10 == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "checkpoint_%04d.pth", i);
            nnet->save_checkpoint(nnet, config.checkpoint, filename);
            printf("Saved checkpoint: %s\n", filename);
        }
    }

    // Clean up
    destroy_self_play_pipeline(pipeline);
    nnet->destroy(nnet);
    game->destroy(game);

    // Reset CUDA device
    CUDA_CHECK(cudaDeviceReset());

    printf("Training completed successfully\n");
    return 0;
}