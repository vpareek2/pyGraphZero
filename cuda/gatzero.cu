#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "games/connect4/connect4.cuh"
#include "games/tictactoe/tictactoe.cuh"
#include "networks/gat/gat.cuh"
#include "self_play/self_play.cuh"
#include "utils/cuda_utils.cuh"

int main(int argc, char* argv[]) {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // Create game instance
    TicTacToe* game = create_tictactoe_game();
    if (!game) {
        fprintf(stderr, "Failed to create TicTacToe game instance");
        return 1;
    }

    // Connect4Game* game = create_connect4_game();
    // if (!game) {
    //     fprintf(stderr, "Failed to create Connect4 game instance");
    //     return 1;
    // }

    // Create neural network instance
    INeuralNet* nnet = create_gat_model((IGame*)game);
    if (!nnet) {
        fprintf(stderr, "Failed to create GAT neural network instance");
        destroy_connect4_game(game);
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
    SelfPlayPipeline* pipeline = create_self_play_pipeline((IGame*)game, nnet, config);
    if (!pipeline) {
        fprintf(stderr, "Failed to create self-play pipeline");
        nnet->destroy(nnet);
        destroy_connect4_game(game);
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
        }
    }

    // Clean up
    destroy_self_play_pipeline(pipeline);
    nnet->destroy(nnet);
    destroy_connect4_game(game);

    // Reset CUDA device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}