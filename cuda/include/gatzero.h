#ifndef GATZERO_H
#define GATZERO_H
// In train.cu

// void train_alphazero(IGame* game, GATModel* model, TrainingConfig config) {
//     // Allocate memory for self-play results
//     TrainingExample* examples;
//     cudaMallocManaged(&examples, config.num_games * MAX_GAME_LENGTH * sizeof(TrainingExample));

//     for (int iteration = 0; iteration < config.num_iterations; iteration++) {
//         // Perform self-play
//         self_play<<<(config.num_games + 255) / 256, 256>>>(game, model, examples, config.num_games, config.num_mcts_sims, config.cpuct);
//         cudaDeviceSynchronize();

//         // Train neural network
//         gat_train(&model->base, examples, config.num_games * MAX_GAME_LENGTH);

//         // Evaluate new network
//         if (evaluate_network(game, model) > config.replacement_threshold) {
//             // Update network for next iteration
//             update_network(model);
//         }
//     }

//     cudaFree(examples);
// }
#endif // GATZERO_H