#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "../games/game.h"

#define MAX_FILENAME_LENGTH 256

typedef struct INeuralNet INeuralNet;

typedef struct {
    float* board;
    float* pi;
    float v;
} TrainingExample;

struct INeuralNet {
    void* impl;  // Pointer to the specific implementation (e.g., ResNet)

    void (*init)(INeuralNet* self, const IGame* game);
    void (*train)(INeuralNet* self, TrainingExample* examples, int num_examples);
    void (*predict)(INeuralNet* self, const float* board, float* pi, float* v);
    void (*save_checkpoint)(INeuralNet* self, const char* folder, const char* filename);
    void (*load_checkpoint)(INeuralNet* self, const char* folder, const char* filename);
    void (*destroy)(INeuralNet* self);
};

// Factory function to create a specific neural network implementation
INeuralNet* create_neural_net(const char* net_type, const IGame* game);

#endif // NEURAL_NETWORK_H