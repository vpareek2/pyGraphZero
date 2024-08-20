#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include <cuda_runtime.h>

#define MAX_BOARD_SIZE 64  // Suitable for games up to 8x8 (planning chess scale up)
#define MAX_SYMMETRIES 8   // Change later

typedef struct IGame IGame;

struct IGame {
    // CPU functions
    void (*init)(IGame* self);
    void (*get_init_board)(const IGame* self, int* board);
    void (*get_board_size)(const IGame* self, int* rows, int* cols);
    int (*get_action_size)(const IGame* self);
    void (*get_next_state)(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);
    void (*get_valid_moves)(const IGame* self, const int* board, int player, bool* valid_moves);
    int (*get_game_ended)(const IGame* self, const int* board, int player);
    void (*get_canonical_form)(const IGame* self, const int* board, int player, int* canonical_board);
    void (*get_symmetries)(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries);
    void (*string_representation)(const IGame* self, const int* board, char* str, int str_size);
    void (*display)(const IGame* self, const int* board);
    float (*evaluate)(const IGame* self, const int* board, int player);

    // CUDA-compatible functions
    __device__ void (*get_init_board_cuda)(const IGame* self, int* board);
    __device__ void (*get_next_state_cuda)(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);
    __device__ void (*get_valid_moves_cuda)(const IGame* self, const int* board, int player, bool* valid_moves);
    __device__ int (*get_game_ended_cuda)(const IGame* self, const int* board, int player);
    __device__ void (*get_canonical_form_cuda)(const IGame* self, const int* board, int player, int* canonical_board);
    __device__ float (*evaluate_cuda)(const IGame* self, const int* board, int player);
};

#endif // GAME_H