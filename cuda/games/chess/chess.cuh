#ifndef CHESS_CUH
#define CHESS_CUH

#include "../game.h"

#define CHESS_BOARD_SIZE 8
#define CHESS_NUM_SQUARES (CHESS_BOARD_SIZE * CHESS_BOARD_SIZE)

// Piece representations
#define EMPTY 0
#define PAWN 1
#define KNIGHT 2
#define BISHOP 3
#define ROOK 4
#define QUEEN 5
#define KING 6

// Colors
#define WHITE 1
#define BLACK -1

typedef struct {
    IGame base;
    // You can add chess-specific members here if needed
} ChessGame;

__host__ __device__ void chess_init(IGame* self);
__host__ __device__ void chess_get_init_board(const IGame* self, int* board);
__host__ __device__ void chess_get_board_size(const IGame* self, int* rows, int* cols);
__host__ __device__ int chess_get_action_size(const IGame* self);
__host__ __device__ void chess_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);
__host__ __device__ void chess_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves);
__host__ __device__ int chess_get_game_ended(const IGame* self, const int* board, int player);
__host__ __device__ void chess_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board);
__host__ __device__ float chess_evaluate(const IGame* self, const int* board, int player);

// CPU-only functions
void chess_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries);
void chess_string_representation(const IGame* self, const int* board, char* str, int str_size);
void chess_display(const IGame* self, const int* board);

// Create a Chess game instance
ChessGame* create_chess_game();

// Helper functions (you may need to implement these)
__host__ __device__ bool is_in_check(const int* board, int player);
__host__ __device__ bool is_checkmate(const int* board, int player);
__host__ __device__ bool is_stalemate(const int* board, int player);
__host__ __device__ bool is_insufficient_material(const int* board);

#endif // CHESS_CUH