#include "chess.cuh"

// Initialize the chess game state
__host__ __device__ void chess_init(IGame* self);

// Set up the initial chess board configuration
__host__ __device__ void chess_get_init_board(const IGame* self, int* board);

// Return the dimensions of the chess board (8x8)
__host__ __device__ void chess_get_board_size(const IGame* self, int* rows, int* cols);

// Return the total number of possible moves in chess (e.g., 64 * 73 for all possible start and end squares including promotions)
__host__ __device__ int chess_get_action_size(const IGame* self);

// Apply a move to the current board state and return the new state
__host__ __device__ void chess_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);

// Determine all legal moves for the current player
__host__ __device__ void chess_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves);

// Check if the game has ended (checkmate, stalemate, or draw) and return the result
__host__ __device__ int chess_get_game_ended(const IGame* self, const int* board, int player);

// Convert the board to a canonical form (e.g., always from white's perspective)
__host__ __device__ void chess_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board);

// Heuristic evaluation of the board state (optional, can return 0 if not implemented)
__host__ __device__ float chess_evaluate(const IGame* self, const int* board, int player);

// Generate all symmetries of the board (rotations and reflections)
void chess_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries);

// Convert the board state to a string representation
void chess_string_representation(const IGame* self, const int* board, char* str, int str_size);

// Display the current board state (e.g., print to console)
void chess_display(const IGame* self, const int* board);

// Create a new chess game instance
ChessGame* create_chess_game();

// Free resources associated with a chess game instance
void destroy_chess_game(ChessGame* game);

/***************************************************
 * HELPERS
 **************************************************/

// Check if the given player is in check
__host__ __device__ bool is_check(const ChessBoard* board, int player);

// Check if the given player is in checkmate
__host__ __device__ bool is_checkmate(const ChessBoard* board, int player);

// Check if the given player is in stalemate
__host__ __device__ bool is_stalemate(const ChessBoard* board, int player);

// Check if there's insufficient material for checkmate
__host__ __device__ bool is_insufficient_material(const ChessBoard* board);

// Check for threefold repetition (may need additional state tracking)
__host__ __device__ bool is_threefold_repetition(const ChessBoard* board);

// Check if the fifty-move rule applies
__host__ __device__ bool is_fifty_move_rule(const ChessBoard* board);