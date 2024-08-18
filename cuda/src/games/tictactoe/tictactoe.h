#ifndef TICTACTOE_H
#define TICTACTOE_H

#include <stdbool.h>

#define BOARD_SIZE 3
#define NUM_SQUARES (BOARD_SIZE * BOARD_SIZE)

typedef struct {
    int pieces[NUM_SQUARES];
} Board;

typedef struct {
    Board board;
} TicTacToeGame;

// Board functions
void board_init(Board* b);
bool board_is_legal_move(const Board* b, int pos);
void board_get_legal_moves(const Board* b, int moves[NUM_SQUARES], int* num_moves);
bool board_has_legal_moves(const Board* b);
void board_execute_move(Board* b, int pos, int player);
bool board_is_win(const Board* b, int player);
bool board_is_draw(const Board* b);

// Game functions
void game_init(TicTacToeGame* game);
void game_get_init_board(const TicTacToeGame* game, int board[NUM_SQUARES]);
void game_get_board_size(int* rows, int* cols);
int game_get_action_size(void);
void game_get_next_state(const TicTacToeGame* game, const int board[NUM_SQUARES], int player, int action, int next_board[NUM_SQUARES], int* next_player);
void game_get_valid_moves(const TicTacToeGame* game, const int board[NUM_SQUARES], int player, bool valid_moves[NUM_SQUARES]);
int game_get_game_ended(const TicTacToeGame* game, const int board[NUM_SQUARES], int player);
void game_get_canonical_form(const TicTacToeGame* game, const int board[NUM_SQUARES], int player, int canonical_board[NUM_SQUARES]);
void game_get_symmetries(const TicTacToeGame* game, const int board[NUM_SQUARES], const float pi[], int symmetries[8][NUM_SQUARES], float symmetries_pi[8][NUM_SQUARES], int* num_symmetries);
void game_string_representation(const TicTacToeGame* game, const int board[NUM_SQUARES], char* str, int str_size);
void game_display(const int board[NUM_SQUARES]);

// Player functions
int random_player_play(const TicTacToeGame* game, const int board[NUM_SQUARES]);
int human_player_play(const TicTacToeGame* game, const int board[NUM_SQUARES]);
int greedy_player_play(const TicTacToeGame* game, const int board[NUM_SQUARES]);

#endif // TICTACTOE_H