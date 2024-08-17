#include "tictactoe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

// Board functions
void board_init(Board* b) {
    memset(b->pieces, 0, sizeof(b->pieces));
}

bool board_is_legal_move(const Board* b, int x, int y) {
    return (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE && b->pieces[x][y] == 0);
}

void board_get_legal_moves(const Board* b, int moves[NUM_SQUARES], int* num_moves) {
    *num_moves = 0;
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board_is_legal_move(b, x, y)) {
                moves[(*num_moves)++] = BOARD_SIZE * x + y;
            }
        }
    }
}

bool board_has_legal_moves(const Board* b) {
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board_is_legal_move(b, x, y)) {
                return true;
            }
        }
    }
    return false;
}

void board_execute_move(Board* b, int x, int y, int player) {
    b->pieces[x][y] = player;
}

bool board_is_win(const Board* b, int player) {
    // Check rows and columns
    for (int i = 0; i < BOARD_SIZE; i++) {
        if ((b->pieces[i][0] == player && b->pieces[i][1] == player && b->pieces[i][2] == player) ||
            (b->pieces[0][i] == player && b->pieces[1][i] == player && b->pieces[2][i] == player)) {
            return true;
        }
    }
    
    // Check diagonals
    if ((b->pieces[0][0] == player && b->pieces[1][1] == player && b->pieces[2][2] == player) ||
        (b->pieces[0][2] == player && b->pieces[1][1] == player && b->pieces[2][0] == player)) {
        return true;
    }
    
    return false;
}

bool board_is_draw(const Board* b) {
    return !board_has_legal_moves(b) && !board_is_win(b, 1) && !board_is_win(b, -1);
}

// Game functions
void game_init(TicTacToeGame* game) {
    board_init(&game->board);
}

void game_get_init_board(const TicTacToeGame* game, int board[BOARD_SIZE][BOARD_SIZE]) {
    memcpy(board, game->board.pieces, sizeof(game->board.pieces));
}

void game_get_board_size(int* rows, int* cols) {
    *rows = BOARD_SIZE;
    *cols = BOARD_SIZE;
}

int game_get_action_size(void) {
    return NUM_SQUARES;
}

void game_get_next_state(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE], int player, int action, int next_board[BOARD_SIZE][BOARD_SIZE], int* next_player) {
    memcpy(next_board, board, sizeof(int) * BOARD_SIZE * BOARD_SIZE);
    int x = action / BOARD_SIZE;
    int y = action % BOARD_SIZE;
    next_board[x][y] = player;
    *next_player = -player;
}

void game_get_valid_moves(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE], int player, bool valid_moves[NUM_SQUARES]) {
    for (int i = 0; i < NUM_SQUARES; i++) {
        int x = i / BOARD_SIZE;
        int y = i % BOARD_SIZE;
        valid_moves[i] = (board[x][y] == 0);
    }
}

int game_get_game_ended(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE], int player) {
    Board b;
    memcpy(b.pieces, board, sizeof(b.pieces));
    
    if (board_is_win(&b, player)) return 1;
    if (board_is_win(&b, -player)) return -1;
    if (board_is_draw(&b)) return 1e-4;
    return 0;
}

void game_get_canonical_form(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE], int player, int canonical_board[BOARD_SIZE][BOARD_SIZE]) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            canonical_board[i][j] = board[i][j] * player;
        }
    }
}

void game_get_symmetries(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE], const float pi[], int symmetries[8][NUM_SQUARES], float symmetries_pi[8][NUM_SQUARES], int* num_symmetries) {
    *num_symmetries = 8;
    
    // Original
    memcpy(symmetries[0], board, NUM_SQUARES * sizeof(int));
    memcpy(symmetries_pi[0], pi, NUM_SQUARES * sizeof(float));
    
    // Rotate 90
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[1][BOARD_SIZE*j + (BOARD_SIZE-1-i)] = board[i][j];
    
    // Rotate 180
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[2][BOARD_SIZE*(BOARD_SIZE-1-i) + (BOARD_SIZE-1-j)] = board[i][j];
    
    // Rotate 270
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[3][BOARD_SIZE*(BOARD_SIZE-1-j) + i] = board[i][j];
    
    // Flip horizontal
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[4][BOARD_SIZE*i + (BOARD_SIZE-1-j)] = board[i][j];
    
    // Flip vertical
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[5][BOARD_SIZE*(BOARD_SIZE-1-i) + j] = board[i][j];
    
    // Flip diagonal
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[6][BOARD_SIZE*j + i] = board[i][j];
    
    // Flip anti-diagonal
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[7][BOARD_SIZE*(BOARD_SIZE-1-j) + (BOARD_SIZE-1-i)] = board[i][j];
    
    // Apply the same transformations to pi
    for (int s = 1; s < 8; s++)
        for (int i = 0; i < NUM_SQUARES; i++)
            symmetries_pi[s][i] = pi[symmetries[s][i]];
}

void game_string_representation(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE], char* str, int str_size) {
    snprintf(str, str_size, "%d%d%d%d%d%d%d%d%d", 
             board[0][0], board[0][1], board[0][2],
             board[1][0], board[1][1], board[1][2],
             board[2][0], board[2][1], board[2][2]);
}

void game_display(const int board[BOARD_SIZE][BOARD_SIZE]) {
    printf("   0 1 2\n");
    printf("  -------\n");
    for (int i = 0; i < BOARD_SIZE; i++) {
        printf("%d |", i);
        for (int j = 0; j < BOARD_SIZE; j++) {
            char symbol = ' ';
            if (board[i][j] == 1) symbol = 'X';
            else if (board[i][j] == -1) symbol = 'O';
            printf("%c|", symbol);
        }
        printf("\n  -------\n");
    }
}

// Player functions
int random_player_play(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE]) {
    bool valid_moves[NUM_SQUARES];
    game_get_valid_moves(game, board, 1, valid_moves);
    
    int num_valid_moves = 0;
    int valid_actions[NUM_SQUARES];
    for (int i = 0; i < NUM_SQUARES; i++) {
        if (valid_moves[i]) {
            valid_actions[num_valid_moves++] = i;
        }
    }
    
    if (num_valid_moves == 0) return -1;  // No valid moves
    
    return valid_actions[rand() % num_valid_moves];
}

int human_player_play(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE]) {
    bool valid_moves[NUM_SQUARES];
    game_get_valid_moves(game, board, 1, valid_moves);
    
    printf("Valid moves:\n");
    for (int i = 0; i < NUM_SQUARES; i++) {
        if (valid_moves[i]) {
            printf("%d %d\n", i / BOARD_SIZE, i % BOARD_SIZE);
        }
    }
    
    while (1) {
        int x, y;
        printf("Enter your move (row col): ");
        if (scanf("%d %d", &x, &y) != 2) {
            printf("Invalid input. Try again.\n");
            while (getchar() != '\n');  // Clear input buffer
            continue;
        }
        
        int action = BOARD_SIZE * x + y;
        if (action >= 0 && action < NUM_SQUARES && valid_moves[action]) {
            return action;
        }
        printf("Invalid move. Try again.\n");
    }
}

int greedy_player_play(const TicTacToeGame* game, const int board[BOARD_SIZE][BOARD_SIZE]) {
    bool valid_moves[NUM_SQUARES];
    game_get_valid_moves(game, board, 1, valid_moves);
    
    int best_action = -1;
    float best_score = -FLT_MAX;
    
    for (int action = 0; action < NUM_SQUARES; action++) {
        if (!valid_moves[action]) continue;
        
        int next_board[BOARD_SIZE][BOARD_SIZE];
        int next_player;
        game_get_next_state(game, board, 1, action, next_board, &next_player);
        
        float score = -game_get_game_ended(game, next_board, 1);
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    
    return best_action;
}