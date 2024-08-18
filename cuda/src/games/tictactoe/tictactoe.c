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

bool board_is_legal_move(const Board* b, int pos) {
    return (pos >= 0 && pos < NUM_SQUARES && b->pieces[pos] == 0);
}

void board_get_legal_moves(const Board* b, int moves[NUM_SQUARES], int* num_moves) {
    *num_moves = 0;
    for (int pos = 0; pos < NUM_SQUARES; pos++) {
        if (board_is_legal_move(b, pos)) {
            moves[(*num_moves)++] = pos;
        }
    }
}

bool board_has_legal_moves(const Board* b) {
    for (int pos = 0; pos < NUM_SQUARES; pos++) {
        if (board_is_legal_move(b, pos)) {
            return true;
        }
    }
    return false;
}

void board_execute_move(Board* b, int pos, int player) {
    b->pieces[pos] = player;
}

bool board_is_win(const Board* b, int player) {
    // Check rows
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (b->pieces[i*BOARD_SIZE] == player && 
            b->pieces[i*BOARD_SIZE + 1] == player && 
            b->pieces[i*BOARD_SIZE + 2] == player) {
            return true;
        }
    }
    
    // Check columns
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (b->pieces[i] == player && 
            b->pieces[i + BOARD_SIZE] == player && 
            b->pieces[i + 2*BOARD_SIZE] == player) {
            return true;
        }
    }
    
    // Check diagonals
    if ((b->pieces[0] == player && b->pieces[4] == player && b->pieces[8] == player) ||
        (b->pieces[2] == player && b->pieces[4] == player && b->pieces[6] == player)) {
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

void game_get_init_board(const TicTacToeGame* game, int board[NUM_SQUARES]) {
    memcpy(board, game->board.pieces, sizeof(game->board.pieces));
}

void game_get_board_size(int* rows, int* cols) {
    *rows = BOARD_SIZE;
    *cols = BOARD_SIZE;
}

int game_get_action_size(void) {
    return NUM_SQUARES;
}

void game_get_next_state(const TicTacToeGame* game, const int board[NUM_SQUARES], int player, int action, int next_board[NUM_SQUARES], int* next_player) {
    memcpy(next_board, board, sizeof(int) * NUM_SQUARES);
    next_board[action] = player;
    *next_player = -player;
}

void game_get_valid_moves(const TicTacToeGame* game, const int board[NUM_SQUARES], int player, bool valid_moves[NUM_SQUARES]) {
    for (int i = 0; i < NUM_SQUARES; i++) {
        valid_moves[i] = (board[i] == 0);
    }
}

int game_get_game_ended(const TicTacToeGame* game, const int board[NUM_SQUARES], int player) {
    Board b;
    memcpy(b.pieces, board, sizeof(b.pieces));
    
    if (board_is_win(&b, player)) return 1;
    if (board_is_win(&b, -player)) return -1;
    if (board_is_draw(&b)) return 1e-4;
    return 0;
}

void game_get_canonical_form(const TicTacToeGame* game, const int board[NUM_SQUARES], int player, int canonical_board[NUM_SQUARES]) {
    for (int i = 0; i < NUM_SQUARES; i++) {
        canonical_board[i] = board[i] * player;
    }
}

void game_get_symmetries(const TicTacToeGame* game, const int board[NUM_SQUARES], const float pi[], int symmetries[8][NUM_SQUARES], float symmetries_pi[8][NUM_SQUARES], int* num_symmetries) {
    *num_symmetries = 8;
    
    // Original
    memcpy(symmetries[0], board, NUM_SQUARES * sizeof(int));
    memcpy(symmetries_pi[0], pi, NUM_SQUARES * sizeof(float));
    
    // Rotate 90
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[1][BOARD_SIZE*j + (BOARD_SIZE-1-i)] = board[BOARD_SIZE*i + j];
    
    // Rotate 180
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[2][BOARD_SIZE*(BOARD_SIZE-1-i) + (BOARD_SIZE-1-j)] = board[BOARD_SIZE*i + j];
    
    // Rotate 270
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[3][BOARD_SIZE*(BOARD_SIZE-1-j) + i] = board[BOARD_SIZE*i + j];
    
    // Flip horizontal
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[4][BOARD_SIZE*i + (BOARD_SIZE-1-j)] = board[BOARD_SIZE*i + j];
    
    // Flip vertical
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[5][BOARD_SIZE*(BOARD_SIZE-1-i) + j] = board[BOARD_SIZE*i + j];
    
    // Flip diagonal
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[6][BOARD_SIZE*j + i] = board[BOARD_SIZE*i + j];
    
    // Flip anti-diagonal
    for (int i = 0; i < BOARD_SIZE; i++)
        for (int j = 0; j < BOARD_SIZE; j++)
            symmetries[7][BOARD_SIZE*(BOARD_SIZE-1-j) + (BOARD_SIZE-1-i)] = board[BOARD_SIZE*i + j];
    
    // Apply the same transformations to pi
    for (int s = 1; s < 8; s++)
        for (int i = 0; i < NUM_SQUARES; i++)
            symmetries_pi[s][i] = pi[symmetries[s][i]];
}

void game_string_representation(const TicTacToeGame* game, const int board[NUM_SQUARES], char* str, int str_size) {
    snprintf(str, str_size, "%d%d%d%d%d%d%d%d%d", 
             board[0], board[1], board[2],
             board[3], board[4], board[5],
             board[6], board[7], board[8]);
}

void game_display(const int board[NUM_SQUARES]) {
    printf("   0 1 2\n");
    printf("  -------\n");
    for (int i = 0; i < BOARD_SIZE; i++) {
        printf("%d |", i);
        for (int j = 0; j < BOARD_SIZE; j++) {
            char symbol = ' ';
            if (board[i*BOARD_SIZE + j] == 1) symbol = 'X';
            else if (board[i*BOARD_SIZE + j] == -1) symbol = 'O';
            printf("%c|", symbol);
        }
        printf("\n  -------\n");
    }
}

float game_evaluate(const TicTacToeGame* game, const int board[NUM_SQUARES], int player) {
    int opponent = -player;
    int game_result = game_get_game_ended(game, board, player);
    
    if (game_result == 1) {
        return 1.0;  // Current player wins
    } else if (game_result == -1) {
        return -1.0;  // Opponent wins
    } else if (game_result == 1e-4) {
        return 0.0;  // Draw
    }
    
    // If the game hasn't ended, we'll do a simple evaluation
    float score = 0.0;
    
    // Check rows, columns, and diagonals
    int lines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  // Rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  // Columns
        {0, 4, 8}, {2, 4, 6}  // Diagonals
    };
    
    for (int i = 0; i < 8; i++) {
        int player_count = 0;
        int opponent_count = 0;
        for (int j = 0; j < 3; j++) {
            if (board[lines[i][j]] == player) player_count++;
            else if (board[lines[i][j]] == opponent) opponent_count++;
        }
        
        if (player_count > 0 && opponent_count == 0) {
            score += 0.1 * player_count;
        } else if (opponent_count > 0 && player_count == 0) {
            score -= 0.1 * opponent_count;
        }
    }
    
    return score;
}

// Player functions
int random_player_play(const TicTacToeGame* game, const int board[NUM_SQUARES]) {
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

int human_player_play(const TicTacToeGame* game, const int board[NUM_SQUARES]) {
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

int greedy_player_play(const TicTacToeGame* game, const int board[NUM_SQUARES]) {
    bool valid_moves[NUM_SQUARES];
    game_get_valid_moves(game, board, 1, valid_moves);
    
    int best_action = -1;
    float best_score = -FLT_MAX;
    
    for (int action = 0; action < NUM_SQUARES; action++) {
        if (!valid_moves[action]) continue;
        
        int next_board[NUM_SQUARES];
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

void tictactoe_game_wrapper_init(TicTacToeGameWrapper* wrapper) {
    wrapper->base.init = (void (*)(IGame*))game_init;
    wrapper->base.get_init_board = (void (*)(const IGame*, int*))game_get_init_board;
    wrapper->base.get_board_size = (void (*)(const IGame*, int*, int*))game_get_board_size;
    wrapper->base.get_action_size = (int (*)(const IGame*))game_get_action_size;
    wrapper->base.get_next_state = (void (*)(const IGame*, const int*, int, int, int*, int*))game_get_next_state;
    wrapper->base.get_valid_moves = (void (*)(const IGame*, const int*, int, bool*))game_get_valid_moves;
    wrapper->base.get_game_ended = (int (*)(const IGame*, const int*, int))game_get_game_ended;
    wrapper->base.get_canonical_form = (void (*)(const IGame*, const int*, int, int*))game_get_canonical_form;
    wrapper->base.get_symmetries = (void (*)(const IGame*, const int*, const float*, int (*)[MAX_BOARD_SIZE], float (*)[MAX_BOARD_SIZE], int*))game_get_symmetries;
    wrapper->base.string_representation = (void (*)(const IGame*, const int*, char*, int))game_string_representation;
    wrapper->base.display = (void (*)(const IGame*, const int*))game_display;
    
    wrapper->base.evaluate = (float (*)(const IGame*, const int*, int))game_evaluate;

    game_init(&wrapper->game);
}