#include "arena.cuh"
#include <stdio.h>
#include <stdlib.h>

Arena* create_arena(Player* player1, Player* player2, IGame* game) {
    Arena* arena = (Arena*)malloc(sizeof(Arena)); // Allocate memory for the arena
    arena->player1 = player1; // Initialize player 1
    arena->player2 = player2; // Initialize player 2
    arena->game = game; // Initialize game instance
    return arena;
}

int play_game(Arena* arena, bool verbose) {
    int board_size = arena->game->get_action_size(arena->game); // Get board size from game
    int* board = (int*)malloc(sizeof(int) * board_size); // Allocate memory for board
    arena->game->get_init_board(arena->game, board); // Initialize board
    
    int current_player = 1; // Initialize current player
    int game_result = 0; // Initialize game result
    
    while (game_result == 0) {
        Player* current_player_obj = (current_player == 1) ? arena->player1 : arena->player2; // Get current player object
        int action = current_player_obj->get_action(current_player_obj, board); // Get action from current player
        
        int next_board[MAX_BOARD_SIZE]; // Next board state
        int next_player; // Next player
        arena->game->get_next_state(arena->game, board, current_player, action, next_board, &next_player); // Get next state
        
        memcpy(board, next_board, sizeof(int) * board_size); // Update board
        current_player = next_player; // Update current player
        
        game_result = arena->game->get_game_ended(arena->game, board, current_player); // Check game ended
        
        if (verbose) {
            arena->game->display(arena->game, board); // Display board if verbose
        }
    }
    
    free(board); // Free board memory
    return game_result;
}

void play_games(Arena* arena, int num_games, int* wins, int* losses, int* draws) {
    *wins = 0; // Initialize wins
    *losses = 0; // Initialize losses
    *draws = 0; // Initialize draws
    
    for (int i = 0; i < num_games; i++) {
        int result = play_game(arena, false); // Play game
        if (result == 1) (*wins)++; // Increment wins
        else if (result == -1) (*losses)++; // Increment losses
        else (*draws)++; // Increment draws
    }
}

void destroy_arena(Arena* arena) {
    free(arena); // Free arena memory
}