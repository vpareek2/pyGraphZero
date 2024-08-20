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
    int* board = arena->board;
    arena->game->get_init_board(arena->game, board);
    
    int current_player = 1;
    int game_result = 0;
    
    while (game_result == 0) {
        Player* current_player_obj = (current_player == 1) ? arena->player1 : arena->player2;
        int action = current_player_obj->get_action(current_player_obj, board);
        
        if (!arena->game->is_valid_action(arena->game, board, action)) {
            fprintf(stderr, "Error: Invalid action %d\n", action);
            return 0; // Draw or handle error as appropriate
        }
        
        int next_board[MAX_BOARD_SIZE];
        int next_player;
        arena->game->get_next_state(arena->game, board, current_player, action, next_board, &next_player);
        
        memcpy(board, next_board, sizeof(int) * arena->board_size);
        current_player = next_player;
        
        game_result = arena->game->get_game_ended(arena->game, board, current_player);
        
        if (verbose) {
            arena->game->display(arena->game, board);
        }
    }
    
    return game_result;
}

void play_games(Arena* arena, int num_games, int* wins, int* losses, int* draws) {
    *wins = 0;
    *losses = 0;
    *draws = 0;
    
    for (int i = 0; i < num_games; i++) {
        int result = play_game(arena, false);
        if (result == 1) (*wins)++;
        else if (result == -1) (*losses)++;
        else (*draws)++;
        
        // Swap players every other game
        if (i == num_games / 2 - 1) {
            Player* temp = arena->player1;
            arena->player1 = arena->player2;
            arena->player2 = temp;
        }
    }
}

void destroy_arena(Arena* arena) {
    free(arena); // Free arena memory
}