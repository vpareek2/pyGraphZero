#include "arena.cuh"
#include <stdio.h>
#include <stdlib.h>

Arena* create_arena(Player* player1, Player* player2, IGame* game) {
    Arena* arena = (Arena*)malloc(sizeof(Arena));
    arena->player1 = player1;
    arena->player2 = player2;
    arena->game = game;
    return arena;
}

int play_game(Arena* arena, bool verbose) {
    int board_size = arena->game->get_action_size(arena->game);
    int* board = (int*)malloc(sizeof(int) * board_size);
    arena->game->get_init_board(arena->game, board);
    
    int current_player = 1;
    int game_result = 0;
    
    while (game_result == 0) {
        Player* current_player_obj = (current_player == 1) ? arena->player1 : arena->player2;
        int action = current_player_obj->get_action(current_player_obj, board);
        
        int next_board[MAX_BOARD_SIZE];
        int next_player;
        arena->game->get_next_state(arena->game, board, current_player, action, next_board, &next_player);
        
        memcpy(board, next_board, sizeof(int) * board_size);
        current_player = next_player;
        
        game_result = arena->game->get_game_ended(arena->game, board, current_player);
        
        if (verbose) {
            arena->game->display(arena->game, board);
        }
    }
    
    free(board);
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
    }
}

void destroy_arena(Arena* arena) {
    free(arena);
}