#ifndef ARENA_CUH
#define ARENA_CUH

#include "../player/player.cuh"
#include "../game/game.cuh"

#define MAX_BOARD_SIZE 64  // Assuming the maximum board size is for 8x8 chess

typedef struct Arena {
    Player* player1;
    Player* player2;
    IGame* game;
} Arena;

Arena* create_arena(Player* player1, Player* player2, IGame* game);
int play_game(Arena* arena, bool verbose);
void play_games(Arena* arena, int num_games, int* wins, int* losses, int* draws);
void destroy_arena(Arena* arena);

#endif // ARENA_CUH
