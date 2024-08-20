#ifndef PLAYERS_CUH
#define PLAYERS_CUH

#include "../games/game.cuh"
#include "../mcts/mcts.cuh"
#include "../networks/gat/gat.cuh"

typedef struct {
    int (*get_action)(const IGame* game, const int* board, int player);
} IPlayer;

typedef struct {
    IPlayer base;
} RandomPlayer;

typedef struct {
    IPlayer base;
    MCTSState* mcts_state;
    float temperature;
} MCTSPlayer;

typedef struct {
    IPlayer base;
    INeuralNet* net;
    MCTSState* mcts_state;
    float temperature;
} NNetPlayer;

RandomPlayer* create_random_player();
MCTSPlayer* create_mcts_player(IGame* game, int num_simulations, float temperature);
NNetPlayer* create_nnet_player(IGame* game, INeuralNet* net, int num_simulations, float temperature);

void destroy_random_player(RandomPlayer* player);
void destroy_mcts_player(MCTSPlayer* player);
void destroy_nnet_player(NNetPlayer* player);

#endif // PLAYERS_CUH