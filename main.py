import logging
import coloredlogs
import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from alphazero.self_play import Coach
from games.tictactoe import TicTacToeGame
from networks.tictactoe_resnet import NNetWrapper as nn
# from networks.tictactoe_gat import NNetWrapper as nn
# from networks.connect4_resnet import NNetWrapper as nn
# from networks.connect4_gat import NNetWrapper as nn
# from networks.chess_resnet import NNetWrapper as nn
# from networks.chess_gat import NNetWrapper as nn
from utils import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_training(rank, world_size, args):
    setup(rank, world_size)
    
    log = logging.getLogger(__name__)
    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

    log.info('Loading %s...', TicTacToeGame.__name__)
    g = TicTacToeGame()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TicTacToe AlphaZero Training')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    parser.add_argument('--load_model', action='store_true', help='Load a pre-trained model')
    parser.add_argument('--load_folder_file', nargs=2, type=str, default=('/dev/models/8x100x50', 'best.pth.tar'), help='Folder and file to load the model from')
    
    cli_args = parser.parse_args()

    args = dotdict({
        'numIters': 1000,
        'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,
        'updateThreshold': 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
        'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,
        'checkpoint': './temp/',
        'load_model': cli_args.load_model,
        'load_folder_file': cli_args.load_folder_file,
        'numItersForTrainExamplesHistory': 20,
        'distributed': cli_args.distributed,
        'world_size': cli_args.world_size,
    })

# Usage: python main.py --distributed --world_size 8
# Means 8 gpus.

    if args.distributed:
        mp.spawn(run_training, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        run_training(0, 1, args)