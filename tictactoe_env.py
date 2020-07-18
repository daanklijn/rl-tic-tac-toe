import random

from gym.spaces import Discrete, Box
from gym import Env

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env import BaseEnv


class TicTacToeEnv(Env):
    """Single-player environment for tic tac toe."""

    EMPTY_SYMBOL = 0
    X_SYMBOL = 1
    O_SYMBOL = 2
    SYMBOL_MAP = {
        X_SYMBOL: 'X',
        O_SYMBOL: 'O',
        EMPTY_SYMBOL: '.',
    }
    NUMBER_FIELDS = 9
    BOARD_WIDTH = 3

    def __init__(self, config):
        self.action_space = Discrete(9)
        self.observation_space = Box(0, 2, [9])
        self.reset()
        self.history = []

    def reset(self):
        self._init_board()
        self.history = []
        self._save_board()
        return self.board

    def step(self, action):
        rew = 0
        if self._field_is_filed(action):
            rew = -1
        else:
            self.board[action] = self.X_SYMBOL

        done = self._board_is_full()
        if done:
            rew = self._evaluate_board()
            self.reset()
        else:
            self._make_move()

        obs = self.board
        done = self._board_is_full()
        self._save_board()
        return obs, rew, done, {}

    def _save_board(self):
        self.history.append(self.board.copy())

    def _evaluate_board(self):
        # somehow scores for horizontal wins not right
        horizontal_groups = [self.board[0:2], self.board[3:5], self.board[6:8]]
        vertical_groups = [[self.board[0], self.board[3], self.board[6]],
                           [self.board[1], self.board[4], self.board[7]],
                           [self.board[2], self.board[5], self.board[8]]]
        diagonal_groups = [[self.board[0], self.board[4], self.board[8]],
                           [self.board[2], self.board[4], self.board[6]]]
        for group in (horizontal_groups + vertical_groups + diagonal_groups):
            if group.count(self.X_SYMBOL) == self.BOARD_WIDTH:
                return 10
            if group.count(self.O_SYMBOL) == self.BOARD_WIDTH:
                return -10
        return 0

    def _print_board(self, board):
        for i, field in enumerate(board):
            if i % 3 == 0:
                print('')
            print(self.SYMBOL_MAP[field], end='')

    def _make_move(self):
        field = random.choice(self._empty_fields())
        self.board[field] = self.O_SYMBOL

    def _empty_fields(self):
        return [i for i in self.board if i is self.EMPTY_SYMBOL]

    def _field_is_filed(self, field_index):
        return self.board[field_index] != self.EMPTY_SYMBOL

    def _board_is_full(self):
        return len(self._empty_fields()) == 0

    def _init_board(self):
        self.board = [self.EMPTY_SYMBOL for _ in range(self.NUMBER_FIELDS)]

    def _print_history(self):
        for i, board in enumerate(self.history):
            print(f"\n\n---ROUND-{i}---")
            self._print_board(board)
        print("\nSCORE: "+str(self._evaluate_board()))
