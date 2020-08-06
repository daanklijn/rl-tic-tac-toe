import random

from ray.rllib.policy.policy import Policy

from dqn_tictactoe.tictactoe_multi_env import TicTacToeMultiEnv as env


class FirstEmptyFieldPolicy(Policy):
    """Pick first empty field the heuristic finds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        def determine_action(obs):
            # Wait if it's not player's turn.
            if obs[env.USER_TURN_INDEX] != obs[env.USER_SYMBOL_INDEX]:
                return env.WAIT_MOVE

            # Make a move on the first empty field heuristic can find.
            for i, symbol in enumerate(obs):
                if symbol == env.EMPTY_SYMBOL:
                    return i
            raise Exception('Heuristic did not find empty.')

        return [determine_action(obs) for obs in obs_batch], [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        return None
