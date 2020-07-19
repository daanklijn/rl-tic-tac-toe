from dqn_tictactoe.tictactoe_env import TicTacToeEnv
from pytest import raises


def test_initial_board_is_empty():
    env = TicTacToeEnv({})
    assert len(env._empty_fields()) == env.NUMBER_FIELDS


def test_making_valid_action():
    env = TicTacToeEnv({})
    env.step(0)
    assert env.board[0] == env.X_SYMBOL
    assert len(env._empty_fields()) == (env.NUMBER_FIELDS - 2)


def test_making_invalid_action():
    env = TicTacToeEnv({})
    obs, rew, done, _ = env.step(0)
    assert rew == 0
    obs, rew, done, _ = env.step(0)
    assert rew == -1

def test_action_greater_than_board_fields_raises_value_error():
    env = TicTacToeEnv({})
    with raises(ValueError):
        env.step(env.NUMBER_FIELDS+1)

def test_game_is_over_after_5_steps():
    env = TicTacToeEnv({})
    for i in range(5):
        action = env._empty_fields()[0]
        obs, rew, done, _ = env.step(action)
    assert done


def test_receives_reward_when_game_is_over():
    env = TicTacToeEnv({})
    done = False
    while not done:
        action = env._empty_fields()[0]
        obs, rew, done, _ = env.step(action)
    assert rew in [env.DRAW_REWARD, env.LOSE_REWARD, env.WIN_REWARD]


