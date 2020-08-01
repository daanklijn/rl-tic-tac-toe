from dqn_tictactoe.tictactoe_multi_env import TicTacToeMultiEnv
from pytest import raises, fixture

X_SYMBOL = TicTacToeMultiEnv.X_SYMBOL
O_SYMBOL = TicTacToeMultiEnv.O_SYMBOL


@fixture
def env():
    return TicTacToeMultiEnv({})


def test_initial_board_is_empty(env):
    obs = env.reset()
    for i in range(env.NUMBER_FIELDS):
        assert obs[X_SYMBOL][i] == env.EMPTY_SYMBOL
        assert obs[O_SYMBOL][i] == env.EMPTY_SYMBOL


def test_that_player_knows_its_symbol(env):
    obs = env.reset()
    assert obs[X_SYMBOL][env.USER_SYMBOL_INDEX] == X_SYMBOL
    assert obs[O_SYMBOL][env.USER_SYMBOL_INDEX] == O_SYMBOL


def test_X_has_first_turn(env):
    obs = env.reset()
    assert obs[X_SYMBOL][env.USER_TURN_INDEX] == X_SYMBOL
    assert obs[O_SYMBOL][env.USER_TURN_INDEX] == X_SYMBOL


def test_O_has_second_turn_after_correct_move(env):
    env.reset()
    action = {
        X_SYMBOL: 0,
        O_SYMBOL: env.WAIT_MOVE
    }
    obs, rew, done, _ = env.step(action)
    assert obs[X_SYMBOL][env.USER_TURN_INDEX] == O_SYMBOL
    assert obs[O_SYMBOL][env.USER_TURN_INDEX] == O_SYMBOL


def test_X_has_second_turn_after_incorrect_move(env):
    env.reset()
    action = {
        X_SYMBOL: env.WAIT_MOVE,
        O_SYMBOL: env.WAIT_MOVE
    }
    obs, rew, done, _ = env.step(action)
    assert obs[X_SYMBOL][env.USER_TURN_INDEX] == X_SYMBOL
    assert obs[O_SYMBOL][env.USER_TURN_INDEX] == X_SYMBOL


def test_rewards_correct_move(env):
    env.reset()
    action = {
        X_SYMBOL: 0,
        O_SYMBOL: env.WAIT_MOVE
    }
    obs, rew, done, _ = env.step(action)
    assert rew[X_SYMBOL] == env.GOOD_MOVE_REWARD
    assert rew[O_SYMBOL] == env.GOOD_MOVE_REWARD


def test_rewards_incorrect_move(env):
    env.reset()
    action = {
        X_SYMBOL: env.WAIT_MOVE,
        O_SYMBOL: env.WAIT_MOVE
    }
    obs, rew, done, _ = env.step(action)
    assert rew[X_SYMBOL] == env.BAD_MOVE_REWARD
    assert rew[O_SYMBOL] == env.GOOD_MOVE_REWARD


def test_win_lose_reward(env):
    env.reset()
    actions = [
        {X_SYMBOL: 0, O_SYMBOL: env.WAIT_MOVE},
        {X_SYMBOL: env.WAIT_MOVE, O_SYMBOL: 8},
        {X_SYMBOL: 1, O_SYMBOL: env.WAIT_MOVE},
        {X_SYMBOL: env.WAIT_MOVE, O_SYMBOL: 7},
        {X_SYMBOL: 2, O_SYMBOL: env.WAIT_MOVE},
        {X_SYMBOL: env.WAIT_MOVE, O_SYMBOL: 6},
        ]

    env.set_verbose()
    for action in actions:
        obs, rew, done, _ = env.step(action)
    assert rew[X_SYMBOL] == env.WIN_REWARD
    assert rew[O_SYMBOL] == env.LOSE_REWARD
    assert done

def test_draw_reward(env):
    env.reset()
    actions = [
        {X_SYMBOL: 0, O_SYMBOL: env.WAIT_MOVE},
        {X_SYMBOL: env.WAIT_MOVE, O_SYMBOL: 1},
        {X_SYMBOL: 2, O_SYMBOL: env.WAIT_MOVE},
        {X_SYMBOL: env.WAIT_MOVE, O_SYMBOL: 4},
        {X_SYMBOL: 3, O_SYMBOL: env.WAIT_MOVE},
        {X_SYMBOL: env.WAIT_MOVE, O_SYMBOL: 5},
        {X_SYMBOL: 7, O_SYMBOL: env.WAIT_MOVE},
        {X_SYMBOL: env.WAIT_MOVE, O_SYMBOL: 6},
        {X_SYMBOL: 8, O_SYMBOL: env.WAIT_MOVE},
    ]

    env.set_verbose()
    for action in actions:
        obs, rew, done, _ = env.step(action)
    assert done
    assert rew[X_SYMBOL] == env.DRAW_REWARD
    assert rew[O_SYMBOL] == env.DRAW_REWARD
