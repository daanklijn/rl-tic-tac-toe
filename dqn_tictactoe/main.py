import ray.rllib.agents
import wandb
from ray.rllib.agents.registry import get_agent_class

from dqn_tictactoe.first_empty_field_policy import FirstEmptyFieldPolicy
from dqn_tictactoe.tictactoe_multi_env import TicTacToeMultiEnv

wandb.init(project='dqn-tic-tac-toe')
ray.init()

trainer = 'A3C'
trained_policy = trainer + '_policy'

def policy_mapping_fn(agent_id):
    mapping = {TicTacToeMultiEnv.O_SYMBOL: trained_policy,
               TicTacToeMultiEnv.X_SYMBOL: "heuristic"}
    return mapping[agent_id]

config = {
    "env": TicTacToeMultiEnv,
    "multiagent": {
        "policies_to_train": [trained_policy],
        "policies": {
            trained_policy: (None, TicTacToeMultiEnv.OBSERVATION_SPACE,
                           TicTacToeMultiEnv.ACTION_SPACE, {}),
            "heuristic": (FirstEmptyFieldPolicy, TicTacToeMultiEnv.OBSERVATION_SPACE,
                          TicTacToeMultiEnv.ACTION_SPACE, {}),
        },
        "policy_mapping_fn": policy_mapping_fn
    },
}

cls = get_agent_class(trainer) if isinstance(trainer, str) else trainer
trainer_obj = cls(config=config)
env = trainer_obj.workers.local_worker().env

while True:
    results = trainer_obj.train()
    results.pop('config')
    wandb.log(results)
    if results['episodes_total'] > 10000:
        break
