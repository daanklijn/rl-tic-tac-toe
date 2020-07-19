import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import wandb
from tictactoe_env import TicTacToeEnv

ray.init(include_webui=False)
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["eager"] = False

# maybe pass in more than only model config, also env settings
wandb.init(project="rl-tic-tac-toe", config=config['model'])
trainer = dqn.DQNTrainer(config=dqn.DEFAULT_CONFIG, env=TicTacToeEnv)
# trainer = ppo.PPOTrainer(config=config, env=TicTacToeEnv)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    result_dict = {k: v for (k, v) in result.items() if not isinstance(v, dict)}
    wandb.log(result_dict)
    print(trainer.workers.local_worker().env._print_history())
    print("AVG REWARD: "+str(result_dict['episode_reward_mean']))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
