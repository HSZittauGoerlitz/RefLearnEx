# %% Imports
import ray
import ray.rllib.agents.dqn as dqn
import shutil

from cartpole import CartPoleRegulatorEnv

# %% Environments
env = CartPoleRegulatorEnv({})

# %% Configuration
CHECKPOINT_ROOT = "tmp/"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
ray_results = "./ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

config = dqn.DEFAULT_CONFIG.copy()
config['log_level'] = "WARN"
config['num_workers'] = 0
config['framework'] = "torch"
config['horizon'] = 500

# %% init ray
ray.shutdown()
session = ray.init(ignore_reinit_error=True)

# %% Controller
agent = dqn.DQNTrainer(config, env=CartPoleRegulatorEnv)

for epoch in range(50):
    print(f"{epoch}: {agent.train()['episode_reward_mean']}")


# %% demo mode
obs = env.reset()
done = False
while not done:
    aIdx = agent.compute_single_action(obs)
    obs, _, done, _ = env.step(aIdx)
    env.render()
env.close()
