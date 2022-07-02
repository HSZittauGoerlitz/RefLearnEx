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
env.render_sleep = 0.

for epoch in range(500):
    res = agent.train()
    print(f"{epoch}: {res['episode_reward_mean']}")

    if res['episode_reward_mean'] > 25:
        break

    obs = env.reset(True)
    if epoch % 10 == 0:
        for _ in range(500):
            aIdx = agent.compute_single_action(obs)
            obs, _, done, _ = env.step(aIdx)
            env.render()
            if done:
                break
        env.close()


# %% demo mode
env.render_sleep = 0.02
obs = env.reset(True)
done = False
while not done:
    aIdx = agent.compute_single_action(obs)
    obs, _, done, _ = env.step(aIdx)
    env.render()
env.close()
