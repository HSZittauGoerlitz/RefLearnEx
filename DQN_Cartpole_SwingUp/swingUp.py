# %% Imports
from d3rlpy.algos import DQN
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

from cartpole import CartPoleRegulatorEnv

# %% Environments
train_env = CartPoleRegulatorEnv(1., 0., mode="train")
eval_env = CartPoleRegulatorEnv(1., 0., mode="eval")
env = CartPoleRegulatorEnv(1., 0., mode="demo")


# %% Controller
dqn = DQN(batch_size=32,
          learning_rate=2.5e-4,
          target_update_interval=100,
          use_gpu=False)

# replay buffer for experience replay
buffer = ReplayBuffer(maxlen=100000, env=train_env)

# epilon-greedy explorer
explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                    end_epsilon=0.1,
                                    duration=10000)

# %% Training
dqn.fit_online(train_env,
               buffer,
               explorer,
               n_steps=30000,
               eval_env=eval_env,
               n_steps_per_epoch=1000,
               update_start_step=1000)

# %%
obs = env.reset()
done = False
while not done:
    aIdx = dqn.predict([obs])[0]
    obs, _, done, _ = env.step(aIdx)
    env.render()
env.close()
# %%
