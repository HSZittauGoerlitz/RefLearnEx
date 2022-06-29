# %% imports
import numpy as np
import plotly.graph_objects as go
from cartpole import CartPoleRegulatorEnv

# %% Prepare
cMax = 1.
env = CartPoleRegulatorEnv(cMax, mode="train")
env.reset()


def get_c_x(x):
    c_x_theta_min = env._getCost(x, 0)[1]
    c_x_theta_max = env._getCost(x, 0.25*np.pi)[1]

    return c_x_theta_min + c_x_theta_max


def get_c_theta(theta):
    c_theta_x_min = env._getCost(0., theta)[1]
    c_theta_x_max = env._getCost(4.8, theta)[1]

    return c_theta_x_min + c_theta_x_max


# %% create Data
x = np.arange(-4.8, 4.81, 0.01)
theta = np.arange(-12 * 2 * np.pi / 360,
                  12 * 2 * np.pi / 360 + 1e-4, 1e-4)

c_x = np.fromiter(map(get_c_x, x), np.float32)
c_theta = np.fromiter(map(get_c_theta, theta), np.float32)

# %% View
l_x = go.Scatter({"x": x,
                  "y": c_x,
                  "name": "cost Position",
                  "uid": "uid_l_c_x"
                  })
l_theta = go.Scatter({"x": theta,
                      "y": c_theta,
                      "name": "cost Angle",
                      "uid": "uid_l_c_theta"
                      })
go.Figure([l_x, l_theta], {'xaxis_title': "value",
                           'yaxis_title': "costs"})

# %%
