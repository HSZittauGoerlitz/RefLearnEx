# %% imports
import numpy as np
import plotly.graph_objects as go
from cartpole import CartPoleRegulatorEnv

# %% Prepare
cMax = 1.
env = CartPoleRegulatorEnv(cMax, mode="train")
env.reset()

# %% parameter
omega_x = 0.6
w_x = 0.01
o_x = -0.0075

omega_theta = 0.125*env.theta_success_range
w_theta = 0.02
o_theta = -0.0125


# %% create Data
x = np.arange(-4.8, 4.81, 0.01)
theta = np.arange(-12 * 2 * np.pi / 360,
                  12 * 2 * np.pi / 360 + 1e-4, 1e-4)

c_x = env._costSmooth(x, omega_x, w_x, o_x)
c_theta = env._costSmooth(theta, omega_theta, w_theta, o_theta)

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
