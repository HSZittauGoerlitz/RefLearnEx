"""
Modified version of classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
# flake8: noqa
import math
from time import sleep
from typing import Callable, List, Tuple

import gym
from gym.envs.classic_control.rendering import Viewer
import numpy as np
from gym import spaces
from gym.utils import seeding

from pyglet.window import mouse

class CartPoleRegulatorEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, cMax, cFix=-0.001, mode="train"):
        self.gravity = 9.8
        self.masscart = 1.
        self.masspole = 0.05
        self.total_mass = self.masspole + self.masscart
        self.length = 1.  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.low = [-2.4, -10., -np.pi, -0.1]
        self.high = [2.4, 10., np.pi, 0.1]
        self.render_sleep = 0.01

        # Failure state description
        self.x_threshold = 4.8
        self.theta_threshold_radians = math.pi / 2

        self.cMax = cMax
        self.cFix = cFix

        if self.cFix > 0:
            print("The fixed cost value is greater than 0, "
                  "this will punish longer Transitions.")

        self.action_space = spaces.Discrete(2)

        self.seed()
        self.viewer = None
        self.state = None
        self.episode_step = 0

        self.observation_space = spaces.Box(
          np.array([-4.8000002e+00, -3.4028235e+38,
                    -2.*np.pi, -3.4028235e+38]),
          np.array([4.8000002e+00, 3.4028235e+38,
                    2.*np.pi, 3.4028235e+38]),
          (4,), np.float64)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_next_state(self, action, push=False):
        x, x_dot, theta, theta_dot = self.state
        if push:
            force = self.force_mag * action
        else:
            force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # limit position to observation space
        if x < -4.8000002e+00:
            x = -4.8000002e+00
        elif x > 4.8000002e+00:
            x = 4.8000002e+00

        # limit theta to 2pi range
        if theta > 2*np.pi:
            theta -= 2*np.pi
        elif theta < -2*np.pi:
            theta += 2*np.pi

        return np.array([x, x_dot, theta, theta_dot])

    def _costSmooth(self, e, omega, w, offset):
        return np.tanh(e/omega)**2 * w + offset

    def _getCost(self, x, theta):
        e_x = np.abs(0. - x)
        e_theta = np.abs(0. - theta)

        if e_x > self.x_threshold:
            return True, self.cMax

        c_x = self._costSmooth(e_x, 0.6, 0.04, -0.04)
        c_theta = self._costSmooth(e_theta, 0.05, 0.06, -0.06)

        return False, c_x + c_theta + self.cFix

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        self.state = self._compute_next_state(action)
        x, _, theta, _ = self.state

        self.episode_step += 1

        done, cost = self._getCost(x, theta)

        if ~isinstance(cost, float):
            cost = 0

        return [self.state, cost, done, {}]

    def reset(self):
        self.state = self.np_random.uniform(
          low=self.low, high=self.high,
          size=self.observation_space.shape)
        self.episode_step = 0

        return self.state

    def render(self, mode="human"):
        screen_width = 1200
        screen_height = 800

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 20.0
        polelen = scale * (2 * self.length)
        cartwidth = 100.0
        cartheight = 60.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

            # interaction function for window handlers
            def onClick(x, y, dx, dy, buttons, modifiers):
                if buttons & mouse.LEFT:
                    cw_half = cartwidth * 0.5
                    ch_half = cartheight * 0.5
                    if ((x > self.carttrans.translation[0] - cw_half) and
                        (x < self.carttrans.translation[0] + cw_half) and
                        (y > self.carttrans.translation[1] - ch_half) and
                        (y < self.carttrans.translation[1] + ch_half)):
                        action = max(min(dx * 0.1, 1.5), -1.5)
                        self.state = self._compute_next_state(self.state,
                                                              action, True)

            self.viewer.window.on_mouse_drag = onClick

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        sleep(self.render_sleep)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
