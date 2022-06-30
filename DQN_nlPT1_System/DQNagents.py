from collections import namedtuple
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from numba.experimental import jitclass
from numba import float32    # import the types


# Preparing object to manage agents memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'nextState', 'costs', 'done'))


class ReplayMemory(object):
    """ Source: pytorch DQN tutorial """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Network variants
class DQN(nn.Module):
    def __init__(self, nState, nActions):
        """ Init DQN network

        Args:
            nState (int): Number of State Variables
            nActions (int): Number of of possible controller actions
        """
        super(DQN, self).__init__()
        self.inLayer = nn.Linear(nState, 20)
        self.H1Layer = nn.Linear(20, 20)
        self.H2Layer = nn.Linear(20, nActions)
        self.outLayer = nn.Linear(nActions, nActions)

    def forward(self, xb):
        xb = self.inLayer(xb)
        xb = torch.sigmoid(self.H1Layer(xb))
        xb = torch.sigmoid(self.H2Layer(xb))
        return self.outLayer(xb)


# Agent(s)
class DQNagent():
    def __init__(self, nState, nActions, capacity, batchSize,
                 epsStart, epsMin, epsDecay, targetUpdate):
        """ Init DQN Agent

        Arguments:
            nState {int} -- Number of State Variables
            nActions {int} -- Number of possible actions
            capacity {int} -- Size of process memory
            batchSize {int} -- Number of samples used from memory
                               for each training
            epsStart {float} -- Initial value for Epsilon
            epsMin {float} -- Minimum value for Epsilon
            epsDecay {int} -- Strength of Epsilon decay
                              (after this number of Epochs Epsilon is
                               approx 0.5 of Start-End)
            targetUpdate {int} -- Rate of Epochs to update target model

        Raises:
            ValueError: Reports erroneous inputs
        """
        # TODO: Catch erroneous inputs
        self.stateSize = nState
        self.actionSize = nActions

        self.mean = 0.
        self.std = 1.

        # discount of future reward
        self.gamma = 0.95

        # Training parameter (exploration)
        self.EpsilonStart = epsStart
        self.Epsilon = self.EpsilonStart
        self.EpsilonDecay = epsDecay
        self.EpsilonEnd = epsMin

        self.targetUpdate = targetUpdate

        # Memory and Model
        self.batchSize = batchSize
        self.memory = ReplayMemory(capacity)
        self.model = DQN(nState, nActions)
        # use second model for optimisation target calculation
        # this model is updated slower to stabilise learning
        # (Source: PyTorch DQN Tutorial)
        self.targetModel = DQN(nState, nActions)
        self.targetModel.load_state_dict(self.model.state_dict().copy())
        self.targetModel.eval()
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=0.00025,
                                       momentum=0.95,
                                       alpha=0.95,
                                       eps=0.01
                                       )
        self.criterion = nn.SmoothL1Loss()

    def _getQbounds(self, Q):
        return (Q.min(), Q.max())

    def _getQtarget(self, state):
        """ Evaluate Q-Network
        """
        with torch.no_grad():
            return self.targetModel(torch.FloatTensor(state))

    def _updateStateStats(self):
        state = torch.cat(Transition(*zip(*self.memory.memory))
                          .state
                          ).reshape(-1, self.stateSize)
        # get data properties to rescale
        self.mean = state.mean(0, keepdim=True).flatten().numpy()
        self.std = state.std(0, keepdim=True).flatten().numpy()
        self.std[self.std == 0] = 1.

    def _updateTargetModel(self):
        self.targetModel.load_state_dict(
          self.model.state_dict().copy())
        self.targetModel.eval()

    def act(self, state):
        """ Act with epsilon-greedy exploration
        """
        if np.random.random() > self.Epsilon:
            with torch.no_grad():
                Q = self.model(torch.FloatTensor((state - self.mean) /
                                                 self.std)).flatten()
                Qmin = Q.min(0)
                QminValue = Qmin[0]
                QeqMin = Q == QminValue
                nMin = QeqMin.sum()
                if nMin > 1:
                    return (QeqMin
                            .nonzero(as_tuple=False)
                            [np.random.randint(nMin)].item()
                            )
                else:
                    return Qmin[1].item()
        else:
            return np.random.randint(self.actionSize)

    def batchTrainModel(self):
        batch = Transition(*zip(*self.memory.sample(self.batchSize)))

        state = torch.cat(batch.state).reshape(-1, self.stateSize)
        action = torch.reshape(torch.LongTensor(batch.action),
                               (self.batchSize, -1))
        nextState = torch.cat(batch.nextState).reshape(-1, self.stateSize)
        costs = torch.FloatTensor(batch.costs)
        done = torch.FloatTensor(batch.done)

        state = (state - self.mean) / self.std
        nextState = (nextState - self.mean) / self.std

        # Calculate future costs as training target
        # actual costs + discounted future costs
        Qtarget = self._getQtarget(nextState)
        # use min Q values for target estimation
        # no future cost estimation for finished episodes
        Q = costs + self.gamma * Qtarget.min(axis=1).values * (1. - done)
        Q[Q < 0.] = 0.
        Qbounds = self._getQbounds(Q)

        # estimate future costs from actual state -> test current NN
        # train only Q values for known actions
        Qest = self.model(state).gather(1, action)

        # NN Training
        # Compute loss
        loss = self.criterion(Qest, Q.reshape(action.size()))
        self.optimizer.zero_grad()
        loss.backward()
        # update model
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return (loss.item(), Qbounds)

    def callModel(self, state):
        with torch.no_grad():
            Q = self.model(torch.FloatTensor((state - self.mean) /
                                             self.std)).numpy()
            return (self.QbMax - self.QbMin) * Q + self.QbMin

    def loadModel(self, fName='agentModel', loc=''):
        self.model.load_state_dict(torch.load(loc + fName + '.pt'))
        self._updateTargetModel()

    def remember(self, state, action, costs, nextState, done):
        self.memory.push(torch.FloatTensor(state),
                         action,
                         torch.FloatTensor(nextState),
                         costs, done)

    def replay(self, nEpoch, updateEpsilon=False, updateStats=False):
        """Training of agent model to generalise memory
            - Train all model outputs, update the output corresponding to
              the memory action by bellman equation (with costs of memory)

        Args:
            nEpoch {int} -- Epoch number of agent training

        Keyword Arguments:
            updateEpsilon {bool} -- When True Epsilon is recalculated
                                    (default: {False})
            updateStats {bool} -- When True state mean and std are updated
                                  (default: {False})

        Returns:
            (float, (float, float)) -- Loss of ANN training and
                                       spread of Q Values
        """
        if len(self.memory) < self.batchSize:
            return (None, (None, None))

        loss, Qbounds = self.batchTrainModel()

        if updateEpsilon:
            self.Epsilon = (self.EpsilonEnd +
                            (self.EpsilonStart - self.EpsilonEnd) *
                            math.exp(-1 * nEpoch / self.EpsilonDecay)
                            )

        if nEpoch % self.targetUpdate == 0:
            self._updateTargetModel()

        if updateStats:
            self._updateStateStats()

        return (loss, Qbounds)

    def reset(self):
        for param in self.model.parameters():
            torch.nn.init.normal_(param, 0., 1.)
        for param in self.targetModel.parameters():
            torch.nn.init.normal_(param, 0., 1.)

        self.Epsilon = self.EpsilonStart

        self.memory.memory = []
        self.memory.position = 0

    def saveModel(self, fName='agentModel', loc=''):
        torch.save(self.model.state_dict(), loc + fName + '.pt')


@jitclass([('min', float32), ('max', float32), ('state', float32)])
class DOE():
    """ Dynamic output element for generating continuous actuations

    As dynamic an integral property is used, as proposed by Hafner
    """
    def __init__(self, min, max) -> None:
        """ Init DOE

        Arguments:
            min {float} -- Min. possible state
            max {float} -- Max. possible state
        """
        if min < max:
            self.min = min
            self.max = max
        else:
            raise ValueError("Min. value must be smaller than max. value")

        self.state = 0.

    def reset(self):
        self.state = 0.0

    def step(self, u):
        """ Update state by

        state += u

        Arguments:
            u {float} -- Input for state update
        """
        self.state += u

        if self.state < self.min:
            self.state = self.min
        elif self.state > self.max:
            self.state = self.max

        return self.state
