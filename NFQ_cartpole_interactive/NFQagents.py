from collections import namedtuple
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


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

    def updateCapacity(self, capacity):
        self.capacity = capacity
        self.position = len(self.memory) % self.capacity

    def __len__(self):
        return len(self.memory)


# Network variants
class NFQnet(nn.Module):
    def __init__(self, nState, nActions):
        """ Init DQN network

        Args:
            nState (int): Number of State Variables
            nActions (int): Number of of possible controller actions
        """
        super(NFQnet, self).__init__()
        self.H1Layer = nn.Linear(nState, 5)
        self.H2Layer = nn.Linear(5, 5)
        self.outLayer = nn.Linear(5, nActions)

    def forward(self, xb):
        xb = torch.tanh(self.H1Layer(xb))
        xb = torch.sigmoid(self.H2Layer(xb))
        return torch.sigmoid(self.outLayer(xb))


# Agent(s)
class NFQagent():
    def __init__(self, nState, nActions, QbMin, QbMax, epsilon, logger=None):
        """ Init DQN Agent

        Arguments:
            nState {int} -- Number of State Variables
            nActions {int} -- Number of possible actions
            capacity {int} -- Size of process memory
            batchSize {int} -- Number of samples used from memory
                               for each training
            QbMin {float} -- Lower bound for Q Values
            QbMax {float} -- Upper bound for Q Values
            epsilon {float} -- Exploration parameter

        Raises:
            ValueError: Reports erroneous inputs
        """
        # TODO: Catch erroneous inputs
        self.actionSize = nActions
        self.stateSize = nState

        # input state variables are symmetric to 0
        # -> no mean is necessary
        self.std = 1.

        if QbMax < QbMin:
            raise ValueError("QbMax must be greater than QbMin")

        self.QbMin = QbMin
        self.QbMax = QbMax

        # discount of future reward
        self.gamma = 0.95

        # Exploration parameter
        self.Epsilon = epsilon

        # Model
        self.model = NFQnet(nState, nActions)
        self.optimizer = optim.Rprop(self.model.parameters())
        self.criterion = nn.SmoothL1Loss()
        for param in self.model.parameters():
            torch.nn.init.normal_(param, 0., 1.)

        self.logger = logger

    def _checkModelFileName(self, fName):
        # remove file ending
        temp = fName.split(".")
        if len(temp) > 1:
            fName = temp[0]

        return fName

    def _getQbounds(self, Q):
        return (Q.min(), Q.max())

    def _getQtarget(self, state):
        """ Evaluate Q-Network and scale results between Q-bounds

        Since the NN uses sigmoid activation for output layer the network
        results are between 0 and 1.
        """
        with torch.no_grad():
            Q = self.model(state)

        return (self.QbMax - self.QbMin) * Q + self.QbMin

    def act(self, state):
        """ Act with epsilon-greedy exploration
        """
        if np.random.random() > self.Epsilon:
            with torch.no_grad():
                Q = self.model(torch.FloatTensor(state / self.std)).flatten()
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

    def batchTrainModel(self, memory, batchSize):
        batch = Transition(*zip(*memory.sample(batchSize)))

        state = torch.cat(batch.state).reshape(-1, self.stateSize)
        action = torch.reshape(torch.LongTensor(batch.action),
                               (batchSize, -1))
        nextState = torch.cat(batch.nextState).reshape(-1, self.stateSize)
        costs = torch.FloatTensor(batch.costs)
        done = torch.FloatTensor(batch.done)

        # scale data to get a std of 1
        state = state / self.std
        nextState = nextState / self.std

        # Calculate future costs as training target
        # actual costs + discounted future costs
        Qtarget = self._getQtarget(nextState)
        # use min Q values for target estimation
        Q = costs + self.gamma * Qtarget.min(axis=1).values * (1. - done)
        # prevent degradation of Q-Function
        Qbounds = self._getQbounds(Q)
        Q -= (Qbounds[0] - self.QbMin)

        # estimate future costs from actual state -> test current NN
        # train only Q values for known actions
        Qest = self.model(state).gather(1, action)

        # Scale Qest to [QbMin, QbMax], due output activation (sigmoid)
        Qest = (self.QbMax - self.QbMin) * Qest + self.QbMin

        if self.logger:
            self.logger.info(f"Training Data Cost stats - mean: "
                             f"{costs.mean()}, min: {costs.min()}, "
                             f"max: {costs.max()}, std: f{costs.std()}")
            self.logger.info(f"Q estimation stats - mean: {Qest.mean()}, "
                             f"min: {Qest.min()}, max: {Qest.max()}, "
                             f"std: f{Qest.std()}")
        # NN Training
        # Compute loss
        loss = self.criterion(Qest, Q.reshape(action.size()))
        self.optimizer.zero_grad()
        loss.backward()
        # update model
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1e2, 1e2)
        self.optimizer.step()

        return (loss.item(), Qbounds)

    def loadModel(self, fName="PreTrainedAgent"):
        fName = self._checkModelFileName(fName)
        self.model = torch.load(f"{fName}.pt")
        self.std = torch.load(f"{fName}_Stats.pt")

    def saveModel(self, fName="PreTrainedAgent"):
        fName = self._checkModelFileName(fName)
        torch.save(self.model, f"{fName}.pt")
        torch.save(self.std, f"{fName}_Stats.pt")

    def updateStateStats(self, memory):
        state = torch.cat(Transition(*zip(*memory.memory))
                          .state
                          ).reshape(-1, self.stateSize)
        # get data properties to rescale
        self.std = state.std(0, keepdim=True).flatten().numpy()
        self.std[self.std == 0] = 1.
