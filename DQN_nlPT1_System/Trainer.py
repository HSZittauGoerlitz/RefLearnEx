import numpy as np
from plotly.subplots import make_subplots
from plotly import graph_objects as go
import random
import time
from DQNagents import DQNagent, DOE


class DQNtrainer():
    def __init__(self, nBatch, batchSize, targetUpdate,
                 eps0, epsMin, epsDecay, cMax, mu, system,
                 memCapacity):
        self.nBatch = nBatch
        self.batchSize = batchSize

        self.cMax = cMax
        self.mu = mu

        self.Epoch = 0
        self.nEpochVis = 5000

        self.agent = DQNagent(6, 7, memCapacity, batchSize,
                              eps0, epsMin, epsDecay, targetUpdate)
        self.actuator = DOE(system.uSmin, system.uSmax)

        self.A_CONV = [system.uSmin * 0.5, system.uSmin * 0.1,
                       system.uSmin * 0.05,
                       0.,
                       system.uSmax * 0.05,
                       system.uSmax * 0.1, system.uSmax * 0.5]

        self._initBatchVis(system)
        self._initTrainVis()

    def _getCosts(self, w, y):
        e = np.abs(w-y)
        omega = np.tanh(np.sqrt(0.95) / self.mu)

        return np.tanh(e*omega)**2. * self.cMax

    def _getReward(self, w, y, w_last, y_last):
        diff = abs(w_last - y_last) - abs(w - y)
        if diff > 0.:
            # 5. as default scaling
            return self.cMax * 5. * diff
        else:
            return 0.

    def _getState(self, w, uS, y, system, actuator):
        return np.array([w, y, system.yS, uS, system.uZ, actuator.state])

    def _initBatchVis(self, system):
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        fig.update_xaxes(title_text="time in s", row=2, col=1)
        fig.update_yaxes(row=1, col=1,
                         title_text="plant values", range=[-6., 6.])
        fig.update_yaxes(row=2, col=1,
                         title_text="costs", range=[0., self.cMax])

        # data
        self.time = np.arange(self.nBatch) * system.dt
        self.t_w = np.array([self.time[0], self.time[-1]])
        self.t_wRange = np.array([self.time[0], self.time[-1],
                                  self.time[-1], self.time[0]])
        self.w = np.zeros(2)
        self.wRange = np.array([self.mu, self.mu, -self.mu, -self.mu])
        self.y = np.zeros(self.nBatch)
        self.c = np.zeros(self.nBatch)

        # line objects
        lineW = go.Scatter({"x": self.t_w,
                            "y": self.w,
                            "name": "w",
                            "uid": "uid_lineW",
                            "yaxis": "y1",
                            "line": {"color": "#4C78A8",
                                     "dash": "dash",
                                     "width": 1
                                     },
                            "mode": 'lines'
                            })
        lineWRange = go.Scatter(x=self.t_wRange,
                                y=self.wRange,
                                fill='toself',
                                name="wRange",
                                uid="uid_lineWRange",
                                yaxis="y1",
                                fillcolor="rgba(76,120,168,0.2)",
                                line_color="rgba(76,120,168,0.0)",
                                showlegend=False
                                )
        lineY = go.Scatter({"x": self.time,
                            "y": self.y,
                            "name": "y",
                            "uid": "uid_lineY",
                            "yaxis": "y1",
                            "line": {"color": "#4C78A8",
                                     "width": 1
                                     }
                            })
        lineC = go.Scatter({"x": self.time,
                            "y": self.c,
                            "name": "c",
                            "uid": "uid_lineC",
                            "yaxis": "y1",
                            "line": {"color": "#000000",
                                     "width": 1
                                     }
                            })

        fig.update_layout(title='', height=800)
        # add lines to plots
        fig.add_trace(lineW, row=1, col=1)
        fig.add_trace(lineWRange, row=1, col=1)
        fig.add_trace(lineY, row=1, col=1)
        fig.add_trace(lineC, row=2, col=1)
        # create widget
        self.batchVis = go.FigureWidget(fig)

    def _initTrainVis(self):
        self.xEpochs = np.arange(self.nEpochVis)
        self.cEnd = np.zeros(self.nEpochVis)
        self.loss = np.zeros(self.nEpochVis)

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.02)
        fig.update_xaxes(title_text="Number of Epoch", row=2, col=1)
        fig.update_yaxes(row=1, col=1, title_text="Costs",
                         range=[0., self.nBatch * self.cMax])
        fig.update_yaxes(row=2, col=1, title_text="ANN Loss", autorange=True)

        lineCostsEnd = go.Scatter({"x": self.xEpochs,
                                   "y": self.cEnd,
                                   "name": "costs",
                                   "opacity": 0.25,
                                   "uid": "uid_rEndLine",
                                   "yaxis": "y1",
                                   "line": {"color": "#000000",
                                            "width": 1
                                            }
                                   })
        lineCostsEndM = go.Scatter({"x": self.xEpochs,
                                    "y": self.cEnd,
                                    "name": "costs mean",
                                    "uid": "uid_rEndLine",
                                    "yaxis": "y1",
                                    "line": {"color": "#000000",
                                             "width": 1
                                             }
                                    })
        lineCostsStdU = go.Scatter({"x": self.xEpochs,
                                    "y": self.cEnd,
                                    "name": "costs Std U",
                                    "uid": "uid_rStdULine",
                                    "yaxis": "y1",
                                    "line": {"color": "#000000",
                                             "width": 0.5,
                                             "dash": "dash",
                                             },
                                    "showlegend": False
                                    })
        lineCostsStdL = go.Scatter({"x": self.xEpochs,
                                    "y": self.cEnd,
                                    "name": "costs Std L",
                                    "uid": "uid_rStdLLine",
                                    "yaxis": "y1",
                                    "line": {"color": "#000000",
                                             "width": 0.5,
                                             "dash": "dash",
                                             },
                                    "showlegend": False
                                    })
        lineLoss = go.Scatter({"x": self.xEpochs,
                               "y": self.loss,
                               "name": "loss",
                               "uid": "uid_lineLoss",
                               "yaxis": "y1",
                               "line": {"color": "#D83C20",
                                        "width": 1
                                        }
                               })
        fig.update_layout(title='', height=800)
        fig.add_trace(lineCostsEnd, row=1, col=1)
        fig.add_trace(lineCostsEndM, row=1, col=1)
        fig.add_trace(lineCostsStdU, row=1, col=1)
        fig.add_trace(lineCostsStdL, row=1, col=1)
        fig.add_trace(lineLoss, row=2, col=1)
        # create widget
        self.trainVis = go.FigureWidget(fig)

    def reset(self):
        self.Epoch = 0
        self.agent.reset()
        self.xEpochs = np.arange(self.nEpochVis)
        self.cEnd *= 0.
        self.loss *= 0.

    def train(self, system, btnRun, btnStep, btnEnd, btnReset):
        wStep = 5. if random.random() < 0.5 else -5.

        cMean = np.zeros_like(self.cEnd)
        cStdU = np.zeros_like(self.cEnd)
        cStdL = np.zeros_like(self.cEnd)

        while btnEnd.state:
            if btnReset.state:
                self.reset()
                btnReset.state = False
                cMean *= 0.
                cStdU *= 0.
                cStdL *= 0.

            if btnRun.state or btnStep.state:
                if btnStep.state:
                    btnStep.state = False
                # Initialize the environment and state
                system.reset()
                self.actuator.reset()
                uStep = 0.
                if random.random() < 0.5:
                    wStep = -wStep
                costs = 0
                # get init state
                NNstate = self._getState(wStep, uStep, 0,
                                         system, self.actuator)

                cEpoch = 0.
                for batch in range(self.nBatch):
                    aIdx = self.agent.act(NNstate)
                    uStep = self.actuator.step(self.A_CONV[aIdx])
                    yStep = system.step(uStep)

                    # get immediate costs
                    costs = self._getCosts(wStep, yStep)
                    reward = self._getReward(wStep, yStep,
                                             NNstate[0], NNstate[1])
                    costs -= reward
                    if costs < 0.:
                        costs = 0.

                    cEpoch += costs

                    # get new step in state
                    NNstateNext = self._getState(wStep, uStep, yStep,
                                                 system, self.actuator)

                    if batch + 1 < self.nBatch:
                        # Store the transition in memory
                        self.agent.remember(NNstate, aIdx, costs,
                                            NNstateNext, 0.)
                        # Train model ANN
                        self.agent.replay(self.Epoch)
                        # save state for next iteration
                        NNstate = np.copy(NNstateNext)

                    # update vis data
                    self.y[batch] = yStep
                    self.c[batch] = costs

                self.agent.remember(NNstate, aIdx, costs, NNstateNext, 1.)

                # train agent model
                if self.Epoch < self.nEpochVis:
                    self.loss[self.Epoch], _ = self.agent.replay(self.Epoch,
                                                                 True,
                                                                 self.Epoch ==
                                                                 1000)
                    self.cEnd[self.Epoch] = cEpoch
                    idxStart = self.Epoch - 50
                    cMean[self.Epoch] = self.cEnd[idxStart:self.Epoch].mean()
                    cStd = self.cEnd[idxStart:self.Epoch].std()
                    cStdU[self.Epoch] = cMean[self.Epoch] + cStd
                    cStdL[self.Epoch] = cMean[self.Epoch] - cStd
                else:
                    self.xEpochs[:-1] = self.xEpochs[1:]
                    self.xEpochs[-1] += 1
                    self.loss[:-1] = self.loss[1:]
                    self.loss[-1], _ = self.agent.replay(self.Epoch,
                                                         True, False)
                    self.cEnd[:-1] = self.cEnd[1:]
                    self.cEnd[-1] = cEpoch
                    idxStart = self.nEpochVis - 50
                    cMean[:-1] = cMean[1:]
                    cMean[-1] = self.cEnd[idxStart:].mean()
                    cStd = self.cEnd[idxStart:].std()
                    cStdU[:-1] = cStdU[1:]
                    cStdU[-1] = cMean[-1] + cStd
                    cStdL[:-1] = cStdL[1:]
                    cStdL[-1] = cMean[-1] - cStd

                # update visualisation
                with self.batchVis.batch_update():
                    self.batchVis.data[0].x = self.t_w
                    self.batchVis.data[1].x = self.t_wRange
                    self.batchVis.data[2].x = self.time
                    self.batchVis.data[3].x = self.time
                    self.batchVis.data[0].y = self.w + wStep
                    self.batchVis.data[1].y = self.wRange + wStep
                    self.batchVis.data[2].y = self.y
                    self.batchVis.data[3].y = self.c
                    self.batchVis.layout['title'] = "Epoch: {}".format(
                                                      self.Epoch)
                # reset vis data
                self.y *= 0
                self.w *= 0
                self.c *= 0

                with self.trainVis.batch_update():
                    self.trainVis.data[0].x = self.xEpochs
                    self.trainVis.data[1].x = self.xEpochs
                    self.trainVis.data[2].x = self.xEpochs
                    self.trainVis.data[3].x = self.xEpochs
                    self.trainVis.data[4].x = self.xEpochs
                    self.trainVis.data[0].y = self.cEnd
                    self.trainVis.data[1].y = cMean
                    self.trainVis.data[2].y = cStdU
                    self.trainVis.data[3].y = cStdL
                    self.trainVis.data[4].y = self.loss

                self.Epoch += 1
            else:
                time.sleep(0.1)
