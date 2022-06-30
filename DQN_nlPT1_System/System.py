from numba.experimental import jitclass
from numba import float32    # import the types
import numpy as np

# System and simulation
spec = [('uSmin', float32),
        ('uSmax', float32),
        ('yS', float32),
        ('uZ', float32),
        ('yZ', float32),
        ('K', float32),
        ('TS', float32),
        ('TZ', float32),
        ('aS', float32),
        ('bS', float32),
        ('aZ', float32),
        ('bZ', float32),
        ('dt', float32),
        ('probZ', float32),
        ]


@jitclass(spec)
class System():
    def __init__(self, yS0, TS=5.0, dt=0.1, probZ=0.001,
                 uSmin=-5.0, uSmax=5.0):
        # in-/outputs
        self.uSmin = uSmin
        self.uSmax = uSmax
        self.yS = yS0
        self.uZ = 0.0
        self.yZ = 0.0
        # System parameter
        self._update_K()
        self.TS = TS
        self.TZ = 10.0 * TS
        # simulation
        if dt > 0:
            self.dt = dt
        else:
            raise ValueError("The Step-Time must be greater than 0")
        self.aS = np.exp(-self.dt/self.TS)
        self.aZ = np.exp(-self.dt/self.TZ)
        self._update_bS()
        self.bZ = 1 - self.aZ
        # disturbance
        if (probZ >= 0) & (probZ <= 1):
            self.probZ = probZ
        else:
            raise ValueError("The disturbance probability must be between "
                             "0 and 1")

    def _check_uS(self, uS):
        if uS < self.uSmin:
            return self.uSmin
        elif uS > self.uSmax:
            return self.uSmax
        else:
            return uS

    def _update_bS(self):
        self.bS = self.K * (1 - self.aS)

    def _update_K(self):
        self.K = 0.1 * self.yS + 2.0

    def getEq(self, uS, steps=1000):
        # let the system settle
        for i in range(self.TS / self.dt * 5):
            self.step(uS)

        # simulate system and get mean yS (according to disturbance)
        y = 0
        for i in range(steps):
            self.step(uS)
            y += self.yS

        return y / float(steps)

    def reset(self):
        self.yS = 0.0
        self.yZ = 0.0
        self._update_K()
        self._update_bS()
        self.uZ = 0.0

    def shuffle_u(self):
        return np.round(self.uSmin + (self.uSmax - self.uSmin) *
                        np.random.random(), 1)

    def step(self, uS):
        # calculate disturbance
        if np.random.random() < self.probZ:
            self.uZ = np.round(-2 + 4*np.random.random(), 1)
        self.yZ = self.uZ * self.bZ + self.aZ * self.yZ
        # calculate plant
        uS = self._check_uS(uS)
        self._update_K()
        self._update_bS()
        self.yS = uS * self.bS + self.aS * self.yS

        return self.yZ + self.yS
