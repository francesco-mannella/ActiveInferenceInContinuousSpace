import numpy as np
from funcs import ProportionalDerivative as PropDer, g


class Env:
    """ A 2D arm with 1DoF controlled by a proportional-derivative
    dynamical system (velocity control)
    """

    def __init__(self, rng):
        """
        Args:
            rng: (np.random.RandomState) a random number generator

        """
        self.rng = rng

        # latentdistribution
        self.mu = 0    # central value of the distribution of the latent variable
        self.dmu = 0    # derivative of the central value

        # sensory distributions
        # standard deviation of proprioceptive sensory state (joint position)
        self.sp_sigma = 0.011
        self.sv_sigma = 0.011   # standard deviation of visual state (xy coordinates)

        self.fh = 0.01   # dynamics integration step (dt/decay)

        self.arm_length = 1

        self.dynamics = PropDer(k=1, phi=2)

    def step(self, action):
        """ A step of the simulation

        Args:
            action: (float) Joint angle change

        Returns:
            sensory state: (float, float, float) joint angle, visual x, visual y

        """

        # update dynamics
        x = self.dynamics([self.mu, self.dmu], action)
        self.mu += self.fh*x[0]
        self.dmu += self.fh*x[1]

        # sensory state
        state = self.generateSensoryData()

        return state

    def generateSensoryData(self):
        """ Sensory state from latent distribution

        Returns:
            (float, float, float) joint angle, visual x, visual y
        """
        self.istate = [self.mu, *self.arm_length*g(self.mu)]

        state = self.rng.randn(3) * \
            [self.sp_sigma, self.sv_sigma, self.sv_sigma] + \
            self.istate

        return state

    def reset(self, mu=0, dmu=0):
        """ Reset variables

        Args:
            mu: (float) initial value of mu
            dmu: (float) initial value of derivative

        Returns:
            (float, float, float) joint angle, visual x, visual y

        """

        self.mu = mu
        self.dmu = dmu

        return self.generateSensoryData()
