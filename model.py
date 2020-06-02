import numpy as np
from funcs import ProportionalDerivative as PropDer, g, dg


class Model:
    """ A generative model of a 2D environment with a 1DoF arm  controlled by
    a proportional-derivative dynamical system (velocity control).
    """

    def __init__(self, rng, mu=0, dmu=0, rho=None):
        """
        Args:
            rng: (np.random.RandomState) a random number generator
            mu: (float) initial central value of the distribution of the latent variable
            dmu: (float) initial derivative of the central value
        """

        self.rng = rng

        self.rho = rho   # target position

        # latentdistribution
        self.mu = mu   # central value of the distribution of the latent variable
        self.dmu = dmu    # derivative of the central value
        self.a = 0    # current angular velocity (action)

        # sensory
        # standard deviation of proprioceptive sensory state (joint position)
        self.sp_sigma = 0.001
        self.sv_sigma = 0.001*np.eye(2)    # covariance matrix of visual state (xy coordinates)
        self.inv_sv_sigma = np.linalg.inv(self.sv_sigma)    # inverse of sv_sigma
        self.sm_sigma = 0.001    # standard deviation of angle change (joint angular velocity)

        self.h = 0.001   # integration step (dt/decay)

        self.arm_length = 1

        # stores for sensory states and action
        self.sp = None
        self.sv = None
        self.da = 1

        # init state od dynamics
        self.f = np.zeros(2)

        # the dynamical model object
        self.dynamics = PropDer(k=1, phi=2)

    def generate(self):
        """ Generate a 'fake' sensory state from internal distributions

        Returns:
            generated sensory state: (float, float, float) joint angle, visual x, visual y
        """

        gstate = np.zeros(3)
        gstate[0] = self.sp_sigma*self.rng.randn() + self.mu
        gstate[1:] = np.random.multivariate_normal(
            self.arm_length*g(self.mu),
            self.sv_sigma)
        self.gstate = gstate
        return gstate

    def update(self, state):
        """ Update central values of latent distributions by minimizing
        the VARIATIONAL LAPLACE ENCODED FREE ENERGY via gradient descent.

        1) VARIATIONAL LAPLACE ENCODED FREE ENERGY

        F = -log p(s, mu) + Consts

        2) Facorized p(s, mu)

        p(s, mu) = p(sp|mu)*p(sv|mu)*p(dmu|mu, pho)

        Args:

            state: (float, float, float) joint angle, visual x, visual y

        Returns

            action: (float) updated angular velocity

        """

        # generate fake sensory states (useless at the moment)
        gstate = self.generate()

        # current state
        sp, sv = state[0], state[1:]
        self.sp = sp if self.sp is None else self.sp
        self.sv = sv if self.sv is None else self.sv

        df = self.dynamics(self.f, self.rho)
        self.f += self.h*df

        # modify central value of latent variable through gradient descent
        dmu = \
            + (sp - self.mu) / self.sp_sigma \
            + np.dot(np.dot(self.inv_sv_sigma, dg(self.mu)), (sv - g(self.mu))) \
            + self.sp_sigma * self.f[1] * (self.dmu - self.f[0])
        self.mu += self.dmu + self.h*dmu

        # modify first order of central value of latent variable through gradient descent
        ddmu = \
            self.sm_sigma * (self.f[0] - self.dmu)
        self.dmu += self.h*ddmu

        # modify action variable through gradient descent
        dsp = np.abs(sp - self.sp)
        dsv = np.abs(sv - self.sv)
        self.da = \
            - dsp * (sp - self.mu) / self.sp_sigma \
            - np.dot(np.dot(self.inv_sv_sigma, dsv), (sv - g(self.mu)))
        self.a += self.h*self.da

        # store state of the step as the previous state of next step
        self.sp, self.sv = sp, sv.copy()

        return self.a
