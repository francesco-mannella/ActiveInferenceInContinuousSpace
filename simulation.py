# %%
import numpy as np
from env import Env
from model import Model
from plotter import Plotter
from funcs import g


def simulation():

    real_mu = 0*np.pi

    model_mu = 0*np.pi
    model_rho = -0.35*np.pi
    stime = 50000

    rng = np.random.RandomState()
    plotter = Plotter(time_window=stime)

    # init the generative model (agent) and the generative process
    # (environment)
    gprocess = Env(rng)
    gmodel = Model(rng, mu=model_mu, rho=model_rho)

    state = gprocess.reset(mu=real_mu)
    for t in range(stime):

        # Update model via gradient descent and get action
        action = gmodel.update(state)

        # Generated fake sensory state from model
        gstate = gmodel.gstate

        # do action
        state = gprocess.step(action)

        # update plot every n steps
        plotter.append_mu(gprocess.mu, gmodel.mu)
        if t % 1000 == 0 or t == stime-1:
            plotter.sensed_arm.update(state[0], state[1:])
            plotter.real_arm.update(gprocess.istate[0], gprocess.istate[1:])
            plotter.generated_arm.update(gstate[0], gstate[1:])
            plotter.target_arm.update(gmodel.rho, g(gmodel.rho))
            plotter.update()

    input("Press any button to close.")


if __name__ == "__main__":
    simulation()
