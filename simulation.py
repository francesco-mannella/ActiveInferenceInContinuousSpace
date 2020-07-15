# %% Headings
import numpy as np
from env import Env
from model import Model
from plotter import Plotter
from funcs import g

# %% Constants

real_mu = 0*np.pi
model_mu = 0*np.pi
model_rho = -0.35*np.pi
stime = 20000

rng = np.random.RandomState()

# %% Initializing the simulation objects for the first simulation

gprocess_sigmas = 0.2, 0.2, 0.2
gmodel_sigmas = 0.1, 0.15, 0.2

title = "gp\_sigma=%6.4f gm\_sigma=%6.4f"
for gprocess_sigma, gmodel_sigma in zip(gprocess_sigmas, gmodel_sigmas):

    plotter = Plotter(
        time_window=stime,
        title=title % (gprocess_sigma, gmodel_sigma))

    # init the generative model (agent) and the generative process
    # (environment)
    gprocess = Env(rng)
    gmodel = Model(rng, mu=model_mu, rho=model_rho)

    gprocess.set_sigma(gprocess_sigma)
    gmodel.set_sigma(gmodel_sigma)



    # %%  Iteration Loop
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
