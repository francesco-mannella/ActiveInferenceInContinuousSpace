# Active Inference In Continuous Space

A 1Dof 2D arm controlled through active inference.

* ***[env.py](env.py)***: implements the generative process (the real arm and its dynamics)
* ***[model.py](model.py)***: implements the generative model (internal model of arm  and dynamics)
* ***[simulation.py](simulation.py)***: a test with the main loop of iteration. At each timestep, first the model is updated based on the current sensory state. Then the environment (the generative process is updated based on the action returned by the model.
* ***[funcs.py](funcs.py)***: Utility functions as the forward model g(mu) from the proprioceptive state to the 2D coordinates of the hand, and the dynamical system (a proportional-derivative controller)
* ***[plotter.py](plotter.py)***: a collection of objects for plotting
* ***[Gradients.ipynb](Gradients.ipynb)***: A notebook describing the derivation of gradients from the free energy equation

### References

[Is this my body? (Part I)](https://msrmblog.github.io/is-this-my-body-1/) by Pablo Lanillos

[An active inference implementation of phototaxis](https://www.mitpressjournals.org/doi/pdfplus/10.1162/isal_a_011) by Manuel Baltieri and Christopher L. Buckley
