# Active Inference In Continuous Space

A 1Dof 2D arm controlled through active inference.

* ***env.py***: implements the generative process (the real arm and its dynamics)
* ***model.py***: implements the generative model (internal model of arm  and dynamics)
* ***simulation.py***: a test with the main loop of iteration. In the demo the initial values of the target proprioceptive position and the real position are equivalent. The central value of the internal model of the position is different.
* ***funcs.py***: Utility functions as the forward model g(mu) from the proprioceptive state to the 2D coordinates of the hand, and the dynamical system (a proportional-derivative controller)
* ***plotter.py***: a collection of objects for plotting

### References

[Is this my body? (Part I)](https://msrmblog.github.io/is-this-my-body-1/) by Pablo Lanillos

[An active inference implementation of phototaxis](https://www.mitpressjournals.org/doi/pdfplus/10.1162/isal_a_011) by Manuel Baltieri and Christopher L. Buckley
