import numpy as np

# angle to visual forward model


def g(mu):
    """ Converts joint angle into ccordinates

    Returns:
        (float, float) x,y coordinates
    """
    return np.array([np.cos(mu), np.sin(mu)])


def dg(mu):
    return np.array([-np.sin(mu), np.cos(mu)])

# PD controller


class ProportionalDerivative:
    """ Simple proportional-derivative-like control
    """

    def __init__(self, k=.1, phi=0.01, m=1):
        """
        Args:
            k: (float) proportional amplitude
            phi: (float) derivative amplitude
            m: (float) mass of the controlled object
        """

        self.m = m
        self.k = k
        self.phi = phi

    def __call__(self, x, a):
        """ Compute the change of px and dx

        Args:
            x: np.array(float, float) pair containing the variable (px)
                and its first order derivative (dx)
            a: (float) target value of px
        """

        px, dx = x

        # PD
        px, dx = dx, (a - self.k*px - self.phi*dx)/self.m

        return np.array([px, dx])

# TEST -----------------------------------------------------------------------


if __name__ == "__main__":

    # Test PD controller
 
    import matplotlib.pyplot as plt

    store = []
    T = 500

    pd = ProportionalDerivative(k=1, phi=4, m=1.)

    # simulate
    x0 = np.array([0.0, 0.0])
    x = x0.copy()
    a = 0.25*np.pi
    for t in range(T):
        x += 0.1*pd(x, a)
        store.append(x.copy().T)
    store = np.vstack(store)

    # render simulation
    plt.ion()
    x = x0[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    target, = ax.plot([0, np.cos(a)], [0, np.sin(a)], lw=3, color="red")
    armBase = ax.scatter(0, 0, s=200, color="blue")
    arm, = ax.plot([0, np.cos(x)], [0, np.sin(x)], lw=6, color="blue")
    ax.set_xlim([-.2, 1.2])
    ax.set_ylim([-.2, 1.2])
    for t in range(T):
        x = store[t, 0]
        arm.set_data([0, np.cos(x)], [0, np.sin(x)])
        plt.pause(0.01)
    #

    input()
