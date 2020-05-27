import numpy as np
import matplotlib.pyplot as plt

class ProportionalDerivative:
    """ Simple proportional-derivative control
    """

    def __init__(self, k=1, phi=0.1, m=1):
        """
        Args:
            k: (float) proportional amplitude 
            phi: (float) derivative amplitude 
            m: (float) mass of the controlled object 
        """
        
        self.m = m
        self.k = k
        self.phi = phi
        
        self.h = 0.01

    def __call__(self, x, a):
        """ Compute the change of px and dx
        
        Args:
            x: np.array(float, float) pair containing the variable (px) 
                and its first order derivative (dx)
            a: target value of px
        """

        px, dx = x

        # PD
        px += self.h*dx
        dx += self.h*(a - self.k*px - self.phi*dx)/self.m

        return np.array([px, dx])

if __name__ == "__main__":

    xx = []
    T = 5000

    pd = ProportionalDerivative(k=1, phi=1.5, m=1.)

    x = np.array([0.4, -0.8])
    for t in range(T):
        a = .2
        x = pd(x, a)
        print(x.T)
        xx.append(x.T)

    xx = np.vstack(xx)
    plt.plot(*xx.T)
    plt.xlim([-0.1, 0.5])
    plt.ylim([-0.8,0.8])
    plt.show()
