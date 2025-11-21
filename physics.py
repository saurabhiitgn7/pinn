import numpy as np

def oscillator_solution(d, w0, t):
    """
    Analytical solution to the Under-damped Harmonic Oscillator.
    Equation: u(t) = e^(-d*t) * cos(w * t)
    where w = sqrt(w0^2 - d^2)
    """
    assert d < w0, "This educational example assumes under-damped (d < w0)"
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1 / (2 * np.cos(phi))
    # simplified for tutorial: u(0)=1, u'(0)=0
    return np.exp(-d * t) * (np.cos(w * t))