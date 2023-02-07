import numpy as np

def phase(t, t0, P):
    """Given a time t, reference time t0, and period P, returns a value for phase in [0, 1]. Can also operate on
    numpy arrays"""
    return ((t-t0)/P)%1

def sinf1(x, a, phi):
    return a*np.sin(2*np.pi*(x+phi))

def chisq(model, data, err):
    return np.sum(((model-data)/err)**2)

def red_chisq(model, data, err, ndof):
    return np.sum(((model-data)/err)**2) / ndof
