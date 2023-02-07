import numpy as np
from scipy.optimize import minimize
from phmin.funcs import phase, chisq, red_chisq, sinf1
import matplotlib.pyplot as pl
import time

class ph_minner():
    """Class that stores and operates on a time series to measure periods via phase dispersion minimization.

    Attributes
    ----------
    time: np.array
        Time entries of time series
    data: np.array
        Measurements of time series
    err: np.array
        Weights of time series
    periods : np.array
        Stores the period grid used for the phase dispersion minimization. Acts as an 'x-axis' for the measured red
        chisqs and the stored parameters
    best_amps : np.array
        stores the best amplitudes determined during the scipy minimize step for each period
    best_phcorrs : np.array
        same as best_amps for the phase corrections
    red_chisqs : np.array
        Stores the best red chisq achieved for each period.
    chisqs : np.array
        Stores the best chisq achieved for each period.
    ndof : int
        Number of degrees of freedom in the minimization process.

    Methods
    ----------
    _gen_P_grid(
    """


    def __init__(self, time, data, err=None, periods=None, t0=0):
        """Initializer for ph_minner. Imports data.

        Parameters
        ----------
        time: np.array
            Time entries of time series
        data: np.array
            Measurement entries of time series
        err : np.array, optional
            Weights of time series data points
        periods: np.array, optional
            Period grid to consider. If this isn't specified, a period grid is automatically generated based on
            the time span and number of data points in the time series.
        t0: float
            Reference time. Defaults to 0.
        Returns
        ----------
        Nothing
        """
        self.time = time
        self.data = data
        if err is not None:
            self.err = err
        else:
            # set equal weights if no uncertainties are specified
            # in this case the chi squared is somewhat meaningless
            self.err = np.ones(len(self.time))

        if periods is not None:
            self.periods=periods
        else:
            self._gen_P_grid()

        self.t0=t0

        # compute the ndof
        # functionally have three parameters (period, amplitude, phase)
        self.ndof = len(self.time) - 3

        # initialize best_pars and red_chisqs arrays with
        self.best_amps = np.zeros(len(self.periods))
        self.best_phcorrs = np.zeros(len(self.periods))
        self.red_chisqs = np.zeros(len(self.periods))
        self.chisqs = np.zeros(len(self.periods))

    def _adjust_params(self, in_a, in_p):
        a = in_a
        p = in_p
        # adjust amplitude
        if a < 0:
            a = abs(self.a)
            p += 0.5

        if p > 1 or p < 0:
            p = p % 1

        return a, p


    def _gen_P_grid(self):
        """Automatically generates a period grid through a naive treatment of the input time series. Shouldn't be
        called by user."""
        deltaT = max(self.time) - min(self.time)
        ndpts = len(self.time)
        # determine max period through the minimum frequency that should be reasonably measurable based on
        # 1.5/deltaT
        max_p = deltaT/1.5
        # (lazily) determine min period by assuming even sampling and considering out to the nyquist frequency
        # sample rate is samples/[time], where [time] is the unit in which the time array is specified
        min_p = deltaT/(0.5 * ndpts)
        # determine the number of periods through the number of increments of 1.5/deltaT that will fit into the
        # period range. Multiply by 2 for redundancy.
        nperiods = int(20*(max_p-min_p)/min_p)

        self.periods = np.linspace(min_p, max_p, nperiods)


    def run(self, verbose = False):
        N=0
        print("[phmin][INFO] Running phmin...")
        starttime=time.time()
        for i in range(len(self.periods)):
            phased_time = phase(self.time, self.t0, self.periods[i])
            # lambda function wraps the chi squared function into a form appropriate for scipy minimize
            min_func = lambda pars: chisq(sinf1(phased_time, pars[0], pars[1]), self.data, self.err)
            x0 = np.array([max(self.data-np.mean(self.data)), 0.5])
            min_res = minimize(fun=min_func, x0=x0)


            self.best_amps[i] = min_res.x[0]
            self.best_phcorrs[i] = min_res.x[1]
            self.red_chisqs[i] = red_chisq(sinf1(phased_time, min_res.x[0], min_res.x[1]),
                                        self.data, self.err, self.ndof)
            self.chisqs[i] = chisq(sinf1(phased_time, min_res.x[0], min_res.x[1]),
                                        self.data, self.err)
            if verbose:
                print(f"[phmin][DEBUG] P={self.periods[i]} complete. Red chisq:{self.red_chisqs[i]}")
                print(f"\t\t a={self.best_amps[i]}, phcorr={self.best_phcorrs[i]}")
            N+=1

        print(f"[phmin][INFO] phminner run complete in {time.time()-starttime:.5f} s ({N} it)")
        return 0

    def plot_rchisq(self, show=True, fmt= True):
        pl.plot(self.periods, self.red_chisqs, color="black")
        if fmt:
            pl.xlabel("Period")
            pl.ylabel("Red. Chi Squared")
        if show:
            pl.show()

    def plot_chisq(self, show=True, fmt= True):
        pl.plot(self.periods, self.chisqs, color="black")
        if fmt:
            pl.xlabel("Period")
            pl.ylabel("Chi Squared")
        if show:
            pl.show()









