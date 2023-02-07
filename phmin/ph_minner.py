import numpy as np
from scipy.optimize import minimize
from phmin.funcs import phase, chisq, red_chisq, sinf1
import matplotlib.pyplot as pl
import time
from phmin.custom_exceptions import NoResultsError

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
    nb : int
        Number of bins
    nc : int
        Number of "covers". Each data point occupies nc bins.


    _bins_ll : np.array
        array of cover bin lower limits
    _bins_ul : np.array
        array of cover bin upper limits

    N : int
        Iteration counter
    compute_time : float
        Stores the total time taken to complete a run
    has_run : bool
        indicates if a run has been completed

    Methods
    ----------
    run(verbose=False)
        Iterates over all candidate periods in self.periods, performing sinusoidal model optimizations on the phased
        data at each candidate period. Populates the results arrays.
    best_fit_pars()
        Placeholder
    print_results()
        Prints a basic report on the last run
    plot_results()
        Plots a basic three-stacked subplot figure illustrating the results from the last run.
    """
    has_run = False
    N=0
    compute_time = 0

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

    def _build_bins(self, nb, nc):
        """Populates the _ncbins_x arrays with the bin edges"""
        self.nb = nb
        self.nc = nc
        self._bins_ll = np.zeros(nb*nc)
        self._bins_ul = np.zeros(nb*nc)
        bw = 1/nb
        for i in range(self.nb*self.nc):
            curr_leftedge = i/(nb*nc)
            curr_rightedge = curr_leftedge + bw
            # wrap around unit interval
            if curr_rightedge > 1:
                curr_rightedge -= 1

            self._bins_ll[i] = curr_leftedge
            self._bins_ul[i] = curr_rightedge

    def diagnostic_plot_bins(self):
        fig, ax = pl.subplots()
        for i in range(self.nb*self.nc):
            rgb = np.random.rand(3)
            ax.axvline(self._bins_ll[i], color=rgb, linestyle="--")
            ax.axvline(self._bins_ul[i], color=rgb, linestyle="--")

        ax.set_xlim(0, 1)
        pl.show()




    def run(self, verbose = False):
        self.N=0
        print("[phmin][INFO] Running phmin...")
        starttime=time.time()
        for i in range(len(self.periods)):
            pass

        self.compute_time = time.time() - starttime
        print(f"[phmin][INFO] phminner run complete in {self.compute_time:.5f} s ({self.N} it)")
        self.has_run = True
        return 0

    def best_fit_pars(self):
        """Finds the minimum red chisq and its corresponding period, amplitude, phase
        Returns
        -----------
        minper : float
            The best-fit period
        bestamp : float
            The best-fit amplitude at the best-fit period
        bestphi : float
            The best-fit phase at the best-fit period
        bestredchi : float
            The minimum measured red chisq
        """
        pass

    def print_results(self):
        if not self.has_run:
            raise NoResultsError("Cannot print report as no results have been computed yet. "
                                 "Either something went wrong while computing or you haven't called"
                                 "ph_minner.run() yet.")
        pass

    def plot_results(self):
        """shows a set of three stacked subplots summarizing the results
        Warning, this will show any figures you have in preparation and will clear them after. Manually plot
        if you need to keep other figures."""
        print("[phmin][OUT] Displaying results plot...")
        if not self.has_run:
            raise NoResultsError("Cannot print report as no results have been computed yet. "
                                 "Either something went wrong while computing or you haven't called"
                                 "ph_minner.run() yet.")

if __name__=="__main__":
    testobj = ph_minner([0, 1, 2], [0, 1, 2])
    testobj._build_bins(5,2)
    testobj.diagnostic_plot_bins()