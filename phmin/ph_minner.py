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
    tvar : float
        Total variance of the data array. Used to compute the theta statistic.
    thetas : np.array
        Array of theta statistics. Theta close to zero indicates statistical significance, and the minimum
        theta indicates the most statistically likely period.

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

    def __init__(self, time, data, err=None, periods=None, t0=0, nb=5, nc=2):
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
        self.nb = nb
        self.nc = nc
        self._build_bins(nb=nb, nc=nc)
        self.tvar = np.var(self.data)
        self.thetas = np.zeros(len(self.periods))




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

    def _compute_bin_var(self, phases, ll, ul):
        # if ll > ul, then the bin is wrapping around the unit interval and must be treated differently
        if ll > ul:
            bin_mask = (phases>ll) + (phases < ul)
        else:
            bin_mask = (phases>ll) * (phases < ul)
        bin_data = self.data[bin_mask]
        if len(bin_data) >=2:
            return np.var(self.data[bin_mask]), len(bin_data)
        else:
            return 0, 0

    def run(self, verbose = False):
        self.N = 0
        print("[phmin][INFO] Running phmin...")
        starttime=time.time()
        for i in range(len(self.periods)):
            self.N+=1
            phased_time = phase(self.time, self.t0, self.periods[i])
            # compute vars for each bin
            bin_vars = np.zeros(self.nb*self.nc)
            bin_npts = np.zeros(self.nb * self.nc)
            for k in range(self.nb*self.nc):
                bin_vars[k], bin_npts[k] = self._compute_bin_var(phased_time, self._bins_ll[k], self._bins_ul[k])


            # evaluate according to eq 2 in Stellingwerf 1978
            s_num = 0
            s_den = -self.nb*self.nc  # -M component
            for k in range(self.nb*self.nc):
                s_num += (bin_npts[k]-1)*bin_vars[k]
                s_den += bin_npts[k]
            self.thetas[i] = (s_num / s_den) / self.tvar

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
        """
        best_i = np.argmin(self.thetas)
        best_P = self.periods[best_i]
        min_theta = self.thetas[best_i]
        return best_P, min_theta

    def print_results(self):
        if not self.has_run:
            raise NoResultsError("Cannot print report as no results have been computed yet. "
                                 "Either something went wrong while computing or you haven't called"
                                 "ph_minner.run() yet.")
        else:
            best_P, min_theta = self.best_fit_pars()
            print("[phmin][OUT] Fit Report:")
            print(f"\t\t Run complete in {self.compute_time} s evaluating {self.N} iterations and {self.nb*self.nc} "
                  f"bins per iteration")
            print(f"\t\t Best P: {best_P}")
            print(f"\t\t Min Theta: {min_theta}")

    def plot_results(self):
        """shows a set of three stacked subplots summarizing the results
        Warning, this will show any figures you have in preparation and will clear them after. Manually plot
        if you need to keep other figures."""
        print("[phmin][OUT] Displaying results plot...")
        if not self.has_run:
            raise NoResultsError("Cannot print report as no results have been computed yet. "
                                 "Either something went wrong while computing or you haven't called"
                                 "ph_minner.run() yet.")
        else:
            best_P, min_theta = self.best_fit_pars()
            best_phases = phase(self.time, self.t0, best_P)
            fig, axs = pl.subplots(3, 1)
            fig.set_size_inches(8, 12)

            if np.any(self.err != 1):
                axs[0].errorbar(self.time, self.data, yerr=self.err, lw=1, capsize=3, color="black",
                                linestyle="none", marker=".")
            else:
                axs[0].plot(self.time, self.data, color="black", linestyle="none", marker=".")
            axs[0].set_xlabel("Time")
            axs[0].set_ylabel("Data")

            if np.any(self.err != 1):
                axs[1].errorbar(best_phases, self.data, yerr=self.err, lw=1, capsize=1.5, color="black",
                                label="data", linestyle="none", marker=".")
            else:
                axs[1].plot(best_phases, self.data, color="black", linestyle="none", marker=".")
            axs[1].set_xlabel(f"Phase [0-1], P={best_P}")
            axs[1].set_ylabel("Data")
            axs[2].plot(self.periods, self.thetas, color="black")
            axs[2].axvline(best_P, color="red")
            axs[2].set_xlabel("Period")
            axs[2].set_ylabel("Theta")
            pl.show()
            pl.clf()

if __name__=="__main__":
    testobj = ph_minner([0, 1, 2], [0, 1, 2])
    testobj._build_bins(5,2)
    testobj.diagnostic_plot_bins()