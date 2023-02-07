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
        If run() has been called at least once, returns the best period, best amplitude, best phase, and minimum chi
        squared of the last run.
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
        self.N=0
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
            self.N+=1

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
        best_i = np.argmin(self.red_chisqs)
        if type(best_i) == np.array:
            print("[phmin][WARNING] Multiple equivalent values were found for minimum reduced chi squareds. "
                  "This is most likely to happen if the fits all fail and revert to initial parameters. If this is the"
                  "case, consider manually defining a period grid around the region of interest when initializing"
                  "the ph_minner object.")
            print("[phmin][INFO] Assuming first value to be the true minimum and proceeding...")
            best_i = best_i[0]
        return self.periods[best_i], self.best_amps[best_i], self.best_phcorrs[best_i], self.red_chisqs[best_i]

    def print_results(self):
        if not self.has_run:
            raise NoResultsError("Cannot print report as no results have been computed yet. "
                                 "Either something went wrong while computing or you haven't called"
                                 "ph_minner.run() yet.")
        best_period, best_amp, best_phi, best_redchisq = self.best_fit_pars()
        """Prints the best results to the console."""
        print("[phmin][OUT] FIT REPORT")
        print(f"\t\t Time: {self.compute_time}, It: {self.N}")
        print(f"\t\t Best fit period: {best_period}")
        print(f"\t\t Best fit amplitude: {best_amp}")
        print(f"\t\t Best fit phase: {best_phi}")
        print(f"\t\t Best fit reduced chi squared: {best_redchisq}")

    def plot_results(self):
        """shows a set of three stacked subplots summarizing the results
        Warning, this will show any figures you have in preparation and will clear them after. Manually plot
        if you need to keep other figures."""
        print("[phmin][OUT] Displaying results plot...")
        if not self.has_run:
            raise NoResultsError("Cannot print report as no results have been computed yet. "
                                 "Either something went wrong while computing or you haven't called"
                                 "ph_minner.run() yet.")
        best_period, best_amp, best_phi, best_redchisq = self.best_fit_pars()
        best_phases = phase(self.time, self.t0, best_period)
        fig, axs = pl.subplots(3, 1)
        fig.set_size_inches(8,12)

        if np.any(self.err!=1):
            axs[0].errorbar(self.time, self.data, yerr= self.err, lw=1, fmt='.k', capsize=3, color="black",
                            linestyle="none")
        else:
            axs[0].plot(self.time, self.data, color="black", linestyle="none", marker=".")
        axs[0].set_xlabel("Phase [0-1]")
        axs[0].set_ylabel("Data")

        if np.any(self.err!=1):
            axs[1].errorbar(best_phases, self.data, yerr= self.err, lw=1, fmt='.k', capsize=1.5, color="black",
                            label="data")
        else:
            axs[1].plot(best_phases, self.data, color="black", linestyle="none", marker=".", label="data")
        model_phases = np.linspace(0,1,1000)
        axs[1].plot(model_phases, sinf1(model_phases, best_amp, best_phi), color="red", label="Best Fit Model")
        axs[1].set_xlabel("Phase [0-1]")
        axs[1].set_ylabel("Data")
        axs[1].legend()
        axs[2].plot(self.periods, self.red_chisqs, color="black")
        axs[2].axvline(best_period, color="red")
        axs[2].set_xlabel("Period")
        axs[2].set_ylabel("Red. Chi Squared")
        pl.show()
        pl.clf()










