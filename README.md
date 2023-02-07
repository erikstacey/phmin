# phmin

A basic python package for determining periods in time series data through phase dispersion minimization.  
Author: Erik William Stacey  
Date: 6 Feb, 2023  

## Installation

phmin is presently only available on github and must be downloaded and installed using pip: 
```
git clone https://github.com/erikstacey/phmin.git ./phmin
cd phmin
pip install .
```
## Usage

The ph_minner class is used to store and operate on time series data. Here's a basic example snippit setting up a ph_minner:  
```
import numpy as np
from phmin import ph_minner

x, y, y_err = np.loadtxt("sample.txt", unpack=True)
example_minner = ph_minner(x=x, y=y, err=y_err)
```
The above code will automatically generate a period grid. To run the minimizer, call the run method:  
```
example_minner.run()
```

## Advanced Usage
The ph_minner class can be initialized with optional parameters periods and t0. Setting periods manually defines the period grid, and setting t0 manually sets the reference time for the phasing.
```
manual_period_grid = np.linspace(4.4, 4.8, 500)
example_minner = ph_minner(x=x, y=y, err=y_err, periods=manual_period_grid, t0=2457000.0)
```

The results can be directly accessed through attributes of the ph_minner class.  
```example_minner.periods``` - Array of periods  
```example_minner.red_chisqs``` - Array of reduced chi squared values corresponding to the best fit achieved at each period  
```example_minner.chisqs``` - Array of (unreduced) chi squared values corresponding to the best fit achieved at each period  
```example_minner.best_amps``` - Array of the optimized amplitude at each period  
```example_minner.best_phcorrs``` - Array of the optimized phase at each period. This defines the signal phase relative to t0.  

## Method
This package is intended to solve the problem of determining the principal frequency present in time series data through 
phase dispersion minimization. Once the ph_minner has been initialized and run, it iterates over all candidate periods and
in each iteration:
1) Converts the timeseries time stamps to phases
2) Optimizes a sinusoidal model using scipy minimize in amplitude and phase
3) Measures a reduced chi squared of the optimized sinusoidal model
4) Stores the reduced chi squared and optimized amplitude/phase

Then, after iterating over all candidate periods, the results can be examined and the best-fit period can be determined by
finding the candidate period corresponding to the minimum reduced chi squared. Alternatively, the relationship between
period and goodness of fit can be examined by plotting candidate period v. red chi squared.





Log:  
6 Feb, 2023  
-Working package published on github