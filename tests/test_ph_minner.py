import phmin
import numpy as np

def gen_test_data():
    time = np.linspace(0, 10, 101)
    data = 10*np.sin(2*np.pi*(0.4*time+0.4)) + np.random.normal(loc=0, scale=3, size=len(time))
    return time, data

def test_ph_minner():
    x, y = gen_test_data()
    print("Running test minner. Should determine best period to be 2.5 d.")
    test_minner = phmin.ph_minner(time=x, data=y)
    test_minner.run(verbose=True)
    test_minner.plot_chisq()



