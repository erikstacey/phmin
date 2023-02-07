from setuptools import setup

setup(name="phmin",
      version="0.0.1",
      description="Basic package for phase dispersion minimization in time series data.",
      url="https://www.github.com/erikstacey/PHMIN",
      author="Erik William Stacey",
      author_email = "erik@erikstacey.com",
      license="MIT",
      packages=["phmin"],
      install_requires=[
            "numpy",
            "matplotlib",
            "scipy"
      ],
      tests_require=["pytest"],
      zip_safe = False)