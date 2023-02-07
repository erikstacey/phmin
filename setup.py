from setuptools import setup

setup(name="phmin",
      version="0.1.4",
      description="A basic python package for identifying frequencies present in"
                  " time series data based on the principles of phase dispersion minimization. ",
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