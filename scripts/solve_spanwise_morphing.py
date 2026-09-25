import numpy as np 
import pandas as pd 
import aerosandbox as asb

float vinf = 10 # freestream velocity
c = 1.0 # chord length 
b = # span length 
S = c*s # total wing area
Cl = 0.50 # desired lift coefficient; varies depending on the flight regime (0.50 is a benchmark)

def generated_morphed_naca4412(n_points=101):
# establishes span stations with cosine distribution for N = 101 points
    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2 
    np.clip(x,0,1)

def calculate_circulation(): 

