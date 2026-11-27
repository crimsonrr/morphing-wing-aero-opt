import numpy as np 
import pandas as pd 
import aerosandbox as asb
import scipy
from scipy import integrate

vinf = 50.17 # freestream velocity
c = 0.2 # chord length 
b = 1.312 # span length 
S = c*b # total wing area
Cl = 0.50 # desired lift coefficient; varies depending on the flight regime (0.50 is a benchmark)

def generated_morphed_naca4412(n_points=101):
# establishes span stations with cosine distribution for N = 101 points
    k = np.linspace(0, n_points-1, 101)

    theta_k = (k*np.pi)/(n_points - 1) # values from 0 to pi 
    yk = -b/2*np.cos(theta_k) # normalize so that the span stations are in the range [-b/2, b/2], not [-1, 1]

    return theta_k, yk 

# this function takes the desired 3D Cl, and calculates the 2D section Cl which will then be interpolated for the deflection value 
def calculate_circulation(Cl_target, y_stations, p, vinf, b, S, c, yk): 
    # if p = 2, use closed form: 
    if p == 2: 
        gamma_0 = (2*vinf*S*Cl_target) / (np.pi*b)
    else: 
        f_yk = (1-(2*yk/b)**2)**(1/p) # shape of the ellipse circulation without the scale factor gamma_0
        I = scipy.integrate.simpson(f_yk, yk) # integral of f(y)dy
        gamma_0 = (0.5*vinf*S*Cl) / I # gamma(y) = gamma_0 * f(y) = gamma_0 * I 
