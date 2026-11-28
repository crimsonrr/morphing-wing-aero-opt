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
    k = np.linspace(0, n_points-1, n_points)

    theta_k = (k*np.pi)/(n_points - 1) # values from 0 to pi
    y_stations = -b/2*np.cos(theta_k) # normalize so that the span stations are in the range [-b/2, b/2], not [-1, 1]

    return theta_k, y_stations

# this function takes the desired 3D Cl, and calculates the 2D section Cl which will then be interpolated for the deflection value 
def calculate_circulation(Cl_target, y_stations, p, vinf, b, S, c): 
    base = np.maximum(0, 1 - (2 * y_stations / b)**2) # naje sure it says in the positive range 
    f_y_stations = base**(1 / p) # shape of the ellipse circulation without the scale factor gamma_0

    # if p = 2, use closed form: 
    if p == 2: 
        gamma_0 = (2*vinf*S*Cl_target) / (np.pi*b)
    else: 
        I = scipy.integrate.simpson(f_y_stations, y_stations) # integral of f(y)dy
        gamma_0 = (0.5*vinf*S*Cl_target) / I # gamma(y) = gamma_0 * f(y) = gamma_0 * I 

# calculate the final 2D circulation distribution and lift coefficient
    gamma_y = gamma_0 * f_y_stations 
    Cl_2d = (2 * gamma_y / (vinf * c))
    
    return gamma_y, Cl_2d
    

def verify_branches(Cl_target, I):
    if np.isclose((0.5*vinf*S*Cl_target) / I, (0.5*vinf*S*Cl_target) / (np.pi * b / 4), rtol=1e-4): 
            print("p = 2 Branches agree")
    else: 
        print("p =2 Branches DO NOT agree!")

if __name__ == "__main__": 
    Cl_target = 0.5 
    p = 2 

    theta_k, y_stations = generated_morphed_naca4412(n_points=101)
    gamma_y, Cl_2d = calculate_circulation(Cl_target, y_stations, p, vinf, b, S, c)
    print(np.trapezoid(Cl_2d * c, x=y_stations) / S) # integrates the 101 local 2d lift coefficients to get back to the original 3D target lift coefficient 

    print(y_stations[0], y_stations[50], y_stations[-1]) # verifies that the spanwise positions are correct