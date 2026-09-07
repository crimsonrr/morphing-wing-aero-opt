import numpy as np 
import pandas as pd 
import aerosandbox as asb
import neuralfoil as nf 

# 21 evenly spaced deflection values from -0.10 to 0.10 
delta_range = np.linspace(-0.10, 0.10, 21)

#integer angle of attack values
alpha_range = np.linspace(0, 12, 13)

# flight parameters (c = 1.0 m)
Re = 6.87e5
mach = 0.03

# x is defined to be the total length of the chord
def naca4_thickness(x, t=0.12): 
    return 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

# where m is the maximum camber value (0.04%) and p is the location of the max camber (40% chord) 
# camber generation for the baseline airfoil
def naca4412_camber_baseline(x, m=0.04, p=0.40):
    yc = np.where(x<p, 
             m/p**2 * (2*p*x-x**2),
             (m/(1-p)**2)*((1-2*p)+2*p*x-x**2)
             )
    
    dyc_dx = np.where(x<p, 
             2*m/p**2 * (p-x),
             (2*m/(1-p)**2) * (p-x)
             )
    
    return yc, dyc_dx

def generate_morphed_naca4412(delta, x_h=0.70, n_points=2000):
# use cosine spacing to create normalized positions from the leading to trailing edge
    beta = np.linspace(0, np.pi, n_points // 2)
    x = (1.0 - np.cos(beta)) / 2.0

# extract the baseline camber line and its derivative across the ENTIRE grid
    yc_base, dyc_base = naca4412_camber_baseline(x)

# extract the camber elevation and slope specifically at the hinge (x = 0.70 m)
    yc_h, dyc_h = naca4412_camber_baseline(np.array([x_h]))

# define the vertical height and the slope of the baseline camber at the hinge
    y0_h, dy0_h = yc_h[0], dyc_h[0]

    l = 1.0 - x_h
    A = (-delta - dy0_h * l - y0_h) / (l**2)
    B = dy0_h
    C = y0_h

# generation of camber and slope curvs
    yc_morph = A * (x - x_h)**2 + B * (x - x_h) + C
    dyc_morph = 2 * A * (x - x_h) + B

    yc = np.where(x<x_h, yc_base, yc_morph)
    dyc = np.where(x<x_h, dyc_base, dyc_morph)

# apply thickness normal to camber line
    theta = np.arctan(dyc)
    yt = naca4_thickness(x, t=0.12)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

# order the points
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack((x_coords, y_coords))

Cl_matrix = np.zeros((len(delta_range), len(alpha_range)))
Cd_matrix = np.zeros((len(delta_range), len(alpha_range)))

# run NeuralFoil
for i, delta in enumerate(delta_range):
    coords = generate_morphed_naca4412(delta=delta, x_h=0.70)
    airfoil = asb.Airfoil(name=f"morphed_delta_{delta}", coordinates=coords)
    
    for j, alpha in enumerate(alpha_range):
        aero = nf.get_aero_from_airfoil(
            airfoil=airfoil,
            alpha=float(alpha),
            Re=Re
        )

for i in range(0,20): 
    coords = generate_morphed_naca4412(delta=delta, x_h=0.70, n_points=200) # use 200 points for XFOIL simulations
    airfoil = asb.Airfol(name=f"morphed_d{delta:.2f}", coordinates = coords)

    xf = asb.XFoil(
        airfoil=airfoil,
        Re=Re,
        mach = mach, # incompressible flow conditions
        max_iter = 100, 
        verbose=False
    )

# evaluate across all AoA's
    results = xf.alpha(alpha_range)