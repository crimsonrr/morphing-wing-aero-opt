import numpy as np 
import pandas as pd 
import aerosandbox as asb
import os

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

# allocate storage for Cl and Cd matrices beforehand
Cl_xfoil = np.full((len(delta_range), len(alpha_range)), np.nan)
Cd_xfoil = np.full((len(delta_range), len(alpha_range)), np.nan)

print(f"starting XFOIL batch run across {len(delta_range) * len(alpha_range)} conditons...")

# enumerate through all 21 deflections
for i, delta in enumerate(delta_range): 
    delta = delta_range[i]
    coords = generate_morphed_naca4412(delta=delta, x_h=0.70, n_points=200) # use 200 points for XFOIL simulations
    airfoil = asb.Airfoil(name=f"morphed_d{delta:.2f}", coordinates = coords)

    xf = asb.XFoil(
        airfoil=airfoil,
        Re=Re,
        mach = mach, # incompressible flow conditions
        max_iter = 100, 
        xfoil_command = "bin/xfoil.exe",
        verbose=False
    )

    # evaluate across all AoA's
    results = xf.alpha(alpha_range) 

    # verify that XFOIL returned data 
    if "alpha" in results and len(results["alpha"]) > 0: 
    # all these "converged" values are values that weren't possibly omitted from flow seperation/stall effects
        converged_cls = results["CL"]
        converged_cds = results["CD"]
        converged_alphas = results["alpha"]

    # loop through each converged alpha value 
        for k in range(len(results["alpha"])): 
            a = converged_alphas[k]
    # returns the columnn number where the angle belongs
            matching_cols = np.where(np.isclose(alpha_range, a, atol=1e-2))[0]

            if len(matching_cols) > 0: 
                col_id = matching_cols[0]
            # place the solved value into row i and column col_id
                Cl_xfoil[i, col_id] = converged_cls[k]
                Cd_xfoil[i, col_id] = converged_cds[k]

    converged_count = np.sum(~np.isnan(Cl_xfoil[i, :]))
    print(f"Ccmpleted delta = {delta:+.2f} m | converged: {converged_count}/{len(alpha_range)}")

# export results to CSV
os.makedirs("data", exist_ok=True)
col_names = [f"alpha_{int(a)}deg" for a in alpha_range]

df_cl = pd.DataFrame(Cl_xfoil, index=np.round(delta_range, 2), columns=col_names)
df_cl.index.name = "delta"
df_cl.to_csv("data/Cl_xfoil.csv")

df_cd = pd.DataFrame(Cd_xfoil, index=np.round(delta_range, 2), columns=col_names)
df_cd.index.name = "delta"
df_cd.to_csv("data/Cd_xfoil.csv")

print("Saved 'data/Cl_xfoil.csv' and 'data/Cd_xfoil.csv' successfully.")