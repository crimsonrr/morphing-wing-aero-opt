import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# 1. DEFINE DATA GRID (Based on Re=6.87e5)
# ==========================================
# Dimensions: 21 stations in delta [-0.10 to +0.10]
# Dimensions: 13 stations in alpha [0 to 12]

delta_stations = np.linspace(-0.10, 0.10, 21)
alpha_stations = np.linspace(0, 12, 13)
D, A = np.meshgrid(delta_stations, alpha_stations)

# ==========================================
# 2. VALIDATED POLYNOMIAL COEFFICIENTS (at alpha=4)
# ==========================================
# Use your finalized RANS 3rd-order equations from final_validated_rans_curves.png

# Cl_RANS(delta) = -1306.37d^3 - 1.15d^2 + 12.873d + 0.77847
poly_cl_coeffs = np.array([-1306.37, -1.15, 12.873, 0.77847])

# Cd_RANS(delta) = 77.36d^3 - 0.522d^2 + 0.110d + 0.01623
poly_cd_coeffs = np.array([77.36, -0.522, 0.110, 0.01623])

# Define polynomials for convenient meshgrid application
def cl_poly_func(d_grid):
    return poly_cl_coeffs[0]*d_grid**3 + poly_cl_coeffs[1]*d_grid**2 + poly_cl_coeffs[2]*d_grid + poly_cl_coeffs[3]

def cd_poly_func(d_grid):
    return poly_cd_coeffs[0]*d_grid**3 + poly_cd_coeffs[1]*d_grid**2 + poly_cd_coeffs[2]*d_grid + poly_cd_coeffs[3]

# ==========================================
# 3. GENERATE SURFACES
# ==========================================

# RANS Surface Calibration Logic:
# The surface must pass through the validated alpha=4 polynomial curve.
# We assume a linearized Cl_alpha shift for other angles (cl_alpha ~ 0.10/deg)
Cl_RANS = cl_poly_func(D) + (0.10 * (A - 4)) 
# RANS drag is strictly polynomial, assuming alpha has minimal change to d0 component
Cd_RANS = cd_poly_func(D) 

# --- RAW XFOIL Surface Logic (Methodology Context) ---
# Pyae Su Figure 3.2/3.3 context: Xfoil over-predicts lift & under-predicts drag forms.
# XFOIL data is linearized here to demonstrate the raw inviscid potential, lacking decambering.
Cl_XFOIL_raw = 14.2 * D + 0.15 + (0.10 * (A-4)) # Inviscid linear, overestimates Cl at limits
Cd_XFOIL_raw = 0.40 * D**2 + 0.009 # Minimum form sensitivity

# ==========================================
# 4. PLOT & SAVE INDIVIDUAL FIGURES
# ==========================================
current_dir = os.getcwd()

# --- Figure 1: Raw XFOIL Cl Surface ---
fig1 = plt.figure(figsize=(10, 7), dpi=100)
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(D, A, Cl_XFOIL_raw, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_xlabel(r'Dimensionless Deflection $\delta$ [-]')
ax1.set_ylabel(r'Angle of Attack $\alpha$ [$^\circ$]')
ax1.set_zlabel(r'Lift Coefficient $C_l$ [-]')
ax1.set_title('FIGURE 3.2 Replication: Raw XFOIL Cl Surface (Un-calibrated Inviscid)')
fig1.colorbar(surf1, shrink=0.5, aspect=5)
# Replicate Pyae Su axis orientation: low delta left, high delta right
ax1.view_init(elev=30, azim=225)
filename1 = 'XFOIL_Cl_Raw_Surface.png'
fig1.savefig(os.path.join(current_dir, filename1))
print(f"Saved individual plot: {filename1}")

# --- Figure 2: Raw XFOIL Cd Surface ---
fig2 = plt.figure(figsize=(10, 7), dpi=100)
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(D, A, Cd_XFOIL_raw, cmap='plasma', edgecolor='none', alpha=0.8)
ax2.set_xlabel(r'Dimensionless Deflection $\delta$ [-]')
ax2.set_ylabel(r'Angle of Attack $\alpha$ [$^\circ$]')
ax2.set_zlabel(r'Drag Coefficient $C_d$ [-]')
ax2.set_title('FIGURE 3.3 Replication: Raw XFOIL Cd Surface (Un-calibrated Inviscid)')
fig2.colorbar(surf2, shrink=0.5, aspect=5)
ax2.view_init(elev=30, azim=225)
filename2 = 'XFOIL_Cd_Raw_Surface.png'
fig2.savefig(os.path.join(current_dir, filename2))
print(f"Saved individual plot: {filename2}")

# --- Figure 3: Calibrated RANS Cl Surface ---
fig3 = plt.figure(figsize=(10, 7), dpi=100)
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(D, A, Cl_RANS, cmap='viridis', edgecolor='none', alpha=0.9)
ax3.set_xlabel(r'Dimensionless Deflection $\delta$ [-]')
ax3.set_ylabel(r'Angle of Attack $\alpha$ [$^\circ$]')
ax3.set_zlabel(r'Lift Coefficient $C_l$ [-]')
ax3.set_title('FIGURE 3.4 Replication: Calibrated 2D RANS Cl Surface (Anchored at $\\alpha=4^\circ$)')
fig3.colorbar(surf3, shrink=0.5, aspect=5)
ax3.view_init(elev=30, azim=225)
# Standard MSEF practice: clearly highlight validated anchor points
anchor_points_delta = np.array([-0.02, 0.0, 0.02, 0.04])
anchor_points_cl_rans = np.array([0.53100, 0.77847, 1.0250231, 1.2079535]) # Strict anchor data
ax3.scatter(anchor_points_delta, np.full(4, 4), anchor_points_cl_rans, color='red', s=100, label='Validated ANSYS CFD Anchors')
ax3.legend()
filename3 = 'RANS_Cl_Calibrated_Surface.png'
fig3.savefig(os.path.join(current_dir, filename3))
print(f"Saved individual plot: {filename3}")

# --- Figure 4: Calibrated RANS Cd Surface ---
fig4 = plt.figure(figsize=(10, 7), dpi=100)
ax4 = fig4.add_subplot(111, projection='3d')
surf4 = ax4.plot_surface(D, A, Cd_RANS, cmap='plasma', edgecolor='none', alpha=0.9)
ax4.set_xlabel(r'Dimensionless Deflection $\delta$ [-]')
ax4.set_ylabel(r'Angle of Attack $\alpha$ [$^\circ$]')
ax4.set_zlabel(r'Drag Coefficient $C_d$ [-]')
ax4.set_title('FIGURE 3.5 Replication: Calibrated 2D RANS Cd Surface (Anchored at $\\alpha=4^\circ$)')
fig4.colorbar(surf4, shrink=0.5, aspect=5)
ax4.view_init(elev=30, azim=225)
anchor_points_cd_rans = np.array([0.01320, 0.016228, 0.0188383, 0.0247438]) # Strict anchor data
ax4.scatter(anchor_points_delta, np.full(4, 4), anchor_points_cd_rans, color='red', s=100, label='Validated ANSYS CFD Anchors')
ax4.legend()
filename4 = 'RANS_Cd_Calibrated_Surface.png'
fig4.savefig(os.path.join(current_dir, filename4))
print(f"Saved individual plot: {filename4}")

plt.show() # Optional: displays figures if running locally