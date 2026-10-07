import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the XFOIL databases
cl_morph = pd.read_csv("data/Cl_xfoil.csv", index_col=0)
cd_morph = pd.read_csv("data/Cd_xfoil.csv", index_col=0)
cl_flap = pd.read_csv("data/Cl_flap.csv", index_col=0)
cd_flap = pd.read_csv("data/Cd_flap.csv", index_col=0)

# Extract angle of attack array from headers
alphas = np.array([float(col.split("_")[1].replace("deg", "")) for col in cl_morph.columns])

# Deflection cases to compare: Baseline, Cruise Camber, and High-Lift
cases = [0.00, 0.04, 0.08]
colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

for d, col in zip(cases, colors):
    # Locate closest matching deflection row in each dataset
    idx_m = cl_morph.index[np.argmin(np.abs(cl_morph.index - d))]
    idx_f = cl_flap.index[np.argmin(np.abs(cl_flap.index - d))]

    if np.isclose(d, 0.0):
        # Baseline NACA 4412 is geometrically identical for both
        ax1.plot(
            alphas,
            cl_morph.loc[idx_m],
            color=col,
            linestyle="-",
            linewidth=2,
            label=r"Baseline ($\delta = 0.00$ m)",
        )
        ax2.plot(
            cd_morph.loc[idx_m],
            cl_morph.loc[idx_m],
            color=col,
            linestyle="-",
            linewidth=2,
            label=r"Baseline ($\delta = 0.00$ m)",
        )
    else:
        # Continuous Morphed Configuration (Solid lines)
        ax1.plot(
            alphas,
            cl_morph.loc[idx_m],
            color=col,
            linestyle="-",
            linewidth=2,
            label=rf"Morphing ($\delta = +{d:.2f}$ m)",
        )
        ax2.plot(
            cd_morph.loc[idx_m],
            cl_morph.loc[idx_m],
            color=col,
            linestyle="-",
            linewidth=2,
            label=rf"Morphing ($\delta = +{d:.2f}$ m)",
        )

        # Hinged Plain Flap Configuration (Dashed lines + square markers)
        ax1.plot(
            alphas,
            cl_flap.loc[idx_f],
            color=col,
            linestyle="--",
            marker="s",
            markersize=3.5,
            alpha=0.85,
            label=rf"Hinged Flap ($\delta = +{d:.2f}$ m)",
        )
        ax2.plot(
            cd_flap.loc[idx_f],
            cl_flap.loc[idx_f],
            color=col,
            linestyle="--",
            marker="s",
            markersize=3.5,
            alpha=0.85,
            label=rf"Hinged Flap ($\delta = +{d:.2f}$ m)",
        )

# Left Subplot: Cl vs Alpha
ax1.set_title(r"Cl vs Alpha: Continuous Morphing vs. Hinged Flap ($\pm 20^\circ$)", fontsize=11, fontweight="bold")
ax1.set_xlabel(r"Angle of Attack $\alpha$ (deg)", fontsize=10)
ax1.set_ylabel(r"Section Lift Coefficient $C_l$", fontsize=10)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.set_xlim([-20, 20])
ax1.set_ylim([-2.2, 2.5])
ax1.axhline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.5)
ax1.axvline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.5)
ax1.legend(fontsize=8, loc="upper left")

# Right Subplot: Cl vs Cd (Drag Polar & Drag Bucket)
ax2.set_title("Cl vs Cd: Continuous Morphing vs. Hinged Flap", fontsize=11, fontweight="bold")
ax2.set_xlabel(r"Section Drag Coefficient $C_d$", fontsize=10)
ax2.set_ylabel(r"Section Lift Coefficient $C_l$", fontsize=10)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.set_xlim([0.003, 0.15])
ax2.set_ylim([-2.2, 2.5])
ax2.axhline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.5)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
os.makedirs("docs", exist_ok=True)
output_path = "docs/morph_vs_hinged_flap_polars.png"
plt.savefig(output_path)
print(f"Saved comparative polar plot to {output_path}")