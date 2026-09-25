import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load datasets
cl_nf = pd.read_csv("data/Cl_lookup_table.csv", index_col=0)
cd_nf = pd.read_csv("data/Cd_lookup_table.csv", index_col=0)
cl_xf = pd.read_csv("data/Cl_xfoil.csv", index_col=0)
cd_xf = pd.read_csv("data/Cd_xfoil.csv", index_col=0)

# Extract angles from column headers
alphas = np.array([float(col.split("_")[1].replace("deg", "")) for col in cl_nf.columns])

# Deflection cases to compare
cases = [-0.10, 0.00, 0.10]
colors = ["#1f77b4", "#2ca02c", "#d62728"]
labels = [
    r"Reflex ($\delta = -0.10$ m)",
    r"Baseline ($\delta = 0.00$ m)",
    r"Cambered ($\delta = +0.10$ m)",
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

for d, col, lbl in zip(cases, colors, labels):
    # Find closest row index
    idx = cl_nf.index[np.argmin(np.abs(cl_nf.index - d))]

    # NeuralFoil curves (Solid lines)
    ax1.plot(
        alphas,
        cl_nf.loc[idx],
        color=col,
        linestyle="-",
        linewidth=2,
        label=f"{lbl} - NeuralFoil",
    )
    ax2.plot(
        cd_nf.loc[idx],
        cl_nf.loc[idx],
        color=col,
        linestyle="-",
        linewidth=2,
        label=f"{lbl} - NeuralFoil",
    )

    # XFOIL data (Dashed lines + open markers)
    ax1.plot(
        alphas,
        cl_xf.loc[idx],
        color=col,
        linestyle="--",
        marker="o",
        markersize=5,
        alpha=0.8,
        label=f"{lbl} - XFOIL",
    )
    ax2.plot(
        cd_xf.loc[idx],
        cl_xf.loc[idx],
        color=col,
        linestyle="--",
        marker="o",
        markersize=5,
        alpha=0.8,
        label=f"{lbl} - XFOIL",
    )

# Formatting Left Plot: Cl vs Alpha
ax1.set_title("Cl v Alpha", fontsize=12, fontweight="bold")
ax1.set_xlabel(r"Angle of Attack $\alpha$ (deg)", fontsize=11)
ax1.set_ylabel(r"Section Lift Coefficient $C_l$", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.set_xlim([-20, 20])
ax1.legend(fontsize=8, loc="upper left")

# Formatting Right Plot: Drag Polar (Cl vs Cd)
ax2.set_title("Cl v Cd", fontsize=12, fontweight="bold")
ax2.set_xlabel(r"Section Drag Coefficient $C_d$", fontsize=11)
ax2.set_ylabel(r"Section Lift Coefficient $C_l$", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.set_xlim([0.003, 0.12])
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/benchmark_neuralfoil_vs_xfoil.png")
print("Saved benchmark plot to docs/benchmark_neuralfoil_vs_xfoil.png")
# plt.show()