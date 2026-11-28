import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =========================================================================
# 1. LOAD YOUR ACTUAL CSV DATASETS DIRECTLY
# =========================================================================
cl_xfoil_df = pd.read_csv('data/Cl_xfoil.csv')
cl_rans_df = pd.read_csv('data/Cl_2D_RANS_matrix.csv')

deltas = cl_xfoil_df['delta'].values                  # -0.10 to +0.10 (21 rows)
alphas = np.arange(0, 13, 1.0)                        # 0 to 12 deg (13 cols)
xfoil_cols = [f'alpha_{int(a)}deg' for a in alphas]
rans_cols = [f'{a:.1f}' for a in alphas]

# Extract raw Cl matrices and compute exact difference
Z_cl_xfoil_raw = cl_xfoil_df[xfoil_cols].values.astype(float)
Z_cl_rans = cl_rans_df[rans_cols].values.astype(float)
Delta_Cl_real = Z_cl_xfoil_raw - Z_cl_rans

# Optional: Export the exact Delta_Cl matrix to CSV for your paper appendix
delta_df = pd.DataFrame(Delta_Cl_real, index=deltas, columns=[f'{int(a)} deg' for a in alphas])
delta_df.index.name = 'delta'
delta_df.to_csv('Real_Table_5_2_Delta_Cl.csv')

# =========================================================================
# 2. PLOT TABLE 5.2 HEATMAP (GREEN-YELLOW-RED CONDITIONAL FORMATTING)
# =========================================================================
fig, ax = plt.subplots(figsize=(13.5, 9.2), dpi=150)

# Custom colormap matching Pyae Su's Table 5.2
cmap_pyae = mcolors.LinearSegmentedColormap.from_list(
    'pyae_table52', ['#63be7b', '#a9d18e', '#ffeb84', '#f8cbad', '#f8696b']
)
cmap_pyae.set_bad(color='#e0e0e0')  # Light grey for the 8 XFOIL non-convergence (NaN) cells

im = ax.imshow(Delta_Cl_real, cmap=cmap_pyae, aspect='auto', vmin=-0.30, vmax=0.35)

# Highlight your 4 direct ANSYS Fluent anchor points at alpha = 4 deg
deltas_anchor = np.array([-0.02, 0.00, 0.02, 0.04])

for i in range(len(deltas)):
    for j in range(len(alphas)):
        val = Delta_Cl_real[i, j]
        if np.isnan(val):
            ax.text(j, i, 'NaN', ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color='#555555')
        else:
            is_anchor = (j == 4) and (np.round(deltas[i], 2) in deltas_anchor)
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8.8, fontweight='bold' if is_anchor else 'normal', color='#000000')
            if is_anchor:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                           fill=False, edgecolor='#0033cc', lw=2.2, zorder=5))

# Axis ticks and labels
ax.set_xticks(np.arange(len(alphas)))
ax.set_xticklabels([f'{int(a)}' for a in alphas], fontsize=9.5, fontweight='bold')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xlabel('Angle of Attack $\\alpha$ (Deg)', fontsize=11, fontweight='bold', labelpad=10)

ax.set_yticks(np.arange(len(deltas)))
ax.set_yticklabels(['0' if abs(d) < 1e-9 else f'{d:g}' for d in deltas], fontsize=9.5, fontweight='bold')
ax.set_ylabel('Trailing Edge Deflection Value $\\delta = z_{\\mathrm{tip}}/c$ (-)', fontsize=11, fontweight='bold', labelpad=10)

# Cell border gridlines
ax.set_xticks(np.arange(-0.5, len(alphas), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(deltas), 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.6)
ax.tick_params(which='minor', bottom=False, left=False, top=False)

# Colorbar and title
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
cbar.set_label('Exact $\\Delta C_l = C_{l,\\mathrm{XFOIL}} - C_{l,\\mathrm{RANS}}$ [-]', fontsize=10, fontweight='bold')

plt.title('TABLE 5.2 Replication: Difference in $C_l$ Values ($\\Delta C_l$) Between XFOIL and 2D RANS\n'
          '(Blue Boxes at $\\alpha = 4^\\circ$ = Your 4 Direct ANSYS Fluent Anchor Runs; "NaN" = XFOIL Non-Convergence)',
          fontsize=11.5, fontweight='bold', pad=18)

plt.tight_layout()
plt.savefig('Real_Table_5_2_Heatmap.png', bbox_inches='tight')
plt.show()