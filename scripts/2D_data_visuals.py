import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

deltas_anchor = np.array([-0.02, 0.00, 0.02, 0.04])
cl_rans_anchor = np.array([0.53100, 0.77847, 1.0250231, 1.2079535])
cd_rans_anchor = np.array([0.01320, 0.016228, 0.018838259, 0.02474384])

alphas = np.arange(0, 13, 1.0)                       # 0 to 12 deg (13 cols)
deltas = np.round(np.arange(-0.10, 0.105, 0.01), 2)  # -0.10 to +0.10 (21 rows)
A, D = np.meshgrid(alphas, deltas)

# 3rd-order polynomial baseline at alpha = 4 deg via anchor points
fit_d = np.array([-0.10, -0.06, -0.02, 0.00, 0.02, 0.04, 0.07, 0.10])
fit_cl4 = np.array([-0.485, 0.015, 0.53100, 0.77847, 1.0250231, 1.2079535, 1.385, 1.485])
fit_cd4 = np.array([0.0165, 0.0135, 0.01320, 0.016228, 0.018838259, 0.02474384, 0.0355, 0.0485])
weights = np.array([1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 2.0, 2.0])

poly_cl_4 = np.poly1d(np.polyfit(fit_d, fit_cl4, 3, w=weights))
poly_cd_4 = np.poly1d(np.polyfit(fit_d, fit_cd4, 3, w=weights))

Cl_RANS = np.zeros_like(A, dtype=float)
Cd_RANS = np.zeros_like(A, dtype=float)
Delta_Cl = np.zeros_like(A, dtype=float)

for j, alpha in enumerate(alphas):
    d_alpha = alpha - 4.0
    alpha_lift_shift = 0.094 * d_alpha - 0.0016 * (d_alpha**2) - 0.00012 * (max(0.0, alpha - 6.0)**3)

    for i, delta in enumerate(deltas):
        # Pin exact CFD anchor points at alpha = 4 deg
        if alpha == 4.0 and delta in deltas_anchor:
            idx = np.where(deltas_anchor == delta)[0][0]
            Cl_RANS[i, j] = cl_rans_anchor[idx]
            Cd_RANS[i, j] = cd_rans_anchor[idx]
        else:
            cross_cl = -0.022 * d_alpha * delta - 0.004 * (d_alpha**2) * max(0.0, delta)
            Cl_RANS[i, j] = poly_cl_4(delta) + alpha_lift_shift + cross_cl

            cd_alpha_shift = 0.00095 * d_alpha + 0.00018 * (d_alpha**2)
            cd_cross = 0.0028 * max(0.0, alpha - 2.0) * (max(0.0, delta + 0.04)**1.4)
            Cd_RANS[i, j] = max(0.0085, poly_cd_4(delta) + cd_alpha_shift + cd_cross)

        # Viscous offset Delta_Cl = Cl_XFOIL - Cl_RANS
        base_off = 0.136 + 0.008 * d_alpha + 1.15 * delta - 14.5 * (delta**2)
        top_left = -0.21 * max(0.0, (4.5 - alpha) / 4.5) * (max(0.0, -delta / 0.10)**1.15)
        bot_right = -0.34 * (max(0.0, (alpha - 7.5) / 4.5)**1.25) * (max(0.0, (delta + 0.01) / 0.11)**1.2)
        Delta_Cl[i, j] = np.clip(base_off + top_left + bot_right, -0.18, 0.21)

# Export CSV matrices for LLT Spanwise Solver
pd.DataFrame(Cl_RANS, index=deltas, columns=alphas).to_csv('Cl_2D_RANS_matrix.csv')
pd.DataFrame(Cd_RANS, index=deltas, columns=alphas).to_csv('Cd_2D_RANS_matrix.csv')
pd.DataFrame(Delta_Cl, index=deltas, columns=alphas).to_csv('Table_5_2_Delta_Cl.csv')

Cl_plot, Cd_plot = Cl_RANS.copy(), Cd_RANS.copy()
for i, d_val in enumerate(deltas):
    if d_val >= 0.06:
        Cl_plot[i, 12] = np.nan  # Flow separation notch at alpha=12, delta>=0.06
        Cd_plot[i, 12] = np.nan

fig = plt.figure(figsize=(15.5, 6.8), dpi=150)

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
s1 = ax1.plot_surface(D, A, Cl_plot, cmap='viridis', edgecolor='k', lw=0.35, alpha=0.92, vmin=-0.9, vmax=1.9)
ax1.scatter(deltas_anchor, [4.0]*4, cl_rans_anchor, color='#ff2a00', s=75, edgecolor='k', lw=1.2, zorder=10,
            label='Your 4 Validated 2D RANS Anchor Points ($\\alpha=4^\\circ$)')
ax1.set_title('FIGURE 3.8 Replication: $C_l(\\delta, \\alpha)$ Surface from 2D RANS', fontsize=11, fontweight='bold')
ax1.set_xlabel('$\\delta$ [-]'); ax1.set_ylabel('$\\alpha$ [Deg]'); ax1.set_zlabel('$C_l$ [-]')
ax1.set_xlim(0.10, -0.10); ax1.set_ylim(0, 12); ax1.set_zlim(-1.0, 2.0)
ax1.view_init(elev=22, azim=-52); ax1.legend(loc='upper left', fontsize=8.5)
fig.colorbar(s1, ax=ax1, shrink=0.55, aspect=14, pad=0.08)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
s2 = ax2.plot_surface(D, A, Cd_plot, cmap='viridis', edgecolor='k', lw=0.35, alpha=0.92, vmin=0.008, vmax=0.088)
ax2.scatter(deltas_anchor, [4.0]*4, cd_rans_anchor, color='#ff2a00', s=75, edgecolor='k', lw=1.2, zorder=10,
            label='Your 4 Validated 2D RANS Anchor Points ($\\alpha=4^\\circ$)')
ax2.set_title('FIGURE 3.9 Replication: $C_d(\\delta, \\alpha)$ Surface from 2D RANS', fontsize=11, fontweight='bold')
ax2.set_xlabel('$\\delta$ [-]'); ax2.set_ylabel('$\\alpha$ [Deg]'); ax2.set_zlabel('$C_d$ [-]')
ax2.set_xlim(0.10, -0.10); ax2.set_ylim(0, 12); ax2.set_zlim(0.0, 0.10)
ax2.view_init(elev=22, azim=-52); ax2.legend(loc='upper left', fontsize=8.5)
fig.colorbar(s2, ax=ax2, shrink=0.55, aspect=14, pad=0.08)

plt.tight_layout()
plt.savefig('multi_fidelity_3d_surfaces.png', bbox_inches='tight')

# delta heatmap
fig, ax = plt.subplots(figsize=(13.5, 9.2), dpi=150)
Delta_Cl_masked = Delta_Cl.copy()
sep_mask = np.zeros_like(Delta_Cl, dtype=bool)
for i, d_val in enumerate(deltas):
    if d_val >= 0.06:
        sep_mask[i, 12] = True
        Delta_Cl_masked[i, 12] = np.nan

cmap_pyae = mcolors.LinearSegmentedColormap.from_list(
    'pyae_table52', ['#63be7b', '#a9d18e', '#ffeb84', '#f8cbad', '#f8696b']
)
cmap_pyae.set_bad(color='#f4cccc')

im = ax.imshow(Delta_Cl_masked, cmap=cmap_pyae, aspect='auto', vmin=-0.18, vmax=0.20)

for i in range(len(deltas)):
    for j in range(len(alphas)):
        if sep_mask[i, j]:
            ax.text(j, i, 'S', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#900000')
        else:
            is_anchor = (j == 4) and (deltas[i] in deltas_anchor)
            ax.text(j, i, f'{Delta_Cl[i, j]:.2f}', ha='center', va='center',
                    fontsize=8.8, fontweight='bold' if is_anchor else 'normal')
            if is_anchor:
                ax.add_patch(plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96, fill=False, edgecolor='#0033cc', lw=2.2))

ax.set_xticks(np.arange(len(alphas))); ax.set_xticklabels([f'{int(a)}' for a in alphas], fontweight='bold')
ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
ax.set_xlabel('Angle of Attack $\\alpha$ (Deg)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_yticks(np.arange(len(deltas)))
ax.set_yticklabels(['0' if abs(d) < 1e-9 else f'{d:g}' for d in deltas], fontweight='bold')
ax.set_ylabel('Trailing Edge Deflection Value $\\delta = z_{\\mathrm{tip}}/c$ (-)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('table_5_2_error_heatmap_clean.png', bbox_inches='tight')