import os
import aerosandbox as asb
import numpy as np
import pandas as pd

# same as XFOIL
delta_range = np.linspace(-0.10, 0.10, 21)
alpha_range = np.linspace(-20.0, 20.0, 41)

Re = 6.87e5
mach = 0.03


def naca4_thickness(x, t=0.12):
  return (
      5
      * t
      * (
          0.2969 * np.sqrt(x)
          - 0.1260 * x
          - 0.3516 * x**2
          + 0.2843 * x**3
          - 0.1015 * x**4
      )
  )


def naca4412_camber_baseline(x, m=0.04, p=0.40):
  yc = np.where(
      x < p,
      m / p**2 * (2 * p * x - x**2),
      (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x**2),
  )
  dyc_dx = np.where(
      x < p, 2 * m / p**2 * (p - x), (2 * m / (1 - p) ** 2) * (p - x)
  )
  return yc, dyc_dx


def generate_hinged_flap_naca4412(delta, x_h=0.70, n_points=200):
  beta = np.linspace(0, np.pi, n_points // 2)
  x = (1.0 - np.cos(beta)) / 2.0

  yc_base, dyc_base = naca4412_camber_baseline(x)
  yc_h, _ = naca4412_camber_baseline(np.array([x_h]))
  y0_h = yc_h[0]

  # linear deflection aft of 0.70c 
  flap_slope = (-delta - y0_h) / (1.0 - x_h)
  yc_flap = y0_h + (x - x_h) * flap_slope

  yc = np.where(x < x_h, yc_base, yc_flap)
  dyc = np.where(x < x_h, dyc_base, flap_slope)

  theta = np.arctan(dyc)
  yt = naca4_thickness(x, t=0.12)

  xu = x - yt * np.sin(theta)
  yu = yc + yt * np.cos(theta)
  xl = x + yt * np.sin(theta)
  yl = yc - yt * np.cos(theta)

  x_coords = np.concatenate([xu[::-1], xl[1:]])
  y_coords = np.concatenate([yu[::-1], yl[1:]])
  return np.column_stack((x_coords, y_coords))


Cl_flap = np.full((len(delta_range), len(alpha_range)), np.nan)
Cd_flap = np.full((len(delta_range), len(alpha_range)), np.nan)

print(
    f"starting XFOIL run for hinged flap across"
    f" {len(delta_range) * len(alpha_range)} conditions..."
)

for i, delta in enumerate(delta_range):
  coords = generate_hinged_flap_naca4412(delta=delta, x_h=0.70, n_points=200)
  airfoil = asb.Airfoil(name=f"hinged_flap_d{delta:.2f}", coordinates=coords)

  cmd = "xfoil" if os.name != "nt" else "bin/xfoil.exe"
  if os.path.exists("bin/xfoil"):
    cmd = "bin/xfoil"

  xf = asb.XFoil(
      airfoil=airfoil,
      Re=Re,
      mach=mach,
      max_iter=100,
      xfoil_command=cmd,
      verbose=False,
  )

  results = xf.alpha(alpha_range)

  if "alpha" in results and len(results["alpha"]) > 0:
    converged_cls = results["CL"]
    converged_cds = results["CD"]
    converged_alphas = results["alpha"]

    for k in range(len(results["alpha"])):
      a = converged_alphas[k]
      matching_cols = np.where(np.isclose(alpha_range, a, atol=1e-2))[0]
      if len(matching_cols) > 0:
        col_id = matching_cols[0]
        Cl_flap[i, col_id] = converged_cls[k]
        Cd_flap[i, col_id] = converged_cds[k]

  converged_count = np.sum(~np.isnan(Cl_flap[i, :]))
  print(
      f"Flap delta = {delta:+.2f} m | Converged:"
      f" {converged_count}/{len(alpha_range)}"
  )

os.makedirs("data", exist_ok=True)
col_names = [f"alpha_{int(round(a))}deg" for a in alpha_range]

df_cl = pd.DataFrame(Cl_flap, index=np.round(delta_range, 2), columns=col_names)
df_cl.index.name = "delta"
df_cl.to_csv("data/Cl_flap.csv")

df_cd = pd.DataFrame(Cd_flap, index=np.round(delta_range, 2), columns=col_names)
df_cd.index.name = "delta"
df_cd.to_csv("data/Cd_flap.csv")

print("generated and saved 'data/Cl_flap.csv' and 'data/Cd_flap.csv'.")