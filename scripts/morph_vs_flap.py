import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


# Discretize chord
n_pts = 1000
beta = np.linspace(0, np.pi, n_pts)
x = (1.0 - np.cos(beta)) / 2.0
x_h = 0.70
delta = 0.08  # Representative downward deflection

# 1. Continuous Morphing Camber (2nd Order Polynomial)
yc_base, dyc_base = naca4412_camber_baseline(x)
yc_h, dyc_h = naca4412_camber_baseline(np.array([x_h]))
y0_h, dy0_h = yc_h[0], dyc_h[0]
l = 1.0 - x_h
A = (-delta - dy0_h * l - y0_h) / (l**2)
B = dy0_h
C = y0_h
yc_morph = np.where(
    x < x_h, yc_base, A * (x - x_h) ** 2 + B * (x - x_h) + C
)
dyc_morph = np.where(x < x_h, dyc_base, 2 * A * (x - x_h) + B)

# 2. Discrete Hinged Flap Camber (Linear aft of 0.70c)
yc_flap = np.where(
    x < x_h, yc_base, y0_h + (x - x_h) * ((-delta - y0_h) / (1.0 - x_h))
)
dyc_flap = np.where(x < x_h, dyc_base, (-delta - y0_h) / (1.0 - x_h))

# Apply thickness distribution
yt = naca4_thickness(x, t=0.12)

# Morphing coordinates
th_m = np.arctan(dyc_morph)
xu_m, yu_m = x - yt * np.sin(th_m), yc_morph + yt * np.cos(th_m)
xl_m, yl_m = x + yt * np.sin(th_m), yc_morph - yt * np.cos(th_m)

# Hinged flap coordinates
th_f = np.arctan(dyc_flap)
xu_f, yu_f = x - yt * np.sin(th_f), yc_flap + yt * np.cos(th_f)
xl_f, yl_f = x + yt * np.sin(th_f), yc_flap - yt * np.cos(th_f)

# Plotting
plt.figure(figsize=(12, 4.5), dpi=300)
plt.plot(
    np.concatenate([xu_m[::-1], xl_m]),
    np.concatenate([yu_m[::-1], yl_m]),
    "b-",
    linewidth=2,
    label=r"Continuous Morphed Airfoil ($\delta = +0.08$ m)",
)
plt.plot(
    np.concatenate([xu_f[::-1], xl_f]),
    np.concatenate([yu_f[::-1], yl_f]),
    "r--",
    linewidth=1.8,
    label=r"Conventional Plain Flap ($\delta = +0.08$ m)",
)

plt.axvline(
    x=x_h,
    color="gray",
    linestyle=":",
    linewidth=1.2,
    label=f"Hinge Location ($x_h = {x_h}$)",
)

plt.title(
    "Aerodynamic Geometry: Continuous Morphing vs. Conventional Hinged Flap",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Chord Station $x/c$", fontsize=11)
plt.ylabel("Thickness Elevation $y/c$", fontsize=11)
plt.axis("equal")
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(loc="lower left", fontsize=9)
plt.tight_layout()

os.makedirs("docs", exist_ok=True)
plt.savefig("docs/morph_vs_flap_comparison.png")
print("Saved geometry comparison to docs/morph_vs_flap_comparison.png")
# plt.show()