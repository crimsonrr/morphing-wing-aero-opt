rho = 1.225
mu = 1.789e-5
c = 0.2  # chord in meters
V = 50.17  #  inlet velocity in m/s
y_plus = 1.0

Re = (rho * V * c) / mu
Cf = 0.026 * (Re ** (-1 / 7))
tau_w = 0.5 * Cf * rho * (V**2)
u_tau = (tau_w / rho) ** 0.5
first_layer_height = (y_plus * mu) / (rho * u_tau)

print(f"Reynolds Number: {Re:.2e}")
print(f"First Layer Height (y1) for y+ = 1: {first_layer_height:.4e} m")
print(f"In millimeters: {first_layer_height * 1e3:.4f} mm")