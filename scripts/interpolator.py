import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


class SectionAeroDatabase:

  def __init__(
      self, cl_path="data/Cl_xfoil.csv", cd_path="data/Cd_xfoil.csv"
  ):
    # Load physical datasets
    df_cl = pd.read_csv(cl_path, index_col=0)
    df_cd = pd.read_csv(cd_path, index_col=0)

    # Extract coordinates
    self.deltas = df_cl.index.to_numpy(dtype=float)
    self.alphas = np.array(
        [float(col.split("_")[1].replace("deg", "")) for col in df_cl.columns]
    )

    # Clean unconverged dropouts across angle sweeps (interpolate linear gaps, backfill/forwardfill edges)
    cl_clean = (
        df_cl.interpolate(method="linear", axis=1)
        .bfill(axis=1)
        .ffill(axis=1)
        .to_numpy()
    )
    cd_clean = (
        df_cd.interpolate(method="linear", axis=1)
        .bfill(axis=1)
        .ffill(axis=1)
        .to_numpy()
    )

    # Ensure strictly increasing axes for RegularGridInterpolator
    sort_a_idx = np.argsort(self.alphas)
    self.alphas = self.alphas[sort_a_idx]
    cl_clean = cl_clean[:, sort_a_idx]
    cd_clean = cd_clean[:, sort_a_idx]

    sort_d_idx = np.argsort(self.deltas)
    self.deltas = self.deltas[sort_d_idx]
    cl_clean = cl_clean[sort_d_idx, :]
    cd_clean = cd_clean[sort_d_idx, :]

    # Build 2D grid interpolators
    self._interp_cl = RegularGridInterpolator(
        (self.deltas, self.alphas),
        cl_clean,
        bounds_error=False,
        fill_value=None,
    )
    self._interp_cd = RegularGridInterpolator(
        (self.deltas, self.alphas),
        cd_clean,
        bounds_error=False,
        fill_value=None,
    )

  def get_cl(self, delta: float, alpha: float) -> float:
    """Returns section lift coefficient Cl at an arbitrary (delta, alpha)."""
    pt = np.array([[delta, alpha]])
    return float(self._interp_cl(pt)[0])

  def get_cd(self, delta: float, alpha: float) -> float:
    """Returns section drag coefficient Cd at an arbitrary (delta, alpha)."""
    pt = np.array([[delta, alpha]])
    return float(self._interp_cd(pt)[0])

  def find_optimal_delta(
      self, alpha_eff: float, target_cl: float = None
  ) -> float:
    """Closed-loop query:

    - If target_cl is specified, finds delta that minimizes |Cl(delta) -
    target_cl| at alpha_eff.
    - If target_cl is None, finds delta that maximizes aerodynamic efficiency
    Cl/Cd at alpha_eff.
    """
    # Sample candidate deflections across physical travel
    candidate_deltas = np.linspace(self.deltas.min(), self.deltas.max(), 200)
    query_points = np.column_stack(
        [candidate_deltas, np.full_like(candidate_deltas, alpha_eff)]
    )

    cls = self._interp_cl(query_points)
    cds = self._interp_cd(query_points)

    if target_cl is not None:
      best_idx = np.argmin(np.abs(cls - target_cl))
    else:
      # Filter for attached, positive-lift regime
      valid_mask = (cds > 0.0) & (cls > 0.0)
      if not np.any(valid_mask):
        return 0.0
      efficiency = np.where(valid_mask, cls / cds, -1.0)
      best_idx = np.argmax(efficiency)

    return float(candidate_deltas[best_idx])


if __name__ == "__main__":
  db = SectionAeroDatabase()

  # Test forward evaluation
  test_a = 4.0
  test_d = 0.04
  cl_sample = db.get_cl(delta=test_d, alpha=test_a)
  cd_sample = db.get_cd(delta=test_d, alpha=test_a)
  print(
      f"Forward Query: alpha={test_a} deg, delta={test_d:+.2f} m -> Cl ="
      f" {cl_sample:.3f}, Cd = {cd_sample:.4f}"
  )

  # Test inverse evaluation for target lift
  req_cl = 1.0
  d_target = db.find_optimal_delta(alpha_eff=test_a, target_cl=req_cl)
  print(
      f"Inverse Query (Target Cl={req_cl} at alpha={test_a} deg): Commanded"
      f" delta* = {d_target:+.4f} m"
  )

  # Test inverse evaluation for maximum efficiency (Cl/Cd)
  d_max_eff = db.find_optimal_delta(alpha_eff=test_a, target_cl=None)
  print(
      f"Inverse Query (Max Cl/Cd at alpha={test_a} deg): Commanded delta* ="
      f" {d_max_eff:+.4f} m"
  )