
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.optimize import linear_sum_assignment


def load_or_generate_wafers(npz_path: str = "data/sample_wafers.npz", grid_size: int = 26) -> dict:
    """Load pre-extracted WM-811K wafers if present; otherwise generate realistic mini samples."""
    path = Path(npz_path)
    if path.exists():
        raw = np.load(path, allow_pickle=True)
        wafers = {}
        for key in raw.files:
            wafers[key] = raw[key].item()
        if wafers:
            return wafers

    c = (grid_size - 1) / 2.0
    y, x = np.ogrid[:grid_size, :grid_size]
    dist = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    wafer_mask = dist <= (grid_size / 2.0 - 0.5)

    patterns = ["Edge-Ring", "Scratch", "Center", "Donut", "Random"]
    wafers = {}

    for idx, pat in enumerate(patterns):
        rng = np.random.default_rng(100 + idx * 23)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        grid[wafer_mask] = 1

        # Sparse baseline random probe failures (~2.5%)
        grid[(rng.random((grid_size, grid_size)) < 0.025) & wafer_mask] = 2

        if pat == "Edge-Ring":
            ring = (dist >= (grid_size / 2.0 - 2.4)) & wafer_mask
            grid[ring & (rng.random((grid_size, grid_size)) < 0.72)] = 2
        elif pat == "Scratch":
            for t in np.linspace(-0.78, 0.78, 45):
                sx = int(round(c + t * (grid_size * 0.42)))
                sy = int(round(c + (0.48 * t + 0.10 * np.sin(4 * t)) * (grid_size * 0.42)))
                for dy in (-1, 0):
                    for dx in (-1, 0):
                        if 0 <= sy + dy < grid_size and 0 <= sx + dx < grid_size:
                            if wafer_mask[sy + dy, sx + dx] and rng.random() < 0.85:
                                grid[sy + dy, sx + dx] = 2
        elif pat == "Center":
            center_zone = (dist <= grid_size * 0.19) & wafer_mask
            grid[center_zone & (rng.random((grid_size, grid_size)) < 0.78)] = 2
        elif pat == "Donut":
            donut_zone = (dist >= grid_size * 0.16) & (dist <= grid_size * 0.29) & wafer_mask
            grid[donut_zone & (rng.random((grid_size, grid_size)) < 0.75)] = 2
        elif pat == "Random":
            grid[(rng.random((grid_size, grid_size)) < 0.09) & wafer_mask] = 2

        # CircuitNet-inspired normalized thermal hotspot grid [0, 1]
        hx, hy = c + rng.uniform(-3.0, 3.0), c + rng.uniform(-3.0, 3.0)
        hotspot = np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * (grid_size * 0.23) ** 2))
        thermal = np.clip(0.22 + 0.68 * hotspot + rng.normal(0, 0.03, (grid_size, grid_size)), 0.05, 1.0)
        thermal[~wafer_mask] = np.nan

        wafers[pat] = {
            "wafer_map": grid,
            "thermal_map": thermal,
            "pattern": pat,
            "source": "Synthetic WM-811K Sample",
        }
    return wafers


def score_wafer_risk(wafer_map: np.ndarray, thermal_map: np.ndarray) -> dict:
    """Compute post-bonding die survival probabilities by fusing 2D spatial neighbor-defect
    density with thermal stress and radial edge warpage."""
    wafer_map = np.asarray(wafer_map)
    valid_mask = wafer_map > 0
    pass_mask = wafer_map == 1
    fail_mask = (wafer_map == 2).astype(float)

    kernel = np.array([
        [0.25, 0.50, 0.50, 0.50, 0.25],
        [0.50, 1.00, 1.00, 1.00, 0.50],
        [0.50, 1.00, 0.00, 1.00, 0.50],
        [0.50, 1.00, 1.00, 1.00, 0.50],
        [0.25, 0.50, 0.50, 0.50, 0.25],
    ])
    neighbor_fails = convolve(fail_mask, kernel, mode="constant", cval=0.0)
    neighbor_valid = convolve(valid_mask.astype(float), kernel, mode="constant", cval=0.0)
    neighbor_density = np.divide(neighbor_fails, np.maximum(neighbor_valid, 1.0))

    h, w = wafer_map.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r_norm = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (max(h, w) / 2.0)
    edge_stress = np.clip(r_norm ** 2, 0.0, 1.0)

    therm_clean = np.nan_to_num(thermal_map, nan=0.0)

    latent_risk = np.clip(
        0.58 * neighbor_density + 0.24 * therm_clean + 0.18 * edge_stress,
        0.0,
        1.0,
    )

    survival_prob = np.full_like(latent_risk, np.nan, dtype=float)
    survival_prob[wafer_map == 2] = 0.0
    survival_prob[pass_mask] = np.clip(0.995 - 0.52 * latent_risk[pass_mask], 0.35, 0.995)

    risk_overlay = np.full_like(latent_risk, np.nan, dtype=float)
    risk_overlay[pass_mask] = latent_risk[pass_mask]

    return {
        "survival_prob": survival_prob,
        "latent_risk": risk_overlay,
        "neighbor_density": neighbor_density,
    }


def optimize_chiplet_pairing(
    wafer_a: dict,
    wafer_b: dict,
    thermal_coupling: float = 0.24,
    monthly_wafer_pairs: int = 500,
    package_value_usd: float = 2500.0,
) -> dict:
    """Compare Standard Sequential Wafer Stacking against StackYield 3D Optimal Pairing."""
    map_a = np.asarray(wafer_a["wafer_map"])
    map_b = np.asarray(wafer_b["wafer_map"])

    res_a = score_wafer_risk(map_a, wafer_a["thermal_map"])
    res_b = score_wafer_risk(map_b, wafer_b["thermal_map"])

    coords_a = np.column_stack(np.nonzero(map_a == 1))
    coords_b = np.column_stack(np.nonzero(map_b == 1))
    n_pairs = min(len(coords_a), len(coords_b))

    pa = res_a["survival_prob"][coords_a[:, 0], coords_a[:, 1]]
    pb = res_b["survival_prob"][coords_b[:, 0], coords_b[:, 1]]
    ta = np.asarray(wafer_a["thermal_map"])[coords_a[:, 0], coords_a[:, 1]]
    tb = np.asarray(wafer_b["thermal_map"])[coords_b[:, 0], coords_b[:, 1]]

    # Full pairwise stacking survival matrix P_stack(i, j)
    p_matrix = np.clip(
        pa[:, None] * pb[None, :] * (1.0 - thermal_coupling * (ta[:, None] * tb[None, :])),
        0.01,
        0.995,
    )

    # 1. Baseline: Standard sequential wafer-index pairing (blind to cross-tier risk)
    base_idx_a = np.arange(n_pairs)
    base_idx_b = np.arange(n_pairs)
    baseline_probs = p_matrix[base_idx_a, base_idx_b]

    # 2. StackYield 3D: Supermodular bipartite optimization maximizing expected working stacks
    opt_idx_a, opt_idx_b = linear_sum_assignment(-p_matrix)
    optimal_probs = p_matrix[opt_idx_a, opt_idx_b]

    baseline_yield = float(np.mean(baseline_probs))
    optimal_yield = float(np.mean(optimal_probs))
    yield_lift = optimal_yield - baseline_yield

    extra_good_stacks_per_pair = yield_lift * n_pairs
    annual_stacks_saved = extra_good_stacks_per_pair * monthly_wafer_pairs * 12

    # Sustainability conversion factors per prevented 3D package scrap:
    # ~42 kWh cleanroom power, ~185 gal Ultra-Pure Water (UPW), ~19 kg CO2e (incl. NF3/SF6 abatement)
    annual_usd_saved = annual_stacks_saved * package_value_usd
    annual_mwh_saved = (annual_stacks_saved * 42.0) / 1000.0
    annual_upw_kgal_saved = (annual_stacks_saved * 185.0) / 1000.0
    annual_co2_tons_saved = (annual_stacks_saved * 19.0) / 1000.0

    top_pairs_df = pd.DataFrame({
        "Logic Die (Row, Col)": [f"({r}, {c})" for r, c in coords_a[opt_idx_a]],
        "Memory Die (Row, Col)": [f"({r}, {c})" for r, c in coords_b[opt_idx_b]],
        "Logic Survival (%)": np.round(pa[opt_idx_a] * 100, 1),
        "Memory Survival (%)": np.round(pb[opt_idx_b] * 100, 1),
        "Thermal Overlap": np.round(ta[opt_idx_a] * tb[opt_idx_b], 3),
        "3D Stack Survival (%)": np.round(optimal_probs * 100, 1),
    }).sort_values("3D Stack Survival (%)", ascending=False).head(10)

    return {
        "n_pairs": n_pairs,
        "baseline_yield": baseline_yield,
        "optimal_yield": optimal_yield,
        "yield_lift": yield_lift,
        "extra_stacks_per_pair": extra_good_stacks_per_pair,
        "high_rel_base": int((baseline_probs >= 0.85).sum()),
        "high_rel_opt": int((optimal_probs >= 0.85).sum()),
        "baseline_probs": baseline_probs,
        "optimal_probs": optimal_probs,
        "res_a": res_a,
        "res_b": res_b,
        "annual_usd_saved": annual_usd_saved,
        "annual_mwh_saved": annual_mwh_saved,
        "annual_upw_kgal_saved": annual_upw_kgal_saved,
        "annual_co2_tons_saved": annual_co2_tons_saved,
        "top_pairs_df": top_pairs_df,
    }
