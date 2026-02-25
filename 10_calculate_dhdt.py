"""
Cálculo de dh/dt (taxa de mudança de elevação)
Versão 5.1b - Baseado no fitsec.py do CAPTOOLKIT

Correções vs v5.1:
  - Fix: write_hdf5 compatível com metadados escalares/string
  - Fix: séries temporais com t_bin de tamanho variável entre tiles

Correções vs v5.0:
  - Normalização de dx/dy na matriz de design
  - RATE_LIM = 15 m/ano
  - P2_LIMIT = 2 m/ano²
  - RESID_LIMIT = 5 m

Credits:
    captoolkit - JPL Cryosphere Altimetry Processing Toolkit
    Johan Nilsson, Fernando Paolo, Alex Gardner (JPL/NASA)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import h5py
from tqdm import tqdm
from scipy import linalg
from scipy.spatial import cKDTree
import gc

# ============================================
# DETECTAR DIRETÓRIO
# ============================================

if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).parent
else:
    SCRIPT_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")

sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    FITSEC_PARAMS,
    TILES_WINTER_DIR,
    DHDT_WINTER_DIR,
    LOGS_DIR,
    GRID_RESOLUTION
)

sys.path.insert(0, str(SCRIPT_DIR / 'utils'))
from io_utils import read_hdf5
from geodetic_utils import xy_to_lonlat

print("=" * 70)
print("CÁLCULO DE dh/dt — VERSÃO 5.1b (FITSEC-BASED, CORRIGIDO)")
print("Baseado em: fitsec.py do CAPTOOLKIT (JPL/NASA)")
print("=" * 70)

# ============================================
# PARÂMETROS
# ============================================

SEARCH_RADIUS = FITSEC_PARAMS['search_radius']
N_RELOC = 1
MIN_POINTS = FITSEC_PARAMS['min_points']

POLY_ORDER = FITSEC_PARAMS['poly_order']
TEMPORAL_ORDER = 2
MAX_ITERATIONS = FITSEC_PARAMS['max_iterations']
SIGMA_THRESHOLD = FITSEC_PARAMS['sigma_threshold']
RESID_LIMIT = 5.0

DT_LIM = 0.1
RATE_LIM = 15.0
P2_LIMIT = 2.0

USE_WEIGHTS = True
CORR_LENGTH = SEARCH_RADIUS * 0.5

TS_STEP = 1.0 / 12
TS_WINDOW = 3.0 / 12
GENERATE_TIMESERIES = True

T_REF = None

print(f"\nParâmetros (equivalentes ao fitsec.py):")
print(f"  Grade: {GRID_RESOLUTION} m (-d)")
print(f"  Raio de busca: {SEARCH_RADIUS} m (-r)")
print(f"  Relocações: {N_RELOC} (-q)")
print(f"  Pontos mínimos: {MIN_POINTS} (-z)")
print(f"  Modelo espacial: ordem {POLY_ORDER} (-m t)")
print(f"  Modelo temporal: ordem {TEMPORAL_ORDER} (-m _ p)")
print(f"  Iterações: {MAX_ITERATIONS} (-i)")
print(f"  Sigma threshold: {SIGMA_THRESHOLD} (-w nsigma)")
print(f"  Limite resíduo: {RESID_LIMIT} m (-w reslim)")
print(f"  Limite taxa: ±{RATE_LIM} m/ano (-l)")
print(f"  Limite aceleração: ±{P2_LIMIT} m/ano² (novo)")
print(f"  Ponderação: {USE_WEIGHTS} (-a)")
print(f"  Séries temporais: {GENERATE_TIMESERIES}")


# ============================================
# FUNÇÃO: SALVAR HDF5 (robusta)
# ============================================

def save_hdf5_robust(filepath, data_dict):
    """
    Salva dicionário em HDF5, tratando corretamente:
      - arrays numpy
      - escalares (int, float)
      - strings
      - None (ignora)

    Isso evita o erro 'only length-1 arrays can be converted to Python scalars'
    que ocorre quando write_hdf5 genérica não distingue tipos.
    """
    with h5py.File(filepath, 'w') as f:
        for key, val in data_dict.items():
            if val is None:
                continue

            try:
                if isinstance(val, np.ndarray):
                    f.create_dataset(key, data=val)

                elif isinstance(val, (list, tuple)):
                    f.create_dataset(key, data=np.asarray(val))

                elif isinstance(val, str):
                    f.attrs[key] = val

                elif isinstance(val, (int, float, np.integer, np.floating)):
                    f.attrs[key] = val

                elif isinstance(val, bool):
                    f.attrs[key] = int(val)

                else:
                    # Tentar converter
                    f.attrs[key] = val

            except Exception as e:
                # Se falhar, salvar como atributo string
                try:
                    f.attrs[key] = str(val)
                except:
                    pass  # Ignorar silenciosamente


# ============================================
# FUNÇÕES AUXILIARES (do fitsec.py)
# ============================================

def mad_std(x):
    """Robust standard deviation using MAD."""
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))


def get_radius_idx(x, y, x0, y0, r, tree, n_rel=0):
    """Get indices within radius with optional relocation."""
    idx = tree.query_ball_point([x0, y0], r)

    reloc_dist = 0.0

    if n_rel < 1 or len(idx) < 2:
        return idx, reloc_dist, x0, y0

    for k in range(n_rel):
        x0_new = np.median(x[idx])
        y0_new = np.median(y[idx])

        reloc_dist = np.hypot(x0_new - x0, y0_new - y0)

        if reloc_dist > r:
            break

        idx = tree.query_ball_point([x0_new, y0_new], r)
        x0, y0 = x0_new, y0_new

        if k + 1 == n_rel:
            break

    return idx, reloc_dist, x0, y0


def build_design_matrix(dx, dy, dt, topo_order=2, temp_order=2):
    """
    Monta a matriz de design com normalização dx/dy.

    Returns:
      A: design matrix (n × m)
      scales: (x_scale, y_scale)
    """
    n = len(dx)

    x_scale = np.std(dx) if np.std(dx) > 1e-6 else 1.0
    y_scale = np.std(dy) if np.std(dy) > 1e-6 else 1.0
    dx_n = dx / x_scale
    dy_n = dy / y_scale

    cols = [np.ones(n)]

    # Temporal
    cols.append(dt)
    if temp_order >= 2:
        cols.append(0.5 * dt**2)

    # Espacial (normalizado)
    if topo_order >= 1:
        cols.append(dx_n)
        cols.append(dy_n)
        cols.append(dx_n * dy_n)

    if topo_order >= 2:
        cols.append(dx_n**2)
        cols.append(dy_n**2)

    return np.column_stack(cols), (x_scale, y_scale)


def lstsq_iterative(A, z, w=None, n_iter=5, n_sigma=3.0, rlim=5.0):
    """
    Mínimos quadrados iterativos com rejeição de outliers.
    Retorna coeficientes + erros formais da covariância.
    """
    n, m = A.shape
    mask = np.ones(n, dtype=bool)

    xhat = None
    ehat = None
    rmse = np.nan

    for iteration in range(n_iter):

        A_f = A[mask]
        z_f = z[mask]
        n_f = int(np.sum(mask))

        if n_f < m + 1:
            break

        if w is not None:
            w_f = w[mask]
            w_sum = np.sum(w_f)
            if w_sum <= 0:
                break
            w_norm = w_f / w_sum * n_f
            W = np.sqrt(w_norm)
            Aw = A_f * W[:, np.newaxis]
            zw = z_f * W
        else:
            Aw = A_f
            zw = z_f

        try:
            xhat_new, residuals, rank, sv = linalg.lstsq(Aw, zw)

            if rank < m:
                break

            xhat = xhat_new

            h_pred = A_f @ xhat
            res = z_f - h_pred
            rmse = np.nanstd(res)

            if w is not None:
                AtWA = Aw.T @ Aw
            else:
                AtWA = A_f.T @ A_f

            try:
                AtWA_inv = linalg.inv(AtWA)
                sigma2 = np.sum(res**2) / max(n_f - m, 1)
                cov_matrix = sigma2 * AtWA_inv
                ehat = np.sqrt(np.abs(np.diag(cov_matrix)))
            except linalg.LinAlgError:
                ehat = np.full(m, np.nan)

            if rmse < 1e-10:
                break

            mad = mad_std(res)
            if mad > 0:
                outliers_sigma = np.abs(res) > n_sigma * mad
            else:
                outliers_sigma = np.zeros(n_f, dtype=bool)

            outliers_abs = np.abs(res) > rlim
            outliers = outliers_sigma | outliers_abs

            if not np.any(outliers):
                break

            mask_indices = np.where(mask)[0]
            mask[mask_indices[outliers]] = False

        except (linalg.LinAlgError, ValueError):
            break

    if xhat is None:
        return None, None, mask, np.nan

    if ehat is None:
        ehat = np.full(len(xhat), np.nan)

    return xhat, ehat, mask, rmse


def resample_timeseries(t, h, w, t_bin, window=3.0/12, use_weights=False):
    """Binar série temporal com janela deslizante."""
    n_bins = len(t_bin)
    z_bin = np.full(n_bins, np.nan)
    e_bin = np.full(n_bins, np.nan)

    for i in range(n_bins):
        idx = (t >= (t_bin[i] - 0.5 * window)) & \
              (t <= (t_bin[i] + 0.5 * window))

        z_w = h[idx]
        w_w = w[idx]

        if len(z_w) == 0:
            continue

        med = np.median(z_w)
        mad = 1.4826 * np.median(np.abs(z_w - med))
        if mad > 0:
            good = np.abs(z_w - med) <= 3.5 * mad
        else:
            good = np.ones(len(z_w), dtype=bool)

        if np.sum(good) == 0:
            continue

        z_g = z_w[good]
        w_g = w_w[good]

        if use_weights and np.sum(w_g) > 0:
            z_bin[i] = np.sum(w_g * z_g) / np.sum(w_g)
            e_bin[i] = np.sqrt(np.sum(w_g * (z_g - z_bin[i])**2) / np.sum(w_g))
        else:
            z_bin[i] = np.mean(z_g)
            e_bin[i] = np.std(z_g)

    return z_bin, e_bin


# ============================================
# FUNÇÃO: PROCESSAR UM NÓ DA GRADE
# ============================================

def fit_node(x_obs, y_obs, h_obs, t_obs, s_obs,
             x_node, y_node,
             topo_order=2, temp_order=2,
             max_iterations=5, sigma_threshold=3.0, resid_limit=5.0,
             min_points=15, use_weights=True, corr_length=1000.0,
             t_ref=None, t_bin=None):
    """
    Ajusta modelo espaço-temporal num nó da grade.
    Returns dict ou None.
    """

    n = len(h_obs)
    if n < min_points:
        return None

    t_min, t_max = t_obs.min(), t_obs.max()
    dt_total = t_max - t_min

    if dt_total < DT_LIM:
        return None

    dx = x_obs - x_node
    dy = y_obs - y_node

    if t_ref is None:
        t_ref_local = float(np.mean(t_obs))
    else:
        t_ref_local = float(t_ref)

    dt = t_obs - t_ref_local
    dr = np.sqrt(dx**2 + dy**2)

    # Pesos
    if use_weights:
        sv = s_obs**2
        sv = np.where(sv < 1e-6, 1.0, sv)
        wc = 1.0 / (sv * (1.0 + (dr / corr_length)**2))
    else:
        wc = np.ones(n)

    # Matriz de design
    A, (x_scale, y_scale) = build_design_matrix(
        dx, dy, dt, topo_order=topo_order, temp_order=temp_order
    )

    # LSQ iterativo
    xhat, ehat, mask, rmse = lstsq_iterative(
        A, h_obs, w=wc if use_weights else None,
        n_iter=max_iterations, n_sigma=sigma_threshold, rlim=resid_limit
    )

    if xhat is None:
        return None

    n_used = int(np.sum(mask))
    if n_used < min_points:
        return None

    # Extrair parâmetros
    p0 = float(xhat[0])
    p1 = float(xhat[1])
    p0_err = float(ehat[0])
    p1_err = float(ehat[1])

    if np.abs(p1) > RATE_LIM:
        return None

    # Aceleração com filtro
    if temp_order >= 2:
        p2 = float(xhat[2])
        p2_err = float(ehat[2])
        if np.abs(p2) > P2_LIMIT:
            p2 = np.nan
            p2_err = np.nan
    else:
        p2 = np.nan
        p2_err = np.nan

    dmin = float(np.min(dr[mask]))

    # Resíduos
    h_pred = A[mask] @ xhat
    residuals = h_obs[mask] - h_pred

    mad_val = mad_std(residuals)
    if mad_val > 0:
        ibad = np.abs(residuals) > sigma_threshold * mad_val
    else:
        ibad = np.zeros(n_used, dtype=bool)

    residuals[ibad] = np.nan
    rmse_final = float(np.nanstd(residuals))

    # Série temporal
    sec_t = None
    rms_t = None

    if GENERATE_TIMESERIES and t_bin is not None and n_used > min_points:
        if temp_order >= 2:
            temporal_cols = [1, 2]
        else:
            temporal_cols = [1]

        A_masked = A[mask]
        residuals_clean = residuals.copy()
        residuals_clean[np.isnan(residuals_clean)] = 0.0

        h_temporal = residuals_clean + A_masked[:, temporal_cols] @ xhat[temporal_cols]

        inan = ~np.isnan(h_obs[mask])
        inan &= ~ibad

        if np.sum(inan) > 5:
            sec_t, rms_t = resample_timeseries(
                t_obs[mask][inan], h_temporal[inan], wc[mask][inan],
                t_bin, window=TS_WINDOW, use_weights=use_weights
            )

    result = {
        'p0': p0,
        'p1': p1,
        'p2': p2,
        'p0_err': p0_err,
        'p1_err': p1_err,
        'p2_err': p2_err,
        'rmse': rmse_final,
        'nobs': n_used,
        'dmin': dmin,
        'tspan': dt_total,
        't_ref': t_ref_local,
    }

    if sec_t is not None:
        result['sec_t'] = sec_t
        result['rms_t'] = rms_t

    return result


# ============================================
# FUNÇÃO: PROCESSAR UM TILE
# ============================================

def process_tile(tile_file, output_dir, grid_res):
    """Processa um tile completo."""

    tile_data = read_hdf5(tile_file)

    required = ['x', 'y', 'h_li']
    if not all(k in tile_data for k in required):
        return None

    if 't_year' in tile_data:
        t_key = 't_year'
    elif 'time' in tile_data:
        t_key = 'time'
    else:
        return None

    x_all = np.asarray(tile_data['x'], dtype=np.float64)
    y_all = np.asarray(tile_data['y'], dtype=np.float64)
    h_all = np.asarray(tile_data['h_li'], dtype=np.float64)
    t_all = np.asarray(tile_data[t_key], dtype=np.float64)

    if 'h_rms' in tile_data:
        s_all = np.asarray(tile_data['h_rms'], dtype=np.float64)
        s_all = np.where(np.isnan(s_all) | (s_all <= 0), 0.3, s_all)
    elif 'h_sigma' in tile_data:
        s_all = np.asarray(tile_data['h_sigma'], dtype=np.float64)
        s_all = np.where(np.isnan(s_all) | (s_all <= 0), 0.3, s_all)
    else:
        s_all = np.full_like(h_all, 0.3)

    valid = ~(np.isnan(x_all) | np.isnan(y_all) |
              np.isnan(h_all) | np.isnan(t_all))
    if np.sum(valid) < MIN_POINTS:
        return None

    x = x_all[valid]
    y = y_all[valid]
    h = h_all[valid]
    t = t_all[valid]
    s = s_all[valid]

    del x_all, y_all, h_all, t_all, s_all, tile_data
    gc.collect()

    # Vetor temporal
    if GENERATE_TIMESERIES:
        t_min_global = np.floor(t.min())
        t_max_global = np.ceil(t.max())
        t_bin = np.arange(t_min_global, t_max_global, TS_STEP) + 0.5 * TS_STEP
    else:
        t_bin = None

    # Grade interna
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    gx_min = np.floor(x_min / grid_res) * grid_res + grid_res / 2
    gx_max = np.ceil(x_max / grid_res) * grid_res
    gy_min = np.floor(y_min / grid_res) * grid_res + grid_res / 2
    gy_max = np.ceil(y_max / grid_res) * grid_res

    grid_x = np.arange(gx_min, gx_max, grid_res)
    grid_y = np.arange(gy_min, gy_max, grid_res)

    if len(grid_x) == 0 or len(grid_y) == 0:
        return None

    gx, gy = np.meshgrid(grid_x, grid_y)
    nodes_x = gx.ravel().copy()
    nodes_y = gy.ravel().copy()
    n_nodes = len(nodes_x)

    tree = cKDTree(np.column_stack([x, y]))

    # Pré-alocar
    out_p0 = np.full(n_nodes, np.nan, dtype=np.float64)
    out_p1 = np.full(n_nodes, np.nan, dtype=np.float64)
    out_p2 = np.full(n_nodes, np.nan, dtype=np.float64)
    out_p0_err = np.full(n_nodes, np.nan, dtype=np.float64)
    out_p1_err = np.full(n_nodes, np.nan, dtype=np.float64)
    out_p2_err = np.full(n_nodes, np.nan, dtype=np.float64)
    out_rmse = np.full(n_nodes, np.nan, dtype=np.float64)
    out_nobs = np.zeros(n_nodes, dtype=np.int32)
    out_dmin = np.full(n_nodes, np.nan, dtype=np.float64)
    out_tspan = np.full(n_nodes, np.nan, dtype=np.float64)
    out_tref = np.full(n_nodes, np.nan, dtype=np.float64)

    ts_sec = [] if GENERATE_TIMESERIES else None
    ts_rms = [] if GENERATE_TIMESERIES else None
    ts_idx = [] if GENERATE_TIMESERIES else None

    # Prediction loop
    for ni in range(n_nodes):

        idx, reloc_dist, x_relocated, y_relocated = get_radius_idx(
            x, y, nodes_x[ni], nodes_y[ni], SEARCH_RADIUS, tree, n_rel=N_RELOC
        )

        if len(idx) < MIN_POINTS:
            continue

        idx = np.array(idx)

        t_span = t[idx].max() - t[idx].min()
        if t_span < DT_LIM:
            continue

        result = fit_node(
            x[idx], y[idx], h[idx], t[idx], s[idx],
            x_relocated, y_relocated,
            topo_order=POLY_ORDER,
            temp_order=TEMPORAL_ORDER,
            max_iterations=MAX_ITERATIONS,
            sigma_threshold=SIGMA_THRESHOLD,
            resid_limit=RESID_LIMIT,
            min_points=MIN_POINTS,
            use_weights=USE_WEIGHTS,
            corr_length=CORR_LENGTH,
            t_ref=T_REF,
            t_bin=t_bin
        )

        if result is None:
            continue

        out_p0[ni] = result['p0']
        out_p1[ni] = result['p1']
        out_p2[ni] = result['p2']
        out_p0_err[ni] = result['p0_err']
        out_p1_err[ni] = result['p1_err']
        out_p2_err[ni] = result['p2_err']
        out_rmse[ni] = result['rmse']
        out_nobs[ni] = result['nobs']
        out_dmin[ni] = result['dmin']
        out_tspan[ni] = result['tspan']
        out_tref[ni] = result['t_ref']

        if N_RELOC > 0:
            nodes_x[ni] = x_relocated
            nodes_y[ni] = y_relocated

        if GENERATE_TIMESERIES and 'sec_t' in result:
            ts_sec.append(result['sec_t'])
            ts_rms.append(result['rms_t'])
            ts_idx.append(ni)

    # Verificar
    n_valid = int(np.sum(~np.isnan(out_p1)))
    if n_valid == 0:
        return None

    # Lon/lat
    lon_nodes, lat_nodes = xy_to_lonlat(nodes_x, nodes_y)

    # --- Salvar com função robusta ---

    # Arrays (datasets HDF5)
    output_arrays = {
        'x': nodes_x,
        'y': nodes_y,
        'longitude': lon_nodes,
        'latitude': lat_nodes,
        'p0': out_p0,
        'p1': out_p1,
        'p2': out_p2,
        'p0_error': out_p0_err,
        'p1_error': out_p1_err,
        'p2_error': out_p2_err,
        'rmse': out_rmse,
        'nobs': out_nobs.astype(np.float64),  # HDF5 mais compatível
        'dmin': out_dmin,
        'tspan': out_tspan,
        't_ref': out_tref,
        # Aliases
        'dhdt': out_p1,
        'dhdt_sigma': out_p1_err,
        'n_points_used': out_nobs.astype(np.float64),
        'time_span': out_tspan,
    }

    # Séries temporais
    if GENERATE_TIMESERIES and len(ts_sec) > 0:
        output_arrays['time'] = t_bin
        output_arrays['sec_t'] = np.vstack(ts_sec)
        output_arrays['rms_t'] = np.vstack(ts_rms)
        output_arrays['sec_node_idx'] = np.array(ts_idx, dtype=np.float64)

    # Metadados escalares (attrs HDF5)
    output_meta = {
        'tile_id': tile_file.stem,
        'topo_order': int(POLY_ORDER),
        'temp_order': int(TEMPORAL_ORDER),
        'search_radius': float(SEARCH_RADIUS),
        'n_reloc': int(N_RELOC),
        'use_weights': int(USE_WEIGHTS),
        'rate_limit': float(RATE_LIM),
        'p2_limit': float(P2_LIMIT),
        'resid_limit': float(RESID_LIMIT),
    }

    out_file = output_dir / f"{tile_file.stem}_dhdt.h5"

    with h5py.File(out_file, 'w') as f:
        # Datasets (arrays)
        for key, val in output_arrays.items():
            if val is not None and isinstance(val, np.ndarray):
                f.create_dataset(key, data=val, compression='gzip',
                                 compression_opts=4)

        # Attributes (metadados escalares)
        for key, val in output_meta.items():
            f.attrs[key] = val

    # --- Estatísticas ---
    p1_valid = out_p1[~np.isnan(out_p1)]
    p2_valid = out_p2[~np.isnan(out_p2)]
    rmse_valid = out_rmse[~np.isnan(out_rmse)]

    n_p2_implausible = int(np.sum(~np.isnan(out_p0) & np.isnan(out_p2)))

    stats = {
        'tile': tile_file.stem,
        'n_nodes': n_nodes,
        'n_valid': n_valid,
        'pct_valid': 100 * n_valid / n_nodes,
        'p1_mean': float(np.mean(p1_valid)),
        'p1_median': float(np.median(p1_valid)),
        'p1_std': float(np.std(p1_valid)),
        'p1_min': float(np.min(p1_valid)),
        'p1_max': float(np.max(p1_valid)),
        'p2_mean': float(np.nanmean(p2_valid)) if len(p2_valid) > 0 else np.nan,
        'p2_median': float(np.nanmedian(p2_valid)) if len(p2_valid) > 0 else np.nan,
        'p2_std': float(np.nanstd(p2_valid)) if len(p2_valid) > 0 else np.nan,
        'p2_valid_pct': float(100 * len(p2_valid) / n_valid) if n_valid > 0 else 0,
        'p2_rejected': n_p2_implausible,
        'mean_rmse': float(np.mean(rmse_valid)) if len(rmse_valid) > 0 else np.nan,
        'mean_tspan': float(np.nanmean(out_tspan[out_tspan > 0])) if np.any(out_tspan > 0) else 0,
        'mean_nobs': float(np.mean(out_nobs[out_nobs > 0])) if np.any(out_nobs > 0) else 0,
        'n_timeseries': len(ts_sec) if GENERATE_TIMESERIES else 0,
    }

    return stats


# ============================================
# PROCESSAR TODOS OS TILES
# ============================================

print("\n1. Listando tiles...")

tile_files = sorted(list(TILES_WINTER_DIR.glob("tile_*.h5")))
n_tiles = len(tile_files)
print(f"   Encontrados: {n_tiles} tiles")

if n_tiles == 0:
    print("\n✗ Nenhum tile encontrado!")
    sys.exit(1)

DHDT_WINTER_DIR.mkdir(exist_ok=True, parents=True)

print("\n2. Calculando dh/dt por tile (fitsec v5.1b)...")

n_processed = 0
n_empty = 0
n_error = 0
all_stats = []

for tile_file in tqdm(tile_files, desc="dh/dt"):

    try:
        stats = process_tile(tile_file, DHDT_WINTER_DIR, GRID_RESOLUTION)

        if stats is None:
            n_empty += 1
        else:
            all_stats.append(stats)
            n_processed += 1

        gc.collect()

    except Exception as e:
        print(f"\n✗ Erro em {tile_file.name}: {e}")
        import traceback
        traceback.print_exc()
        n_error += 1
        continue

# ============================================
# SALVAR LOG
# ============================================

if len(all_stats) > 0:
    df_stats = pd.DataFrame(all_stats)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    stats_file = LOGS_DIR / 'dhdt_statistics_winter_v51b.csv'
    df_stats.to_csv(stats_file, index=False)
    print(f"\n   ✓ Log: {stats_file}")

# ============================================
# RESUMO
# ============================================

print("\n" + "*" * 70)
print("RESUMO (v5.1b — fitsec-based, corrigido)")
print("*" * 70)
print(f"Tiles processados: {n_processed}/{n_tiles}")
print(f"Tiles vazios: {n_empty}")
print(f"Tiles com erro: {n_error}")

if len(all_stats) > 0:
    df = pd.DataFrame(all_stats)

    total_nodes = df['n_nodes'].sum()
    valid_nodes = df['n_valid'].sum()
    total_ts = df['n_timeseries'].sum()

    print(f"\nNós: {valid_nodes:,} / {total_nodes:,} ({100*valid_nodes/total_nodes:.1f}%)")

    print(f"\n{'*' * 70}")
    print(f"dh/dt (p1):")
    print(f"  Mean: {df['p1_mean'].mean():+.5f}  "
          f"Std: {df['p1_std'].mean():.2f}  "
          f"Min: {df['p1_min'].min():+.2f}  "
          f"Max: {df['p1_max'].max():+.2f}  "
          f"RMSE: {df['mean_rmse'].mean():.2f}")

    if not df['p2_mean'].isna().all():
        p2_pct = df['p2_valid_pct'].mean()
        p2_rej = df['p2_rejected'].sum()
        print(f"\nd²h/dt² (p2):")
        print(f"  Mean: {df['p2_mean'].mean():+.5f} m/ano²  "
              f"Median: {df['p2_median'].median():+.5f}  "
              f"Std: {df['p2_std'].mean():.5f}")
        print(f"  Válidos: {p2_pct:.1f}% (|p2| < {P2_LIMIT})  "
              f"Rejeitados: {p2_rej:,}")

    print(f"{'*' * 70}")

    print(f"\nTime span médio: {df['mean_tspan'].mean():.2f} anos")
    print(f"Obs médias por nó: {df['mean_nobs'].mean():.0f}")

    if GENERATE_TIMESERIES:
        print(f"Séries temporais: {total_ts:,}")

    print(f"\nLimites:")
    print(f"  |dh/dt| < {RATE_LIM} m/ano")
    print(f"  |d²h/dt²| < {P2_LIMIT} m/ano²")
    print(f"  |resíduo| < {RESID_LIMIT} m")

print(f"\nResultados em: {DHDT_WINTER_DIR}")
print("*" * 70)
print("\n✓ Concluído! Próximo: 11_join_tiles.py")