"""
Análise estatística completa dos resultados de dh/dt

Paleta de cores consistente:
  VERMELHO = afinamento / derretimento (dh/dt < 0)
  CINZA CLARO = estável / próximo de zero
  AZUL     = espessamento (dh/dt > 0)

Analisa DOIS conjuntos:
  1. Dados pontuais (Script 11 — join)
  2. Grade interpolada (Script 12 — creategrid)
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import sys
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

print("=" * 70)
print("ANÁLISE ESTATÍSTICA COMPLETA DE dh/dt")
print("Compatível com: Script 10 v5.1b / Script 11 v2 / Script 12 v3")
print("Paleta: VERMELHO = afinamento | CINZA = estável | AZUL = espessamento")
print("=" * 70)

# ============================================
# RESOLVER DIRETÓRIOS
# ============================================

BASE_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")

try:
    _results_dir = RESULTS_DIR
except NameError:
    _results_dir = BASE_DIR / 'results'
RESULTS_DIR = _results_dir

try:
    _figures_dir = FIGURES_DIR
except NameError:
    _figures_dir = BASE_DIR / 'figures'
FIGURES_DIR = _figures_dir

try:
    region_name = THWAITES_BBOX['name']
except (NameError, KeyError):
    region_name = 'Amundsen Sea Embayment'

try:
    _grid_res = GRID_RESOLUTION
except NameError:
    _grid_res = 2000
GRID_RES = _grid_res

stats_dir = RESULTS_DIR / 'statistics'
stats_dir.mkdir(exist_ok=True, parents=True)
plots_dir = FIGURES_DIR / 'plots'
plots_dir.mkdir(exist_ok=True, parents=True)
maps_dir = FIGURES_DIR / 'maps'
maps_dir.mkdir(exist_ok=True, parents=True)

# ============================================
# PALETA CUSTOMIZADA: Vermelho → Cinza claro → Azul
# ============================================

def make_rdgbu_cmap(name='RdGrayBu'):
    """
    Vermelho escuro → Vermelho → Cinza claro → Azul → Azul escuro
    Centro (zero) = cinza claro (#D9D9D9).
    """
    colors = [
        (0.00, '#67001F'),
        (0.15, '#B2182B'),
        (0.30, '#D6604D'),
        (0.42, '#F4A582'),
        (0.50, '#D9D9D9'),
        (0.58, '#92C5DE'),
        (0.70, '#4393C3'),
        (0.85, '#2166AC'),
        (1.00, '#053061'),
    ]
    positions = [c[0] for c in colors]
    rgb_colors = [mcolors.hex2color(c[1]) for c in colors]
    return LinearSegmentedColormap.from_list(name, list(zip(positions, rgb_colors)), N=256)


cmap_dhdt = make_rdgbu_cmap()

# ============================================
# FUNÇÃO: LER HDF5 ROBUSTO
# ============================================

def read_h5_robust(filepath):
    """Lê datasets e attrs de um HDF5 (compatível v5.1b)."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            try:
                if isinstance(f[key], h5py.Group):
                    continue
                data[key] = f[key][:]
            except:
                pass
        for key in f.attrs:
            data['_attr_' + key] = f.attrs[key]
    return data


def get_var(data, *keys, default=None):
    """Busca variável por múltiplos nomes possíveis."""
    for key in keys:
        if key in data and isinstance(data[key], np.ndarray):
            return np.asarray(data[key], dtype=np.float64)
    if default is not None:
        return default
    return None

# ============================================
# CARREGAR DADOS PONTUAIS (Script 11 — join)
# ============================================

print("\n1. Carregando dados pontuais (join)...")

dhdt_dir = RESULTS_DIR / 'dhdt_winter'
joined_file = None

if dhdt_dir.exists():
    for name in ['amundsen_sea_dhdt_winter_joined.h5',
                  'amundsen_sea_dhdt_winter_gridded.h5',
                  'thwaites_dhdt_winter_ice_only.h5']:
        candidate = dhdt_dir / name
        if candidate.exists():
            joined_file = candidate
            break
    if joined_file is None:
        h5_files = [f for f in dhdt_dir.glob("*.h5") if 'tile_' not in f.name]
        if h5_files:
            joined_file = sorted(h5_files, key=lambda f: f.stat().st_mtime,
                                 reverse=True)[0]

if joined_file is None:
    print("✗ Arquivo joined não encontrado!")
    sys.exit(1)

print(f"   Arquivo: {joined_file.name}")
pt_data = read_h5_robust(joined_file)

# Variáveis principais
pt_dhdt = get_var(pt_data, 'p1', 'dhdt')
pt_p1_err = get_var(pt_data, 'p1_error', 'dhdt_sigma')
pt_p2 = get_var(pt_data, 'p2')
pt_p2_err = get_var(pt_data, 'p2_error')
pt_p0 = get_var(pt_data, 'p0')
pt_rmse = get_var(pt_data, 'rmse')
pt_nobs = get_var(pt_data, 'nobs', 'n_points_used')
pt_dmin = get_var(pt_data, 'dmin')
pt_tspan = get_var(pt_data, 'tspan', 'time_span')
pt_x = get_var(pt_data, 'x')
pt_y = get_var(pt_data, 'y')
pt_lat = get_var(pt_data, 'latitude')
pt_lon = get_var(pt_data, 'longitude')

if pt_dhdt is None:
    print("✗ Variável dhdt/p1 não encontrada!")
    sys.exit(1)

has_p2 = pt_p2 is not None
has_p2_err = pt_p2_err is not None
has_p1_err = pt_p1_err is not None
has_rmse_fit = pt_rmse is not None
has_dmin = pt_dmin is not None
has_p0 = pt_p0 is not None

valid = ~np.isnan(pt_dhdt)
pt_dhdt = pt_dhdt[valid]
if has_p1_err:
    pt_p1_err = pt_p1_err[valid]
if has_p2:
    pt_p2 = pt_p2[valid]
if has_p2_err:
    pt_p2_err = pt_p2_err[valid]
if has_rmse_fit:
    pt_rmse = pt_rmse[valid]
if pt_nobs is not None:
    pt_nobs = pt_nobs[valid]
if has_dmin:
    pt_dmin = pt_dmin[valid]
if pt_tspan is not None:
    pt_tspan = pt_tspan[valid]
if pt_lat is not None:
    pt_lat = pt_lat[valid]
if pt_lon is not None:
    pt_lon = pt_lon[valid]
if pt_x is not None:
    pt_x = pt_x[valid]
if pt_y is not None:
    pt_y = pt_y[valid]
if has_p0:
    pt_p0 = pt_p0[valid]

print(f"   Nós válidos: {len(pt_dhdt):,}")
print(f"   Variáveis disponíveis: p1✓ "
      f"p1_error{'✓' if has_p1_err else '✗'} "
      f"p2{'✓' if has_p2 else '✗'} "
      f"p2_error{'✓' if has_p2_err else '✗'} "
      f"rmse{'✓' if has_rmse_fit else '✗'} "
      f"dmin{'✓' if has_dmin else '✗'} "
      f"p0{'✓' if has_p0 else '✗'}")

del pt_data

# ============================================
# CARREGAR GRADE INTERPOLADA (Script 12)
# ============================================

print("\n2. Carregando grade interpolada...")

grid_dir = RESULTS_DIR / 'grids'
grid_file = None

if grid_dir.exists():
    for name in ['amundsen_sea_dhdt_winter_grid.h5',
                  'thwaites_dhdt_winter_grid.h5']:
        candidate = grid_dir / name
        if candidate.exists():
            grid_file = candidate
            break
    if grid_file is None:
        h5_files = list(grid_dir.glob("*grid*.h5"))
        if h5_files:
            grid_file = sorted(h5_files, key=lambda f: f.stat().st_mtime,
                               reverse=True)[0]

has_grid = False
has_grid_p2 = False
has_grid_rmse_fit = False

if grid_file:
    print(f"   Arquivo: {grid_file.name}")
    gd = read_h5_robust(grid_file)

    gr_dhdt = get_var(gd, 'Z_pred', 'dhdt_smooth', 'dhdt', 'dhdt_filled')
    gr_rmse_interp = get_var(gd, 'Z_rmse', 'dhdt_sigma', 'dhdt_std')
    gr_nobs = get_var(gd, 'Z_nobs', 'n_obs_per_cell', 'n_points')
    gr_p2 = get_var(gd, 'Z_accel')
    gr_rmse_fit = get_var(gd, 'Z_rmse_fit')
    gr_lon = get_var(gd, 'lon')
    gr_lat = get_var(gd, 'lat')
    gr_x = get_var(gd, 'X', 'x')
    gr_y = get_var(gd, 'Y', 'y')

    if gr_dhdt is not None:
        has_grid = True
        has_grid_p2 = gr_p2 is not None
        has_grid_rmse_fit = gr_rmse_fit is not None
        print(f"   Shape: {gr_dhdt.shape}")
        print(f"   p2 (aceleração): {'✓' if has_grid_p2 else '✗'}")
        print(f"   RMSE fit: {'✓' if has_grid_rmse_fit else '✗'}")

    del gd
else:
    print("   ⚠ Grade não encontrada")

# ============================================
# ESTATÍSTICAS DESCRITIVAS
# ============================================

def compute_stats(data, label=""):
    """Estatísticas descritivas completas."""
    d = data[~np.isnan(data)]
    if len(d) == 0:
        return {}
    return {
        'n': len(d),
        'mean': float(np.mean(d)),
        'median': float(np.median(d)),
        'std': float(np.std(d)),
        'min': float(np.min(d)),
        'max': float(np.max(d)),
        'p1': float(np.percentile(d, 1)),
        'p5': float(np.percentile(d, 5)),
        'p10': float(np.percentile(d, 10)),
        'p25': float(np.percentile(d, 25)),
        'p75': float(np.percentile(d, 75)),
        'p90': float(np.percentile(d, 90)),
        'p95': float(np.percentile(d, 95)),
        'p99': float(np.percentile(d, 99)),
        'iqr': float(np.percentile(d, 75) - np.percentile(d, 25)),
        'mad': float(np.median(np.abs(d - np.median(d)))),
        'skewness': float(sp_stats.skew(d)),
        'kurtosis': float(sp_stats.kurtosis(d)),
    }


print(f"\n{'=' * 70}")
print("3. ESTATÍSTICAS DOS DADOS PONTUAIS (Script 11)")
print("=" * 70)

pt_stats = compute_stats(pt_dhdt)

print(f"\n   dh/dt (p1):")
print(f"      Média:   {pt_stats['mean']:+.4f} m/ano")
print(f"      Mediana: {pt_stats['median']:+.4f} m/ano")
print(f"      Std:     {pt_stats['std']:.4f} m/ano")
print(f"      MAD:     {pt_stats['mad']:.4f} m/ano")
print(f"      Min:     {pt_stats['min']:+.4f} m/ano")
print(f"      Max:     {pt_stats['max']:+.4f} m/ano")

print(f"\n   Percentis:")
for p in ['p1', 'p5', 'p10', 'p25', 'p75', 'p90', 'p95', 'p99']:
    print(f"      {p:>3s}: {pt_stats[p]:+.4f} m/ano")

print(f"\n   Distribuição:")
print(f"      Skewness: {pt_stats['skewness']:.4f}")
print(f"      Kurtosis: {pt_stats['kurtosis']:.4f}")

if has_p2:
    p2_valid = pt_p2[~np.isnan(pt_p2)]
    p2_stats = compute_stats(p2_valid)
    n_p2 = len(p2_valid)

    print(f"\n   d²h/dt² (p2 — aceleração):")
    print(f"      Válidos: {n_p2:,} / {len(pt_dhdt):,} ({100*n_p2/len(pt_dhdt):.1f}%)")
    print(f"      Média:   {p2_stats['mean']:+.5f} m/ano²")
    print(f"      Mediana: {p2_stats['median']:+.5f} m/ano²")
    print(f"      Std:     {p2_stats['std']:.5f} m/ano²")
    print(f"      P5:      {p2_stats['p5']:+.5f} m/ano²")
    print(f"      P95:     {p2_stats['p95']:+.5f} m/ano²")

if has_p1_err:
    err_valid = pt_p1_err[(pt_p1_err > 0) & ~np.isnan(pt_p1_err)]
    print(f"\n   Erro formal (p1_error):")
    print(f"      Média:   {np.mean(err_valid):.4f} m/ano")
    print(f"      Mediana: {np.median(err_valid):.4f} m/ano")
    print(f"      P95:     {np.percentile(err_valid, 95):.4f} m/ano")

if has_rmse_fit:
    rmse_valid = pt_rmse[(pt_rmse > 0) & ~np.isnan(pt_rmse)]
    print(f"\n   RMSE do fit (resíduos do fitsec):")
    print(f"      Média:   {np.mean(rmse_valid):.4f} m")
    print(f"      Mediana: {np.median(rmse_valid):.4f} m")
    print(f"      P95:     {np.percentile(rmse_valid, 95):.4f} m")

if has_dmin:
    dmin_valid = pt_dmin[(pt_dmin > 0) & ~np.isnan(pt_dmin)]
    if len(dmin_valid) > 0:
        print(f"\n   Distância ao ponto mais próximo (dmin):")
        print(f"      Média:   {np.mean(dmin_valid):.0f} m")
        print(f"      Mediana: {np.median(dmin_valid):.0f} m")
        print(f"      Max:     {np.max(dmin_valid):.0f} m")

# ============================================
# ESTATÍSTICAS — GRADE INTERPOLADA
# ============================================

if has_grid:
    print(f"\n{'=' * 70}")
    print("4. ESTATÍSTICAS DA GRADE INTERPOLADA (Script 12)")
    print("=" * 70)

    gr_valid = gr_dhdt[~np.isnan(gr_dhdt)].ravel()
    gr_stats = compute_stats(gr_valid)

    print(f"\n   dh/dt (grade):")
    print(f"      Média:   {gr_stats['mean']:+.4f} m/ano")
    print(f"      Mediana: {gr_stats['median']:+.4f} m/ano")
    print(f"      Std:     {gr_stats['std']:.4f} m/ano")
    print(f"      Min:     {gr_stats['min']:+.4f} m/ano")
    print(f"      Max:     {gr_stats['max']:+.4f} m/ano")

    if gr_rmse_interp is not None:
        rmse_interp_valid = gr_rmse_interp[~np.isnan(gr_rmse_interp) & ~np.isnan(gr_dhdt)].ravel()
        if len(rmse_interp_valid) > 0:
            print(f"\n   Erro de predição (interpgaus RMSE):")
            print(f"      Média:   {np.mean(rmse_interp_valid):.4f} m/ano")
            print(f"      Mediana: {np.median(rmse_interp_valid):.4f} m/ano")
            print(f"      P95:     {np.percentile(rmse_interp_valid, 95):.4f} m/ano")
    else:
        rmse_interp_valid = np.array([])

    if has_grid_p2:
        p2_grid_valid = gr_p2[~np.isnan(gr_p2)].ravel()
        if len(p2_grid_valid) > 0:
            print(f"\n   d²h/dt² (aceleração, grade):")
            print(f"      Média:   {np.mean(p2_grid_valid):+.5f} m/ano²")
            print(f"      Mediana: {np.median(p2_grid_valid):+.5f} m/ano²")
            print(f"      Std:     {np.std(p2_grid_valid):.5f} m/ano²")

    if has_grid_rmse_fit:
        rmse_fit_grid = gr_rmse_fit[~np.isnan(gr_rmse_fit)].ravel()
        if len(rmse_fit_grid) > 0:
            print(f"\n   RMSE do fit (interpolado):")
            print(f"      Média:   {np.mean(rmse_fit_grid):.4f} m")
            print(f"      Mediana: {np.median(rmse_fit_grid):.4f} m")

# ============================================
# CLASSIFICAÇÃO
# ============================================

print(f"\n{'=' * 70}")
print("5. CLASSIFICAÇÃO DE REGIMES")
print("=" * 70)

categories = [
    ('Adelgaçamento rápido', -np.inf, -1.5),
    ('Adelgaçamento moderado', -1.5, -0.5),
    ('Adelgaçamento lento', -0.5, -0.1),
    ('Estável', -0.1, 0.1),
    ('Espessamento lento', 0.1, 0.5),
    ('Espessamento moderado', 0.5, 1.5),
    ('Espessamento rápido', 1.5, np.inf),
]

class_rows = []
for label, lo, hi in categories:
    n_pt = int(np.sum((pt_dhdt >= lo) & (pt_dhdt < hi)))
    pct_pt = 100 * n_pt / len(pt_dhdt)
    if has_grid:
        n_gr = int(np.sum((gr_valid >= lo) & (gr_valid < hi)))
        pct_gr = 100 * n_gr / len(gr_valid)
    else:
        n_gr, pct_gr = 0, 0

    print(f"   {label:30s}: {n_pt:>7,} pts ({pct_pt:5.1f}%) | "
          f"{n_gr:>7,} grid ({pct_gr:5.1f}%)")
    class_rows.append({
        'category': label, 'lo': lo, 'hi': hi,
        'n_points': n_pt, 'pct_points': pct_pt,
        'n_grid': n_gr, 'pct_grid': pct_gr
    })

# ============================================
# ANÁLISE DE INCERTEZA
# ============================================

print(f"\n{'=' * 70}")
print("6. ANÁLISE DE INCERTEZA")
print("=" * 70)

if has_p1_err:
    sigma_valid = pt_p1_err[(pt_p1_err > 0) & ~np.isnan(pt_p1_err)]
    print(f"\n   Erro formal (p1_error — covariância do LSQ):")
    print(f"      Média:   {np.mean(sigma_valid):.4f} m/ano")
    print(f"      Mediana: {np.median(sigma_valid):.4f} m/ano")
    print(f"      P5:      {np.percentile(sigma_valid, 5):.4f} m/ano")
    print(f"      P95:     {np.percentile(sigma_valid, 95):.4f} m/ano")

    n_high = int(np.sum(sigma_valid < 0.05))
    n_med = int(np.sum((sigma_valid >= 0.05) & (sigma_valid < 0.2)))
    n_low = int(np.sum(sigma_valid >= 0.2))
    n_tot = len(sigma_valid)
    print(f"\n   Qualidade (p1_error):")
    print(f"      Alta  (σ < 0.05):       {n_high:,} ({100*n_high/n_tot:.1f}%)")
    print(f"      Média (0.05 < σ < 0.2): {n_med:,} ({100*n_med/n_tot:.1f}%)")
    print(f"      Baixa (σ > 0.2):        {n_low:,} ({100*n_low/n_tot:.1f}%)")

if has_rmse_fit:
    print(f"\n   RMSE do fit vs erro formal:")
    both = ((~np.isnan(pt_rmse)) & (pt_rmse > 0) &
            (~np.isnan(pt_p1_err)) & (pt_p1_err > 0)) if has_p1_err else np.zeros(len(pt_dhdt), dtype=bool)
    if np.sum(both) > 100:
        corr = np.corrcoef(pt_rmse[both], pt_p1_err[both])[0, 1]
        print(f"      Correlação rmse vs p1_error: {corr:.4f}")
        print(f"      Razão média rmse/p1_error:   {np.mean(pt_rmse[both])/np.mean(pt_p1_err[both]):.1f}×")

# ============================================
# COBERTURA
# ============================================

print(f"\n{'=' * 70}")
print("7. ANÁLISE DE COBERTURA")
print("=" * 70)

if pt_nobs is not None:
    npts_valid = pt_nobs[pt_nobs > 0]
    if len(npts_valid) > 0:
        print(f"\n   Observações por nó (nobs):")
        print(f"      Média:   {np.mean(npts_valid):.0f}")
        print(f"      Mediana: {np.median(npts_valid):.0f}")
        print(f"      Min:     {np.min(npts_valid):.0f}")
        print(f"      Max:     {np.max(npts_valid):.0f}")

if pt_tspan is not None:
    tspan_valid = pt_tspan[pt_tspan > 0]
    if len(tspan_valid) > 0:
        print(f"\n   Time span:")
        print(f"      Média:   {np.mean(tspan_valid):.2f} anos")
        print(f"      Mediana: {np.median(tspan_valid):.2f} anos")
        print(f"      Min:     {np.min(tspan_valid):.2f} anos")
        print(f"      Max:     {np.max(tspan_valid):.2f} anos")

cell_area_km2 = (GRID_RES / 1000) ** 2
area_pts = len(pt_dhdt) * cell_area_km2
print(f"\n   Área:")
print(f"      Nós: {area_pts:,.0f} km²")
if has_grid:
    area_grid = len(gr_valid) * cell_area_km2
    print(f"      Grade: {area_grid:,.0f} km²")

# ============================================
# SALVAR CSVs
# ============================================

print(f"\n{'=' * 70}")
print("8. SALVANDO CSVs")
print("=" * 70)

rows = [{'source': 'points', 'variable': 'dhdt', **pt_stats}]
if has_grid:
    rows.append({'source': 'grid', 'variable': 'dhdt', **gr_stats})
if has_p2:
    rows.append({'source': 'points', 'variable': 'p2', **compute_stats(p2_valid)})
if has_grid_p2:
    rows.append({'source': 'grid', 'variable': 'p2', **compute_stats(p2_grid_valid)})

df_stats = pd.DataFrame(rows)
f1 = stats_dir / 'dhdt_statistics_summary.csv'
df_stats.to_csv(f1, index=False)
print(f"   ✓ {f1.name}")

df_class = pd.DataFrame(class_rows)
f2 = stats_dir / 'dhdt_classification.csv'
df_class.to_csv(f2, index=False)
print(f"   ✓ {f2.name}")

bins = np.arange(-8, 6, 0.1)
hist, bin_edges = np.histogram(pt_dhdt, bins=bins)
df_hist = pd.DataFrame({
    'bin_center': (bin_edges[:-1] + bin_edges[1:]) / 2,
    'bin_min': bin_edges[:-1], 'bin_max': bin_edges[1:],
    'count': hist, 'frequency': hist / len(pt_dhdt)
})
f3 = stats_dir / 'dhdt_distribution_points.csv'
df_hist.to_csv(f3, index=False)
print(f"   ✓ {f3.name}")

if has_grid:
    hist_gr, _ = np.histogram(gr_valid, bins=bins)
    df_hist_gr = pd.DataFrame({
        'bin_center': (bin_edges[:-1] + bin_edges[1:]) / 2,
        'count': hist_gr, 'frequency': hist_gr / len(gr_valid)
    })
    f4 = stats_dir / 'dhdt_distribution_grid.csv'
    df_hist_gr.to_csv(f4, index=False)
    print(f"   ✓ {f4.name}")

# ============================================
# GRÁFICOS
# ============================================

print(f"\n{'=' * 70}")
print("9. GERANDO GRÁFICOS")
print("=" * 70)

# --- PLOT 1: HISTOGRAMA COMPARATIVO ---

print("\n   Plot 1: Histograma comparativo...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bins_plot = np.arange(-5, 4, 0.1)

axes[0].hist(pt_dhdt, bins=bins_plot, alpha=0.6, color='steelblue',
             edgecolor='none', density=True, label='Nós (join)')
if has_grid:
    axes[0].hist(gr_valid, bins=bins_plot, alpha=0.5, color='coral',
                 edgecolor='none', density=True, label='Grade (interpgaus)')
axes[0].axvline(0, color='black', linewidth=1.5)
axes[0].axvline(np.mean(pt_dhdt), color='blue', linestyle='--', linewidth=1.5,
                label=f'Média pts = {np.mean(pt_dhdt):+.3f}')
if has_grid:
    axes[0].axvline(np.mean(gr_valid), color='red', linestyle='--', linewidth=1.5,
                    label=f'Média grid = {np.mean(gr_valid):+.3f}')
axes[0].set_xlabel('dh/dt (m/ano)', fontsize=12)
axes[0].set_ylabel('Densidade', fontsize=12)
axes[0].set_title('(a) Linear', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-5, 3)

axes[1].hist(pt_dhdt, bins=bins_plot, alpha=0.6, color='steelblue',
             edgecolor='none', density=True, label='Nós')
if has_grid:
    axes[1].hist(gr_valid, bins=bins_plot, alpha=0.5, color='coral',
                 edgecolor='none', density=True, label='Grade')
axes[1].axvline(0, color='black', linewidth=1.5)
axes[1].set_yscale('log')
axes[1].set_xlabel('dh/dt (m/ano)', fontsize=12)
axes[1].set_ylabel('Densidade (log)', fontsize=12)
axes[1].set_title('(b) Logarítmica', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-5, 3)

fig.suptitle(f'{region_name} — Distribuição de dh/dt', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(plots_dir / 'histogram_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ histogram_comparison.png")
plt.close()

# --- PLOT 2: CDF ---

print("   Plot 2: CDF...")

fig, ax = plt.subplots(figsize=(10, 6))
sorted_pt = np.sort(pt_dhdt)
cdf_pt = np.arange(1, len(sorted_pt) + 1) / len(sorted_pt)
ax.plot(sorted_pt, cdf_pt * 100, 'b-', linewidth=2, label='Nós')
if has_grid:
    sorted_gr = np.sort(gr_valid)
    cdf_gr = np.arange(1, len(sorted_gr) + 1) / len(sorted_gr)
    ax.plot(sorted_gr, cdf_gr * 100, 'r-', linewidth=2, label='Grade')

ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(50, color='gray', linestyle=':', linewidth=1)
for p in [5, 25, 50, 75, 95]:
    val = np.percentile(pt_dhdt, p)
    ax.plot(val, p, 'bo', markersize=6)
    ax.annotate(f'P{p}\n{val:+.2f}', xy=(val, p), fontsize=8, ha='center',
                va='bottom', bbox=dict(boxstyle='round,pad=0.2',
                facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('dh/dt (m/ano)', fontsize=12)
ax.set_ylabel('Percentil (%)', fontsize=12)
ax.set_title(f'{region_name} — CDF de dh/dt', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 3)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(plots_dir / 'cdf_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ cdf_comparison.png")
plt.close()

# --- PLOT 3: BOX PLOT ---

print("   Plot 3: Box plot...")

fig, ax = plt.subplots(figsize=(8, 8))
box_data = [pt_dhdt]
box_labels = ['Nós\n(join)']
box_colors = ['lightblue']
if has_grid:
    box_data.append(gr_valid)
    box_labels.append('Grade\n(interpgaus)')
    box_colors.append('lightsalmon')

bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.5,
                labels=box_labels, showfliers=False)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
for median in bp['medians']:
    median.set_color('red')
    median.set_linewidth(2)

ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
for i, data in enumerate(box_data):
    for p, va in [(5, 'top'), (95, 'bottom')]:
        val = np.percentile(data, p)
        ax.annotate(f'P{p}: {val:+.2f}', xy=(i + 1.3, val), fontsize=9, color='gray')

ax.set_ylabel('dh/dt (m/ano)', fontsize=12)
ax.set_title(f'{region_name} — Box Plot', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(-5, 3)
plt.tight_layout()
plt.savefig(plots_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ boxplot_comparison.png")
plt.close()

# --- PLOT 4: VIOLIN POR LATITUDE ---

if has_grid and gr_lat is not None:
    print("   Plot 4: Violin por latitude...")

    fig, ax = plt.subplots(figsize=(14, 7))
    lat_flat = gr_lat[~np.isnan(gr_dhdt)].ravel()
    dhdt_flat = gr_dhdt[~np.isnan(gr_dhdt)].ravel()

    lat_bins = np.arange(np.floor(np.nanmin(lat_flat)),
                         np.ceil(np.nanmax(lat_flat)) + 1, 2)
    violin_data = []
    violin_pos = []
    for i in range(len(lat_bins) - 1):
        lo, hi = lat_bins[i], lat_bins[i + 1]
        mask = (lat_flat >= lo) & (lat_flat < hi)
        if np.sum(mask) > 50:
            vals = dhdt_flat[mask]
            vals = vals[(vals > -5) & (vals < 3)]
            if len(vals) > 20:
                violin_data.append(vals)
                violin_pos.append((lo + hi) / 2)

    if len(violin_data) > 0:
        vp = ax.violinplot(violin_data, positions=violin_pos,
                           showmeans=True, showmedians=True, widths=1.5)
        for body in vp['bodies']:
            body.set_facecolor('steelblue')
            body.set_alpha(0.6)
        vp['cmeans'].set_color('red')
        vp['cmeans'].set_linewidth(2)
        vp['cmedians'].set_color('orange')
        vp['cmedians'].set_linewidth(2)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Latitude (°S)', fontsize=12)
        ax.set_ylabel('dh/dt (m/ano)', fontsize=12)
        ax.set_title(f'{region_name} — dh/dt por Latitude',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-5, 3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'violin_by_latitude.png', dpi=300, bbox_inches='tight')
    print("   ✓ violin_by_latitude.png")
    plt.close()

# --- PLOT 5: SCATTER dh/dt vs LATITUDE ---

if has_grid and gr_lat is not None:
    print("   Plot 5: Scatter dh/dt vs latitude...")

    fig, ax = plt.subplots(figsize=(12, 7))
    n_sample = min(50000, len(dhdt_flat))
    idx = np.random.choice(len(dhdt_flat), n_sample, replace=False)

    norm_scatter = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    scatter = ax.scatter(lat_flat[idx], dhdt_flat[idx], c=dhdt_flat[idx],
                         cmap=cmap_dhdt, norm=norm_scatter, s=1, alpha=0.3,
                         rasterized=True)
    plt.colorbar(scatter, ax=ax, label='dh/dt (m/ano)')

    for i in range(len(lat_bins) - 1):
        lo, hi = lat_bins[i], lat_bins[i + 1]
        mask = (lat_flat >= lo) & (lat_flat < hi)
        if np.sum(mask) > 10:
            mean_val = np.mean(dhdt_flat[mask])
            ax.plot((lo + hi) / 2, mean_val, 'ko', markersize=8)
            ax.plot((lo + hi) / 2, mean_val, 'wo', markersize=5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Latitude (°S)', fontsize=12)
    ax.set_ylabel('dh/dt (m/ano)', fontsize=12)
    ax.set_title(f'{region_name} — dh/dt vs Latitude\n'
                 f'Vermelho = afinamento | Cinza = estável | Azul = espessamento',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'scatter_dhdt_vs_latitude.png', dpi=300, bbox_inches='tight')
    print("   ✓ scatter_dhdt_vs_latitude.png")
    plt.close()

# --- PLOT 6: ÁREA E VOLUME POR FAIXA ---

if has_grid:
    print("   Plot 6: Área e volume por faixa...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Cores consistentes: vermelho (afinamento) → cinza → azul (espessamento)
    faixas = [
        ('< -2.0',         -np.inf, -2.0,  '#67001F'),   # vermelho muito escuro
        ('-2.0 a -1.0',    -2.0,    -1.0,  '#B2182B'),   # vermelho escuro
        ('-1.0 a -0.5',    -1.0,    -0.5,  '#D6604D'),   # vermelho médio
        ('-0.5 a -0.1',    -0.5,    -0.1,  '#F4A582'),   # salmão claro
        ('-0.1 a +0.1',    -0.1,     0.1,  '#D9D9D9'),   # CINZA CLARO (estável)
        ('+0.1 a +0.5',     0.1,     0.5,  '#92C5DE'),   # azul claro
        ('> +0.5',           0.5, np.inf,   '#2166AC'),   # azul escuro
    ]

    areas = []
    volumes = []
    labels = []
    colors = []

    for label, lo, hi, color in faixas:
        mask = (gr_valid >= lo) & (gr_valid < hi)
        n_cells = int(np.sum(mask))
        area = n_cells * cell_area_km2
        vol = np.mean(gr_valid[mask]) * area / 1000 if n_cells > 0 else 0
        areas.append(area)
        volumes.append(vol)
        labels.append(label)
        colors.append(color)

    bars1 = ax1.barh(labels, areas, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Área (km²)', fontsize=11)
    ax1.set_title('(a) Área por Faixa de dh/dt', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars1, areas):
        if val > 0:
            ax1.text(bar.get_width() + max(areas) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:,.0f}', va='center', fontsize=9)

    bars2 = ax2.barh(labels, volumes, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=1.5)
    ax2.set_xlabel('dV/dt (km³/ano)', fontsize=11)
    ax2.set_title('(b) Volume por Faixa de dh/dt', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    total_vol = sum(volumes)
    ax2.text(0.98, 0.02, f'Total: {total_vol:+.1f} km³/ano',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle(f'{region_name} — Área e Volume por dh/dt\n'
                 f'Vermelho = afinamento | Cinza = estável | Azul = espessamento',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(plots_dir / 'area_volume_by_rate.png', dpi=300, bbox_inches='tight')
    print("   ✓ area_volume_by_rate.png")
    plt.close()

# --- PLOT 7: ANÁLISE DE ERRO ---

print("   Plot 7: Análise de erro...")

fig, axes = plt.subplots(1, 3 if has_rmse_fit else 2, figsize=(18 if has_rmse_fit else 14, 5))

if has_p1_err:
    ax = axes[0]
    ax.hist(sigma_valid, bins=100, edgecolor='none', color='steelblue', alpha=0.8)
    ax.axvline(np.mean(sigma_valid), color='red', linestyle='--',
               label=f'Média = {np.mean(sigma_valid):.4f}')
    ax.axvline(np.median(sigma_valid), color='orange', linestyle='--',
               label=f'Mediana = {np.median(sigma_valid):.4f}')
    ax.set_xlabel('p1_error (m/ano)', fontsize=11)
    ax.set_ylabel('Frequência', fontsize=11)
    ax.set_title('(a) Erro Formal (p1_error)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

if has_grid and len(rmse_interp_valid) > 0:
    ax = axes[1]
    ax.hist(rmse_interp_valid, bins=100, edgecolor='none', color='coral', alpha=0.8)
    ax.axvline(np.mean(rmse_interp_valid), color='red', linestyle='--',
               label=f'Média = {np.mean(rmse_interp_valid):.4f}')
    ax.axvline(np.median(rmse_interp_valid), color='orange', linestyle='--',
               label=f'Mediana = {np.median(rmse_interp_valid):.4f}')
    ax.set_xlabel('RMSE interpgaus (m/ano)', fontsize=11)
    ax.set_ylabel('Frequência', fontsize=11)
    ax.set_title('(b) RMSE da Interpolação', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

if has_rmse_fit:
    ax = axes[2]
    ax.hist(rmse_valid, bins=100, edgecolor='none', color='mediumpurple', alpha=0.8)
    ax.axvline(np.mean(rmse_valid), color='red', linestyle='--',
               label=f'Média = {np.mean(rmse_valid):.4f}')
    ax.axvline(np.median(rmse_valid), color='orange', linestyle='--',
               label=f'Mediana = {np.median(rmse_valid):.4f}')
    ax.set_xlabel('RMSE fit (m)', fontsize=11)
    ax.set_ylabel('Frequência', fontsize=11)
    ax.set_title('(c) RMSE do Fit (fitsec)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'{region_name} — Análise de Incerteza',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(plots_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ error_analysis.png")
plt.close()

# --- PLOT 8: ACELERAÇÃO ---

if has_p2:
    print("   Plot 8: Análise de aceleração (p2)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    p2_plot = p2_valid[(p2_valid > -1) & (p2_valid < 1)]
    axes[0].hist(p2_plot, bins=100, edgecolor='none', color='seagreen', alpha=0.8,
                 density=True)
    axes[0].axvline(0, color='black', linewidth=1.5)
    axes[0].axvline(np.mean(p2_valid), color='red', linestyle='--',
                    label=f'Média = {np.mean(p2_valid):+.5f}')
    axes[0].axvline(np.median(p2_valid), color='orange', linestyle='--',
                    label=f'Mediana = {np.median(p2_valid):+.5f}')
    axes[0].set_xlabel('d²h/dt² (m/ano²)', fontsize=12)
    axes[0].set_ylabel('Densidade', fontsize=12)
    axes[0].set_title('(a) Distribuição da Aceleração', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    both = ~np.isnan(pt_p2)
    if np.sum(both) > 100:
        n_s = min(30000, int(np.sum(both)))
        idx_s = np.random.choice(np.where(both)[0], n_s, replace=False)

        norm_p2_sc = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
        sc = axes[1].scatter(pt_dhdt[idx_s], pt_p2[idx_s],
                             c=pt_p2[idx_s], cmap=cmap_dhdt, norm=norm_p2_sc,
                             s=2, alpha=0.3, rasterized=True)
        plt.colorbar(sc, ax=axes[1], label='d²h/dt² (m/ano²)')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('dh/dt (m/ano)', fontsize=12)
        axes[1].set_ylabel('d²h/dt² (m/ano²)', fontsize=12)
        axes[1].set_title('(b) dh/dt vs Aceleração\n'
                          'Vermelho = acelerando perda | Azul = desacelerando',
                          fontsize=11, fontweight='bold')
        axes[1].set_xlim(-5, 3)
        axes[1].set_ylim(-1, 1)
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'{region_name} — Aceleração (d²h/dt²)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(plots_dir / 'acceleration_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ acceleration_analysis.png")
    plt.close()

# ============================================
# RESUMO FINAL
# ============================================

print(f"\n{'=' * 70}")
print("RESUMO DA ANÁLISE ESTATÍSTICA")
print("=" * 70)

print(f"\n📊 {region_name} — Inverno Austral (JJA)")

print(f"\n❄️ DADOS PONTUAIS (Script 11):")
print(f"   Nós: {len(pt_dhdt):,}")
print(f"   dh/dt = {pt_stats['mean']:+.4f} ± {pt_stats['std']:.4f} m/ano")
if has_p1_err:
    print(f"   Erro formal médio: {np.mean(sigma_valid):.4f} m/ano")
if has_p2:
    print(f"   Aceleração média: {np.mean(p2_valid):+.5f} m/ano²")

if has_grid:
    print(f"\n🗺️  GRADE INTERPOLADA (Script 12):")
    print(f"   Células: {len(gr_valid):,}")
    print(f"   dh/dt = {gr_stats['mean']:+.4f} ± {gr_stats['std']:.4f} m/ano")

thinning = np.sum(pt_dhdt < 0)
thickening = np.sum(pt_dhdt > 0)
print(f"\n📉 Adelgaçamento: {thinning:,} ({100*thinning/len(pt_dhdt):.1f}%)")
print(f"   Taxa média: {np.mean(pt_dhdt[pt_dhdt < 0]):+.4f} m/ano")
print(f"\n📈 Espessamento: {thickening:,} ({100*thickening/len(pt_dhdt):.1f}%)")
if thickening > 0:
    print(f"   Taxa média: {np.mean(pt_dhdt[pt_dhdt > 0]):+.4f} m/ano")

print(f"\n📁 CSVs: {stats_dir}")
print(f"📊 Gráficos: {plots_dir}")

all_plots = [
    'histogram_comparison.png', 'cdf_comparison.png',
    'boxplot_comparison.png', 'violin_by_latitude.png',
    'scatter_dhdt_vs_latitude.png', 'area_volume_by_rate.png',
    'error_analysis.png', 'acceleration_analysis.png',
]
print(f"\n   Gráficos gerados:")
for p in all_plots:
    f = plots_dir / p
    if f.exists():
        print(f"   ✓ {p}")

print("\n" + "=" * 70)
print("✓ Análise estatística concluída!")

print("\nPróximo passo: 14_plot_maps.py")
