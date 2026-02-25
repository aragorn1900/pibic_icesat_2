"""
Interpolação Gaussiana de dh/dt para grade regular 2D
"""

import numpy as np
import h5py
from pathlib import Path
import sys
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy import stats
import gc

sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts\utils")
from io_utils import read_hdf5
from geodetic_utils import xy_to_lonlat

print("=" * 70)
print("INTERPOLAÇÃO GAUSSIANA DE dh/dt PARA GRADE 2D")
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

# ============================================
# PARÂMETROS (equivalentes ao interpgaus.py)
# ============================================

try:
    GRID_DX = GRID_RESOLUTION
    GRID_DY = GRID_RESOLUTION
except NameError:
    GRID_DX = 2000
    GRID_DY = 2000

NOBS_PER_QUADRANT = 25      # -n 25
SEARCH_RADIUS_KM = 10.0     # -r 10 (km)
ALPHA_KM = 5.0              # -a 5  (km) — correlation length
FILTER_DIM_KM = 10.0        # -c 10 3
FILTER_SIGMA = 3.0

# Converter
SEARCH_RADIUS = SEARCH_RADIUS_KM * 1e3
ALPHA = ALPHA_KM * 1e3
FILTER_DIM = FILTER_DIM_KM * 1e3

# Variáveis extras a interpolar (além de dhdt)
INTERP_EXTRAS = True  # Interpolar p2, rmse, p0?

print(f"\nParâmetros (equivalentes ao interpgaus.py):")
print(f"  Grade: {GRID_DX/1000:.0f} × {GRID_DY/1000:.0f} km (-d)")
print(f"  Obs/quadrante: {NOBS_PER_QUADRANT} (-n)")
print(f"  Raio de busca: {SEARCH_RADIUS_KM:.0f} km (-r)")
print(f"  Correlation length: {ALPHA_KM:.0f} km (-a)")
print(f"  Pré-filtro: {FILTER_DIM_KM:.0f} km, {FILTER_SIGMA}σ (-c)")
print(f"  Interpolar extras (p2, rmse): {INTERP_EXTRAS}")

# ============================================
# CARREGAR DADOS
# ============================================

print("\n1. Carregando dados do join...")

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

# Ler com h5py (compatível com formato v5.1b)
data = {}
with h5py.File(joined_file, 'r') as f:
    for key in f.keys():
        try:
            data[key] = f[key][:]
        except:
            pass

xp = np.asarray(data['x'], dtype=np.float64)
yp = np.asarray(data['y'], dtype=np.float64)

# dhdt: usar p1 ou dhdt (são o mesmo)
if 'p1' in data:
    zp = np.asarray(data['p1'], dtype=np.float64)
elif 'dhdt' in data:
    zp = np.asarray(data['dhdt'], dtype=np.float64)
else:
    print("✗ Variável dhdt/p1 não encontrada!")
    sys.exit(1)

# Erro: usar p1_error (formal) primeiro, fallback dhdt_sigma
if 'p1_error' in data:
    sp = np.asarray(data['p1_error'], dtype=np.float64)
    error_source = 'p1_error (erro formal)'
elif 'dhdt_sigma' in data:
    sp = np.asarray(data['dhdt_sigma'], dtype=np.float64)
    error_source = 'dhdt_sigma'
else:
    sp = np.ones_like(zp)
    error_source = 'constante (1.0)'

# Limpar erro
sp = np.where((np.isnan(sp)) | (sp <= 0), np.nanmedian(sp[sp > 0]), sp)

# Extras
has_p2 = 'p2' in data
has_rmse = 'rmse' in data
has_p0 = 'p0' in data

if has_p2:
    p2p = np.asarray(data['p2'], dtype=np.float64)
if has_rmse:
    rmsep = np.asarray(data['rmse'], dtype=np.float64)
if has_p0:
    p0p = np.asarray(data['p0'], dtype=np.float64)

# Remover NaN no dhdt
valid = ~np.isnan(zp) & ~np.isnan(xp) & ~np.isnan(yp)
xp = xp[valid]
yp = yp[valid]
zp = zp[valid]
sp = sp[valid]
if has_p2:
    p2p = p2p[valid]
if has_rmse:
    rmsep = rmsep[valid]
if has_p0:
    p0p = p0p[valid]

n_input = len(zp)
print(f"   Pontos de entrada: {n_input:,}")
print(f"   Erro: {error_source}")
if has_p2:
    n_p2_valid = int(np.sum(~np.isnan(p2p)))
    print(f"   p2 disponível: {n_p2_valid:,} válidos ({100*n_p2_valid/n_input:.1f}%)")
if has_rmse:
    print(f"   rmse disponível: ✓")

del data
gc.collect()

# ============================================
# PRÉ-FILTRO ESPACIAL (como interpgaus.py -c)
# ============================================

print(f"\n2. Pré-filtro espacial ({FILTER_DIM_KM:.0f} km, {FILTER_SIGMA}σ)...")

if FILTER_DIM > 0 and FILTER_SIGMA > 0:
    Nn = int((np.abs(yp.max() - yp.min())) / FILTER_DIM) + 1
    Ne = int((np.abs(xp.max() - xp.min())) / FILTER_DIM) + 1

    try:
        f_bin = stats.binned_statistic_2d(
            xp, yp, zp, statistic='median', bins=(Ne, Nn))
        index = f_bin.binnumber
        ind_unique = np.unique(index)

        zp_clean = zp.copy()
        n_removed = 0

        for uid in ind_unique:
            idx = np.where(index == uid)[0]
            zb = zp_clean[idx]
            if len(zb[~np.isnan(zb)]) == 0:
                continue
            dh = zb - np.nanmedian(zb)
            std_val = np.nanstd(dh)
            if std_val > 0:
                outliers = np.abs(dh) > FILTER_SIGMA * std_val
                zp_clean[idx[outliers]] = np.nan
                n_removed += int(np.sum(outliers))

        keep = ~np.isnan(zp_clean)
        xp = xp[keep]
        yp = yp[keep]
        zp = zp_clean[keep]
        sp = sp[keep]
        if has_p2:
            p2p = p2p[keep]
        if has_rmse:
            rmsep = rmsep[keep]
        if has_p0:
            p0p = p0p[keep]

        print(f"   Removidos: {n_removed:,}")
        print(f"   Restantes: {len(zp):,}")
        n_input = len(zp)

    except Exception as e:
        print(f"   ⚠ Pré-filtro falhou: {e}")
else:
    print("   Desativado")

# ============================================
# CONSTRUIR GRADE
# ============================================

print("\n3. Construindo grade de saída...")

margin = 5 * GRID_DX
xmin = np.floor(xp.min() / GRID_DX) * GRID_DX - margin
xmax = np.ceil(xp.max() / GRID_DX) * GRID_DX + margin
ymin = np.floor(yp.min() / GRID_DY) * GRID_DY - margin
ymax = np.ceil(yp.max() / GRID_DY) * GRID_DY + margin

xi_vec = np.arange(xmin, xmax + GRID_DX / 2, GRID_DX)
yi_vec = np.arange(ymin, ymax + GRID_DY / 2, GRID_DY)

Xi, Yi = np.meshgrid(xi_vec, yi_vec)
n_rows, n_cols = Xi.shape
n_cells = Xi.size

print(f"   Grade: {n_rows} × {n_cols} = {n_cells:,} células")

xi = Xi.ravel()
yi = Yi.ravel()

# ============================================
# FUNÇÃO DE INTERPOLAÇÃO (como interpgaus.py)
# ============================================

def interp_gaussian_quadrant(xi, yi, xp, yp, zp, sp, TreeP,
                              alpha, dmax, nobs_per_q, n_input,
                              min_pts=3):
    """
    Interpolação Gaussiana com seleção por 4 quadrantes.
    Replica a lógica central do interpgaus.py.

    Returns:
        zi: valores preditos
        ei: erros de predição
        ni: número de observações
    """
    n_cells = len(xi)
    n_query = nobs_per_q * 5

    zi = np.full(n_cells, np.nan, dtype=np.float64)
    ei = np.full(n_cells, np.nan, dtype=np.float64)
    ni = np.full(n_cells, np.nan, dtype=np.float64)

    for i in tqdm(range(n_cells), desc="   Interpolando", mininterval=2):

        d, idx = TreeP.query([xi[i], yi[i]], k=min(n_query, len(xp)))

        d = np.atleast_1d(d)
        idx = np.atleast_1d(idx)

        valid_mask = (idx < len(xp)) & np.isfinite(d)
        d = d[valid_mask]
        idx = idx[valid_mask]

        if len(d) == 0 or np.min(d) > dmax:
            continue

        x_local = xp[idx]
        y_local = yp[idx]
        z_local = zp[idx]
        s_local = sp[idx]

        # Seleção por 4 quadrantes
        theta = np.degrees(np.arctan2(y_local - yi[i], x_local - xi[i])) + 180

        selected = []
        for lo, hi in [(0, 90), (90, 180), (180, 270), (270, 360)]:
            q_idx = np.where((theta > lo) & (theta <= hi))[0]
            if len(q_idx) == 0:
                continue
            q_sort = q_idx[np.argsort(d[q_idx])]
            selected.extend(q_sort[:nobs_per_q].tolist())

        if len(selected) == 0:
            continue

        selected = np.array(selected)
        z_sel = z_local[selected]
        s_sel = s_local[selected]
        d_sel = d[selected]

        # Filtrar NaN e distância
        ok = np.isfinite(z_sel) & (d_sel <= dmax)
        if np.sum(ok) < min_pts:
            continue

        z_sel = z_sel[ok]
        s_sel = s_sel[ok]
        d_sel = d_sel[ok]

        # Peso Gaussiano: w = (1/σ²) × exp(-d²/2α²)
        w = (1.0 / (s_sel**2 + 1e-10)) * np.exp(-(d_sel**2) / (2 * alpha**2))
        w += 1e-10

        w_sum = np.sum(w)
        zi[i] = np.sum(w * z_sel) / w_sum

        # Erro: RSS(variabilidade + erro sistemático)
        sigma_r = np.sqrt(np.sum(w * (z_sel - zi[i])**2) / w_sum)
        sigma_s = np.mean(s_sel)
        ei[i] = np.sqrt(sigma_r**2 + sigma_s**2)

        ni[i] = len(z_sel)

    return zi, ei, ni

# ============================================
# INTERPOLAÇÃO PRINCIPAL: dh/dt (p1)
# ============================================

print("\n4. Interpolando dh/dt (p1)...")

TreeP = cKDTree(np.c_[xp, yp])

zi, ei, ni = interp_gaussian_quadrant(
    xi, yi, xp, yp, zp, sp, TreeP,
    alpha=ALPHA, dmax=SEARCH_RADIUS,
    nobs_per_q=NOBS_PER_QUADRANT, n_input=n_input
)

# Remodelar para 2D (SEM flipud — coordenadas já estão corretas)
Zi = zi.reshape(Xi.shape)
Ei = ei.reshape(Xi.shape)
Ni = ni.reshape(Xi.shape)

n_filled = int(np.sum(~np.isnan(Zi)))
pct_filled = 100 * n_filled / n_cells
print(f"   ✓ Preenchidas: {n_filled:,} / {n_cells:,} ({pct_filled:.1f}%)")

# ============================================
# INTERPOLAÇÃO EXTRAS: p2 (aceleração), rmse
# ============================================

P2i = None
RMSEi = None
P0i = None

if INTERP_EXTRAS:

    if has_p2:
        print("\n5a. Interpolando aceleração (p2)...")
        # Usar somente pontos com p2 válido
        p2_valid_mask = ~np.isnan(p2p)
        n_p2 = int(np.sum(p2_valid_mask))

        if n_p2 > 100:
            xp2 = xp[p2_valid_mask]
            yp2 = yp[p2_valid_mask]
            zp2 = p2p[p2_valid_mask]
            sp2 = sp[p2_valid_mask]

            TreeP2 = cKDTree(np.c_[xp2, yp2])

            p2i, p2ei, p2ni = interp_gaussian_quadrant(
                xi, yi, xp2, yp2, zp2, sp2, TreeP2,
                alpha=ALPHA, dmax=SEARCH_RADIUS,
                nobs_per_q=NOBS_PER_QUADRANT, n_input=n_p2
            )

            P2i = p2i.reshape(Xi.shape)
            n_p2_filled = int(np.sum(~np.isnan(P2i)))
            print(f"   ✓ p2 preenchido: {n_p2_filled:,}")
        else:
            print(f"   ⚠ Poucos pontos com p2 válido ({n_p2})")

    if has_rmse:
        print("\n5b. Interpolando RMSE...")
        rmse_valid_mask = ~np.isnan(rmsep) & (rmsep > 0)
        n_rmse = int(np.sum(rmse_valid_mask))

        if n_rmse > 100:
            xr = xp[rmse_valid_mask]
            yr = yp[rmse_valid_mask]
            zr = rmsep[rmse_valid_mask]
            sr = sp[rmse_valid_mask]

            TreeR = cKDTree(np.c_[xr, yr])

            ri, rei, rni = interp_gaussian_quadrant(
                xi, yi, xr, yr, zr, sr, TreeR,
                alpha=ALPHA, dmax=SEARCH_RADIUS,
                nobs_per_q=NOBS_PER_QUADRANT, n_input=n_rmse
            )

            RMSEi = ri.reshape(Xi.shape)
            print(f"   ✓ RMSE preenchido: {int(np.sum(~np.isnan(RMSEi))):,}")
        else:
            print(f"   ⚠ Poucos pontos com RMSE válido ({n_rmse})")

# ============================================
# COORDENADAS GEOGRÁFICAS
# ============================================

step = '5' if not INTERP_EXTRAS else '6'
print(f"\n{step}. Calculando coordenadas geográficas...")

grid_lon = np.full_like(Xi, np.nan)
grid_lat = np.full_like(Yi, np.nan)

CHUNK = 200
for r0 in range(0, n_rows, CHUNK):
    r1 = min(r0 + CHUNK, n_rows)
    xc = Xi[r0:r1].ravel()
    yc = Yi[r0:r1].ravel()
    lo, la = xy_to_lonlat(xc, yc)
    grid_lon[r0:r1] = lo.reshape(r1 - r0, n_cols)
    grid_lat[r0:r1] = la.reshape(r1 - r0, n_cols)

print(f"   ✓ Lon: {np.nanmin(grid_lon):.2f}° a {np.nanmax(grid_lon):.2f}°")
print(f"   ✓ Lat: {np.nanmin(grid_lat):.2f}° a {np.nanmax(grid_lat):.2f}°")

# ============================================
# SALVAR (h5py direto — robusto)
# ============================================

step_n = '6' if not INTERP_EXTRAS else '7'
print(f"\n{step_n}. Salvando grades...")

grid_dir = RESULTS_DIR / 'grids'
grid_dir.mkdir(exist_ok=True, parents=True)

h5_file = grid_dir / 'amundsen_sea_dhdt_winter_grid.h5'

with h5py.File(h5_file, 'w') as f:
    # Datasets principais
    f.create_dataset('X', data=Xi, compression='gzip', compression_opts=4)
    f.create_dataset('Y', data=Yi, compression='gzip', compression_opts=4)
    f.create_dataset('Z_pred', data=Zi, compression='gzip', compression_opts=4)
    f.create_dataset('Z_rmse', data=Ei, compression='gzip', compression_opts=4)
    f.create_dataset('Z_nobs', data=Ni, compression='gzip', compression_opts=4)
    f.create_dataset('lon', data=grid_lon, compression='gzip', compression_opts=4)
    f.create_dataset('lat', data=grid_lat, compression='gzip', compression_opts=4)

    # Extras interpolados
    if P2i is not None:
        f.create_dataset('Z_accel', data=P2i, compression='gzip', compression_opts=4)
    if RMSEi is not None:
        f.create_dataset('Z_rmse_fit', data=RMSEi, compression='gzip', compression_opts=4)

    # Aliases (compatibilidade com scripts 13-17)
    f.create_dataset('x', data=Xi, compression='gzip', compression_opts=4)
    f.create_dataset('y', data=Yi, compression='gzip', compression_opts=4)
    f.create_dataset('dhdt', data=Zi, compression='gzip', compression_opts=4)
    f.create_dataset('dhdt_smooth', data=Zi, compression='gzip', compression_opts=4)
    f.create_dataset('dhdt_std', data=Ei, compression='gzip', compression_opts=4)
    f.create_dataset('dhdt_sigma', data=Ei, compression='gzip', compression_opts=4)
    f.create_dataset('n_points', data=Ni, compression='gzip', compression_opts=4)
    f.create_dataset('n_obs_per_cell', data=Ni, compression='gzip', compression_opts=4)
    f.create_dataset('x_vec', data=xi_vec)
    f.create_dataset('y_vec', data=yi_vec)

    # Atributos (metadados)
    f.attrs['epsg'] = 3031
    f.attrs['resolution_m'] = GRID_DX
    f.attrs['grid_dx'] = GRID_DX
    f.attrs['grid_dy'] = GRID_DY
    f.attrs['nobs_per_quadrant'] = NOBS_PER_QUADRANT
    f.attrs['search_radius_m'] = SEARCH_RADIUS
    f.attrs['alpha_m'] = ALPHA
    f.attrs['n_input_points'] = n_input
    f.attrs['n_filled'] = n_filled
    f.attrs['method'] = 'Gaussian kernel interpolation with 4-quadrant selection (interpgaus.py)'
    f.attrs['error_source'] = error_source
    f.attrs['script_version'] = '12_create_grid v3.0 (fitsec v5.1b compatible)'
    f.attrs['has_accel'] = int(P2i is not None)
    f.attrs['has_rmse_fit'] = int(RMSEi is not None)

    try:
        f.attrs['region'] = THWAITES_BBOX['name']
    except:
        f.attrs['region'] = 'Amundsen Sea Embayment'

file_size = h5_file.stat().st_size / 1024**2
print(f"   ✓ HDF5: {h5_file.name} ({file_size:.1f} MB)")

# NetCDF
try:
    import xarray as xr

    data_vars = {
        'dhdt': (['y', 'x'], Zi,
                 {'long_name': 'dh/dt', 'units': 'm/year'}),
        'dhdt_rmse': (['y', 'x'], Ei,
                      {'long_name': 'Prediction error', 'units': 'm/year'}),
        'n_obs': (['y', 'x'], Ni.astype(np.float32),
                  {'long_name': 'N observations'}),
        'lon': (['y', 'x'], grid_lon),
        'lat': (['y', 'x'], grid_lat),
    }

    if P2i is not None:
        data_vars['accel'] = (['y', 'x'], P2i,
                              {'long_name': 'd2h/dt2', 'units': 'm/year2'})
    if RMSEi is not None:
        data_vars['rmse_fit'] = (['y', 'x'], RMSEi,
                                 {'long_name': 'Fit RMSE', 'units': 'm'})

    ds = xr.Dataset(data_vars, coords={'x': xi_vec, 'y': yi_vec})
    ds.attrs['title'] = 'ASE dh/dt Winter (JJA) - ICESat-2'
    ds.attrs['method'] = 'Gaussian kernel + 4-quadrant (interpgaus.py)'
    ds.attrs['projection'] = 'EPSG:3031'

    nc_file = grid_dir / 'amundsen_sea_dhdt_winter_grid.nc'
    ds.to_netcdf(nc_file)
    print(f"   ✓ NetCDF: {nc_file.name} ({nc_file.stat().st_size/1024**2:.1f} MB)")
except ImportError:
    print("   ⚠ xarray não disponível")

# ============================================
# RESUMO
# ============================================

print("\n" + "=" * 70)
print("RESUMO DA INTERPOLAÇÃO")
print("=" * 70)
print(f"Método: Gaussian kernel + 4 quadrantes (interpgaus.py)")
print(f"Peso: (1/σ²) × exp(-d²/2α²) usando {error_source}")
print(f"Pontos: {n_input:,}")
print(f"Grade: {n_rows} × {n_cols} = {n_cells:,} células")
print(f"Preenchidas: {n_filled:,} ({pct_filled:.1f}%)")
print(f"α={ALPHA_KM} km, r={SEARCH_RADIUS_KM} km, n={NOBS_PER_QUADRANT}/quadrante")

zi_valid = Zi[~np.isnan(Zi)]

if len(zi_valid) > 0:
    print(f"\ndh/dt (grade):")
    print(f"  Média:   {np.mean(zi_valid):+.4f} m/ano")
    print(f"  Mediana: {np.median(zi_valid):+.4f} m/ano")
    print(f"  Std:     {np.std(zi_valid):.4f} m/ano")
    print(f"  Min:     {np.min(zi_valid):+.4f} m/ano")
    print(f"  Max:     {np.max(zi_valid):+.4f} m/ano")

    thin = np.sum(zi_valid < 0)
    thick = np.sum(zi_valid > 0)
    print(f"\n  Adelgaçamento: {thin:,} ({100*thin/len(zi_valid):.1f}%)")
    print(f"  Espessamento:  {thick:,} ({100*thick/len(zi_valid):.1f}%)")

if P2i is not None:
    p2_valid = P2i[~np.isnan(P2i)]
    if len(p2_valid) > 0:
        print(f"\nd²h/dt² (grade):")
        print(f"  Média:   {np.mean(p2_valid):+.5f} m/ano²")
        print(f"  Mediana: {np.median(p2_valid):+.5f} m/ano²")

ei_valid = Ei[~np.isnan(Ei)]
if len(ei_valid) > 0:
    print(f"\nErro de predição:")
    print(f"  Média:   {np.mean(ei_valid):.4f} m/ano")
    print(f"  Mediana: {np.median(ei_valid):.4f} m/ano")

print(f"\nArquivos: {grid_dir}")
print("=" * 70)
print("\n✓ Grade criada!")

print("\nPróximo passo: 13_statistics.py")
