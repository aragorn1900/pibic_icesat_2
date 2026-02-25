"""
Junção de tiles de dh/dt em arquivo único
Equivalente a: join.py do CAPTOOLKIT

Atualizado para Script 10 v5.1b:
  - Lê datasets + attrs do HDF5 (formato novo)
  - Inclui novas variáveis: p0, p2, p0_error, p2_error, rmse, dmin, t_ref
  - Junta séries temporais sec(t) dos tiles
  - Salva com h5py direto (evita erros de tipo)

CAPTOOLKIT Original:
    join.py -o output.h5 tiles/*.h5
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import h5py
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts\utils")
from io_utils import read_hdf5

print("=" * 70)
print("JUNÇÃO DE TILES DE dh/dt")
print("Baseado em: join.py do CAPTOOLKIT")
print("Compatível com: Script 10 v5.1b")
print("=" * 70)

# ============================================
# RESOLVER DIRETÓRIOS
# ============================================

BASE_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")

try:
    _dhdt_dir = DHDT_WINTER_DIR
except NameError:
    candidates = list(BASE_DIR.rglob("*dhdt*"))
    candidates = [d for d in candidates if d.is_dir() and list(d.glob("*_dhdt.h5"))]
    if candidates:
        _dhdt_dir = sorted(candidates, key=lambda p: len(list(p.glob("*_dhdt.h5"))),
                           reverse=True)[0]
    else:
        print("✗ Diretório de dh/dt não encontrado!")
        sys.exit(1)
DHDT_WINTER_DIR = _dhdt_dir

try:
    _results_dir = RESULTS_DIR
except NameError:
    _results_dir = BASE_DIR / 'results'
RESULTS_DIR = _results_dir

try:
    _grid_res = GRID_RESOLUTION
except NameError:
    _grid_res = 2000
GRID_RES = _grid_res

# ============================================
# PARÂMETROS
# ============================================

DHDT_PHYSICAL_LIMIT = 15.0  # m/ano — igual ao RATE_LIM do Script 10

print(f"\nParâmetros:")
print(f"  Resolução da grade: {GRID_RES} m")
print(f"  Limite físico de dh/dt: ±{DHDT_PHYSICAL_LIMIT} m/ano")

# ============================================
# VARIÁVEIS A JUNTAR
# ============================================

# Variáveis 1D (um valor por nó) — datasets HDF5
# Ordem de prioridade: se existir, inclui; se não, ignora
NODE_VARS = [
    # Coordenadas
    'x', 'y', 'longitude', 'latitude',
    # Parâmetros fitsec (novos no v5.1b)
    'p0', 'p1', 'p2',
    'p0_error', 'p1_error', 'p2_error',
    'rmse', 'nobs', 'dmin', 'tspan', 't_ref',
    # Aliases (compatibilidade)
    'dhdt', 'dhdt_sigma', 'n_points_used', 'time_span',
]

# ============================================
# FUNÇÃO: LER TILE ROBUSTO
# ============================================

def read_tile_robust(filepath):
    """
    Lê um tile HDF5 do Script 10 v5.1b.
    Lê datasets (arrays) e attrs (metadados).
    Compatível com formato antigo (tudo como dataset) e novo (attrs separados).
    """
    data = {}
    attrs = {}

    with h5py.File(filepath, 'r') as f:
        # Datasets
        for key in f.keys():
            try:
                ds = f[key][:]
                data[key] = ds
            except Exception:
                pass

        # Attrs
        for key in f.attrs:
            attrs[key] = f.attrs[key]

    return data, attrs

# ============================================
# LISTAR TILES
# ============================================

print("\n1. Listando tiles de dh/dt...")
dhdt_files = sorted(list(DHDT_WINTER_DIR.glob("*_dhdt.h5")))
print(f"   Encontrados: {len(dhdt_files)} tiles")

if len(dhdt_files) == 0:
    print("\n✗ Nenhum tile encontrado!")
    sys.exit(1)

# ============================================
# DETERMINAR VARIÁVEIS DISPONÍVEIS
# ============================================

print("\n2. Determinando estrutura de dados...")

first_data, first_attrs = read_tile_robust(dhdt_files[0])

# Descobrir quais variáveis 1D existem e seu tamanho de referência
ref_key = 'dhdt' if 'dhdt' in first_data else 'p1'
if ref_key not in first_data:
    print("✗ Nenhuma variável dhdt ou p1 encontrada!")
    sys.exit(1)

ref_len = len(first_data[ref_key])

# Variáveis que são 1D e têm o mesmo tamanho que dhdt
available_vars = []
for var in NODE_VARS:
    if var in first_data:
        if isinstance(first_data[var], np.ndarray) and len(first_data[var]) == ref_len:
            available_vars.append(var)

# Descobrir variáveis extras que não estão na lista
for var in first_data:
    if var not in available_vars and var not in ['time', 'sec_t', 'rms_t', 'sec_node_idx']:
        if isinstance(first_data[var], np.ndarray) and len(first_data[var]) == ref_len:
            available_vars.append(var)

print(f"   Variáveis de nó ({len(available_vars)}):")
for var in available_vars:
    print(f"      - {var} ({first_data[var].dtype})")

# Séries temporais?
has_timeseries = 'sec_t' in first_data and 'time' in first_data
if has_timeseries:
    print(f"\n   Séries temporais detectadas!")
    print(f"      time: {len(first_data['time'])} bins")
    print(f"      sec_t: {first_data['sec_t'].shape}")

print(f"\n   Attrs do tile: {list(first_attrs.keys())}")

del first_data, first_attrs

# ============================================
# LER TODOS OS TILES
# ============================================

print("\n3. Lendo todos os tiles...")

all_data = {var: [] for var in available_vars}
all_sec_t = []    # Séries temporais
all_rms_t = []
all_sec_lon = []  # Coordenadas das séries
all_sec_lat = []
common_time = None
n_tiles_merged = 0

for dhdt_file in tqdm(dhdt_files, desc="Lendo tiles"):
    try:
        tile_data, tile_attrs = read_tile_robust(dhdt_file)

        # Verificar que tem a variável de referência
        if ref_key not in tile_data:
            continue

        dhdt_arr = tile_data[ref_key]
        if not isinstance(dhdt_arr, np.ndarray) or len(dhdt_arr) == 0:
            continue

        n_pts = len(dhdt_arr)

        # Concatenar variáveis de nó
        for var in available_vars:
            if var in tile_data:
                value = tile_data[var]
                if isinstance(value, np.ndarray) and len(value) == n_pts:
                    all_data[var].append(value.astype(np.float64))
                else:
                    all_data[var].append(np.full(n_pts, np.nan, dtype=np.float64))
            else:
                all_data[var].append(np.full(n_pts, np.nan, dtype=np.float64))

        # Séries temporais
        if has_timeseries and 'sec_t' in tile_data and 'sec_node_idx' in tile_data:
            sec_idx = tile_data['sec_node_idx'].astype(int)
            sec_t = tile_data['sec_t']
            rms_t = tile_data.get('rms_t', np.full_like(sec_t, np.nan))
            t_bin = tile_data['time']

            # Guardar time vector (deve ser igual para todos os tiles)
            if common_time is None:
                common_time = t_bin
            else:
                # Verificar compatibilidade — usar o mais longo
                if len(t_bin) > len(common_time):
                    common_time = t_bin

            # Coordenadas dos nós com série temporal
            if 'longitude' in tile_data and 'latitude' in tile_data:
                lon_nodes = tile_data['longitude']
                lat_nodes = tile_data['latitude']
            else:
                lon_nodes = np.full(n_pts, np.nan)
                lat_nodes = np.full(n_pts, np.nan)

            for i, ni in enumerate(sec_idx):
                if ni < n_pts:
                    all_sec_t.append(sec_t[i])
                    all_rms_t.append(rms_t[i])
                    all_sec_lon.append(lon_nodes[ni] if ni < len(lon_nodes) else np.nan)
                    all_sec_lat.append(lat_nodes[ni] if ni < len(lat_nodes) else np.nan)

        n_tiles_merged += 1

    except Exception as e:
        print(f"\n⚠ Erro ao ler {dhdt_file.name}: {e}")
        continue

# ============================================
# CONCATENAR
# ============================================

print("\n4. Concatenando arrays...")

for var in tqdm(available_vars, desc="Concatenando"):
    if len(all_data[var]) > 0:
        all_data[var] = np.concatenate(all_data[var])
    else:
        del all_data[var]
        available_vars.remove(var)

n_raw = len(all_data.get('dhdt', all_data.get('p1', [])))
print(f"   Pontos brutos (com sobreposições): {n_raw:,}")

# ============================================
# REMOVER SOBREPOSIÇÕES (COMO join.py)
# ============================================

print("\n5. Removendo sobreposições entre tiles...")

x = all_data['x']
y = all_data['y']

x_quant = np.round(x / GRID_RES).astype(np.int64)
y_quant = np.round(y / GRID_RES).astype(np.int64)

x_offset = x_quant - x_quant.min()
y_offset = y_quant - y_quant.min()
nx = int(x_offset.max()) + 1

grid_key = y_offset * nx + x_offset

unique_keys, inverse, counts = np.unique(grid_key, return_inverse=True,
                                          return_counts=True)

n_unique = len(unique_keys)
n_duplicates = n_raw - n_unique
pct_dup = 100 * n_duplicates / n_raw

print(f"   Posições únicas: {n_unique:,}")
print(f"   Duplicatas: {n_duplicates:,} ({pct_dup:.1f}%)")

if n_duplicates > 0:
    # Critério: menor p1_error (erro formal do dh/dt)
    # Fallback: menor dhdt_sigma → maior nobs
    if 'p1_error' in all_data:
        sigma = all_data['p1_error']
        criterion_name = 'p1_error (erro formal)'
    elif 'dhdt_sigma' in all_data:
        sigma = all_data['dhdt_sigma']
        criterion_name = 'dhdt_sigma'
    else:
        sigma = None
        criterion_name = 'primeiro encontrado'

    if sigma is not None:
        sigma_priority = np.where(np.isnan(sigma) | (sigma <= 0),
                                   999.0, sigma)
    else:
        sigma_priority = np.arange(n_raw, dtype=float)

    print(f"   Critério de seleção: menor {criterion_name}")

    best_indices = np.full(n_unique, -1, dtype=np.int64)
    best_priority = np.full(n_unique, np.inf, dtype=np.float64)

    for i in tqdm(range(n_raw), desc="   Selecionando melhores",
                  disable=n_raw < 100000):
        key_idx = inverse[i]
        if sigma_priority[i] < best_priority[key_idx]:
            best_priority[key_idx] = sigma_priority[i]
            best_indices[key_idx] = i

    assert np.all(best_indices >= 0)

    merged_data = {}
    for var in all_data:
        if isinstance(all_data[var], np.ndarray) and len(all_data[var]) == n_raw:
            merged_data[var] = all_data[var][best_indices]
        else:
            merged_data[var] = all_data[var]

    n_points = n_unique
    print(f"   ✓ Sobreposições removidas: {n_raw:,} → {n_points:,}")

else:
    merged_data = all_data
    n_points = n_raw

del all_data

# ============================================
# FILTRAR VALORES VÁLIDOS
# ============================================

print("\n6. Filtrando valores válidos...")

dhdt_key = 'dhdt' if 'dhdt' in merged_data else 'p1'
valid_mask = ~np.isnan(merged_data[dhdt_key])
n_valid = int(np.sum(valid_mask))

print(f"   Nós válidos: {n_valid:,} / {n_points:,} ({100*n_valid/n_points:.1f}%)")

if n_valid < n_points:
    for var in list(merged_data.keys()):
        if isinstance(merged_data[var], np.ndarray) and len(merged_data[var]) == n_points:
            merged_data[var] = merged_data[var][valid_mask]
    n_points = n_valid

# ============================================
# FILTRO DE OUTLIERS FÍSICOS
# ============================================

print("\n7. Filtrando outliers físicos...")

dhdt_vals = merged_data[dhdt_key]
physical_mask = np.abs(dhdt_vals) < DHDT_PHYSICAL_LIMIT

n_outliers = int(np.sum(~physical_mask))
pct_outliers = 100 * n_outliers / n_points if n_points > 0 else 0

print(f"   Limite: |dh/dt| < {DHDT_PHYSICAL_LIMIT} m/ano")
print(f"   Outliers removidos: {n_outliers:,} ({pct_outliers:.2f}%)")

if n_outliers > 0:
    outlier_vals = dhdt_vals[~physical_mask]
    print(f"   Outliers — min: {outlier_vals.min():.1f}, max: {outlier_vals.max():.1f} m/ano")

    for var in list(merged_data.keys()):
        if isinstance(merged_data[var], np.ndarray) and len(merged_data[var]) == n_points:
            merged_data[var] = merged_data[var][physical_mask]

    n_points = int(np.sum(physical_mask))

# ============================================
# SALVAR (com h5py direto — robusto)
# ============================================

print("\n8. Salvando arquivo mesclado...")

output_dir = RESULTS_DIR / 'dhdt_winter'
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / 'amundsen_sea_dhdt_winter_joined.h5'

with h5py.File(output_file, 'w') as f:

    # --- Datasets de nó (arrays 1D) ---
    for var in merged_data:
        val = merged_data[var]
        if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) > 1:
            f.create_dataset(var, data=val, compression='gzip',
                             compression_opts=4)

    # --- Séries temporais ---
    if has_timeseries and len(all_sec_t) > 0 and common_time is not None:
        n_ts = len(all_sec_t)

        # Padronizar tamanho (tiles podem ter t_bin de tamanhos diferentes)
        n_time = len(common_time)
        sec_matrix = np.full((n_ts, n_time), np.nan, dtype=np.float64)
        rms_matrix = np.full((n_ts, n_time), np.nan, dtype=np.float64)

        for i in range(n_ts):
            n_fill = min(len(all_sec_t[i]), n_time)
            sec_matrix[i, :n_fill] = all_sec_t[i][:n_fill]
            rms_matrix[i, :n_fill] = all_rms_t[i][:n_fill]

        grp = f.create_group('timeseries')
        grp.create_dataset('time', data=common_time)
        grp.create_dataset('sec_t', data=sec_matrix, compression='gzip',
                           compression_opts=4)
        grp.create_dataset('rms_t', data=rms_matrix, compression='gzip',
                           compression_opts=4)
        grp.create_dataset('lon', data=np.array(all_sec_lon))
        grp.create_dataset('lat', data=np.array(all_sec_lat))

        print(f"   Séries temporais: {n_ts:,} nós × {n_time} bins")

    # --- Atributos (metadados) ---
    f.attrs['n_tiles_joined'] = n_tiles_merged
    f.attrs['n_points_total'] = n_points
    f.attrs['n_duplicates_removed'] = n_duplicates
    f.attrs['n_outliers_removed'] = n_outliers
    f.attrs['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.attrs['season'] = 'Winter Austral (JJA)'
    f.attrs['grid_resolution'] = GRID_RES
    f.attrs['dhdt_physical_limit'] = DHDT_PHYSICAL_LIMIT
    f.attrs['script_version'] = '11_join_tiles v2.0 (fitsec v5.1b compatible)'

    try:
        f.attrs['region'] = THWAITES_BBOX['name']
        f.attrs['search_radius'] = FITSEC_PARAMS['search_radius']
        f.attrs['poly_order'] = FITSEC_PARAMS['poly_order']
    except:
        pass

file_size_mb = output_file.stat().st_size / (1024**2)
print(f"   ✓ Salvo: {output_file}")
print(f"   Tamanho: {file_size_mb:.1f} MB")

# ============================================
# ESTATÍSTICAS
# ============================================

print("\n" + "=" * 70)
print("RESUMO DA JUNÇÃO")
print("=" * 70)
print(f"Tiles mesclados: {n_tiles_merged}")
print(f"Pontos brutos: {n_raw:,}")
print(f"Sobreposições removidas: {n_duplicates:,}")
print(f"Outliers removidos: {n_outliers:,}")
print(f"Nós finais: {n_points:,}")

dhdt_vals = merged_data[dhdt_key]

print(f"\ndh/dt (p1):")
print(f"  Média:   {np.mean(dhdt_vals):+.4f} m/ano")
print(f"  Mediana: {np.median(dhdt_vals):+.4f} m/ano")
print(f"  Std:     {np.std(dhdt_vals):.4f} m/ano")
print(f"  Min:     {np.min(dhdt_vals):+.4f} m/ano")
print(f"  Max:     {np.max(dhdt_vals):+.4f} m/ano")

# Aceleração
if 'p2' in merged_data:
    p2_vals = merged_data['p2']
    p2_valid = p2_vals[~np.isnan(p2_vals)]
    if len(p2_valid) > 0:
        print(f"\nd²h/dt² (p2):")
        print(f"  Média:   {np.mean(p2_valid):+.5f} m/ano²")
        print(f"  Mediana: {np.median(p2_valid):+.5f} m/ano²")
        print(f"  Std:     {np.std(p2_valid):.5f} m/ano²")
        print(f"  Válidos: {len(p2_valid):,} / {n_points:,} ({100*len(p2_valid)/n_points:.1f}%)")

# RMSE
if 'rmse' in merged_data:
    rmse_vals = merged_data['rmse']
    rmse_valid = rmse_vals[~np.isnan(rmse_vals)]
    if len(rmse_valid) > 0:
        print(f"\nRMSE dos resíduos:")
        print(f"  Média:   {np.mean(rmse_valid):.4f} m")
        print(f"  Mediana: {np.median(rmse_valid):.4f} m")

# Erro formal
if 'p1_error' in merged_data:
    err_vals = merged_data['p1_error']
    err_valid = err_vals[~np.isnan(err_vals)]
    if len(err_valid) > 0:
        print(f"\nErro formal de dh/dt (p1_error):")
        print(f"  Média:   {np.mean(err_valid):.4f} m/ano")
        print(f"  Mediana: {np.median(err_valid):.4f} m/ano")

print(f"\nPercentis de dh/dt:")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  {p:3d}%: {np.percentile(dhdt_vals, p):+.4f} m/ano")

thinning = np.sum(dhdt_vals < 0)
thickening = np.sum(dhdt_vals > 0)

print(f"\nClassificação:")
print(f"  Adelgaçamento: {thinning:,} ({100*thinning/n_points:.1f}%)")
print(f"  Espessamento:  {thickening:,} ({100*thickening/n_points:.1f}%)")

if 'x' in merged_data and 'y' in merged_data:
    x = merged_data['x']
    y = merged_data['y']
    area_km2 = n_points * (GRID_RES / 1000) ** 2
    print(f"\nÁrea coberta: ~{area_km2:,.0f} km²")

if has_timeseries and len(all_sec_t) > 0:
    print(f"\nSéries temporais:")
    print(f"  Nós com sec(t): {len(all_sec_t):,}")
    print(f"  Bins temporais: {len(common_time)}")
    print(f"  Período: {common_time.min():.2f} a {common_time.max():.2f}")

print(f"\nArquivo: {output_file}")
print("=" * 70)
print("\n✓ Junção concluída!")
print("\nPróximo passo: 12_create_grid_oi.py")