"""
Visualização dos groundtracks dos feixes do ICESat-2 ATL06

Plota:
  1. Todos os groundtracks na região (overview)
  2. Groundtracks coloridos por época/ano
  3. Groundtracks coloridos por feixe (gt1l, gt1r, gt2l, gt2r, gt3l, gt3r)
  4. Por ciclo orbital
  5. Densidade de cobertura (heatmap)
  6. Zoom em tracks individuais
  7. Padrão orbital (ascending/descending)

Dados de entrada:
  - Tiles do Script 09 (dados brutos com lon, lat, beam, cycle, t_year)
  - OU arquivo joined do Script 11 (lon, lat)
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
from glob import glob
from tqdm import tqdm

sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

print("=" * 70)
print("VISUALIZAÇÃO DE GROUNDTRACKS — ICESat-2 ATL06")
print("=" * 70)

# ============================================
# DIRETÓRIOS
# ============================================

BASE_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")

try:
    _figures_dir = FIGURES_DIR
except NameError:
    _figures_dir = BASE_DIR / 'figures'
FIGURES_DIR = _figures_dir

try:
    _tiles_dir = TILES_WINTER_DIR
except NameError:
    _tiles_dir = BASE_DIR / 'data' / 'tiles_winter'
TILES_DIR = _tiles_dir

try:
    _results_dir = RESULTS_DIR
except NameError:
    _results_dir = BASE_DIR / 'results'
RESULTS_DIR = _results_dir

try:
    region_name = THWAITES_BBOX['name']
    bbox = THWAITES_BBOX
except (NameError, KeyError):
    region_name = 'Amundsen Sea Embayment'
    bbox = None

gt_dir = FIGURES_DIR / 'groundtracks'
gt_dir.mkdir(exist_ok=True, parents=True)

# ============================================
# CARREGAR DADOS
# ============================================

print("\n1. Carregando dados dos groundtracks...")

records = []  # lista de dicts, um por tile

tile_files = sorted(list(TILES_DIR.glob("tile_*.h5")))

if len(tile_files) > 0:
    print(f"   Fonte: {len(tile_files)} tiles")

    SUBSAMPLE = 5

    for tf in tqdm(tile_files, desc="   Lendo tiles"):
        try:
            with h5py.File(tf, 'r') as f:
                # Coordenadas (obrigatório)
                if 'lon' in f:
                    lo = f['lon'][::SUBSAMPLE]
                    la = f['lat'][::SUBSAMPLE]
                elif 'longitude' in f:
                    lo = f['longitude'][::SUBSAMPLE]
                    la = f['latitude'][::SUBSAMPLE]
                else:
                    continue

                n = len(lo)
                rec = {'lon': lo, 'lat': la}

                # Tempo
                if 't_year' in f:
                    arr = f['t_year'][::SUBSAMPLE]
                    rec['t_year'] = arr[:n] if len(arr) >= n else np.full(n, np.nan)
                elif 'time' in f:
                    arr = f['time'][::SUBSAMPLE]
                    rec['t_year'] = arr[:n] if len(arr) >= n else np.full(n, np.nan)

                # Feixe
                if 'beam' in f:
                    arr = f['beam'][::SUBSAMPLE]
                    rec['beam'] = arr[:n] if len(arr) >= n else np.full(n, np.nan)
                elif 'spot' in f:
                    arr = f['spot'][::SUBSAMPLE]
                    rec['beam'] = arr[:n] if len(arr) >= n else np.full(n, np.nan)

                # Ciclo
                if 'cycle' in f:
                    arr = f['cycle'][::SUBSAMPLE]
                    rec['cycle'] = arr[:n] if len(arr) >= n else np.full(n, np.nan)

                # Elevação
                if 'h_li' in f:
                    arr = f['h_li'][::SUBSAMPLE]
                    rec['h'] = arr[:n] if len(arr) >= n else np.full(n, np.nan)

                records.append(rec)

        except Exception:
            continue

else:
    print("   Tiles não encontrados, tentando arquivo joined...")

    dhdt_dir = RESULTS_DIR / 'dhdt_winter'
    joined_file = None
    for name in ['amundsen_sea_dhdt_winter_joined.h5']:
        candidate = dhdt_dir / name
        if candidate.exists():
            joined_file = candidate
            break

    if joined_file is None:
        print("✗ Nenhum dado encontrado!")
        sys.exit(1)

    with h5py.File(joined_file, 'r') as f:
        records.append({
            'lon': f['longitude'][:],
            'lat': f['latitude'][:],
        })

# ============================================
# CONCATENAR (alinhado por tile)
# ============================================

if len(records) == 0:
    print("✗ Nenhum tile lido com sucesso!")
    sys.exit(1)

lon = np.concatenate([r['lon'] for r in records])
lat = np.concatenate([r['lat'] for r in records])
n_total = len(lon)


def concat_field(records, key):
    """Concatena campo, preenchendo com NaN onde ausente."""
    arrays = []
    for rec in records:
        n = len(rec['lon'])
        if key in rec:
            arrays.append(rec[key][:n].astype(np.float64))
        else:
            arrays.append(np.full(n, np.nan, dtype=np.float64))
    return np.concatenate(arrays)


has_time = any('t_year' in r for r in records)
has_beam = any('beam' in r for r in records)
has_cycle = any('cycle' in r for r in records)
has_h = any('h' in r for r in records)

if has_time:
    t_year = concat_field(records, 't_year')
if has_beam:
    beam = concat_field(records, 'beam')
if has_cycle:
    cycle = concat_field(records, 'cycle')
if has_h:
    h = concat_field(records, 'h')

del records

# Filtrar NaN em coordenadas
valid = ~np.isnan(lon) & ~np.isnan(lat)
lon = lon[valid]
lat = lat[valid]
if has_time:
    t_year = t_year[valid]
if has_beam:
    beam = beam[valid]
if has_cycle:
    cycle = cycle[valid]
if has_h:
    h = h[valid]

n_pts = len(lon)
print(f"   Pontos carregados: {n_pts:,}")
print(f"   Tempo: {'✓' if has_time else '✗'}")
print(f"   Feixe: {'✓' if has_beam else '✗'}")
print(f"   Ciclo: {'✓' if has_cycle else '✗'}")

if has_time:
    print(f"   Período: {np.nanmin(t_year):.2f} a {np.nanmax(t_year):.2f}")
if has_cycle:
    cycles_unique = np.unique(cycle[~np.isnan(cycle)])
    print(f"   Ciclos: {len(cycles_unique)} ({int(cycles_unique.min())} a {int(cycles_unique.max())})")
if has_beam:
    beams_unique = np.unique(beam[~np.isnan(beam)])
    print(f"   Feixes: {beams_unique}")

# Cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ============================================
# FUNÇÕES DE PLOT
# ============================================

def make_map_ax(fig, pos=111, polar=True):
    """Cria eixo de mapa."""
    if HAS_CARTOPY and polar:
        proj = ccrs.SouthPolarStereo()
        ax = fig.add_subplot(pos, projection=proj)
        if bbox:
            ax.set_extent([bbox['lon_min'] - 2, bbox['lon_max'] + 2,
                           bbox['lat_min'] - 1, bbox['lat_max'] + 1],
                          crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='black', linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        gl = ax.gridlines(draw_labels=True, linestyle='--',
                          color='gray', alpha=0.4, linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        return ax, ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(pos)
        lat_c = np.nanmean(lat)
        ax.set_aspect(1 / np.cos(np.radians(lat_c)))
        ax.set_xlabel('Longitude (°)', fontsize=11)
        ax.set_ylabel('Latitude (°)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        return ax, None


# Subsample para plots
n_plot = min(500000, n_pts)
idx = np.random.choice(n_pts, n_plot, replace=False) if n_pts > n_plot else np.arange(n_pts)

kwargs_scatter = dict(s=0.1, alpha=0.15, rasterized=True)

# Extensão do mapa
if bbox:
    extent = [bbox['lon_min'] - 2, bbox['lon_max'] + 2,
              bbox['lat_min'] - 1, bbox['lat_max'] + 1]
else:
    extent = [np.nanmin(lon) - 1, np.nanmax(lon) + 1,
              np.nanmin(lat) - 1, np.nanmax(lat) + 1]

# Labels dos feixes
beam_colors = {
    1: '#e41a1c', 2: '#ff7f00',
    3: '#4daf4a', 4: '#377eb8',
    5: '#984ea3', 6: '#a65628',
}
beam_labels = {
    1: 'GT1L (strong)', 2: 'GT1R (weak)',
    3: 'GT2L (strong)', 4: 'GT2R (weak)',
    5: 'GT3L (strong)', 6: 'GT3R (weak)',
}

# ============================================
# PLOT 1: OVERVIEW — TODOS OS GROUNDTRACKS
# ============================================

print("\n2. Plot 1: Overview dos groundtracks...")

fig = plt.figure(figsize=(14, 12))
ax, tr = make_map_ax(fig)

if tr:
    ax.scatter(lon[idx], lat[idx], c='steelblue', transform=tr, **kwargs_scatter)
else:
    ax.scatter(lon[idx], lat[idx], c='steelblue', **kwargs_scatter)

ax.set_title(f'{region_name} — Groundtracks ICESat-2 ATL06\n'
             f'{n_pts:,} pontos (sub-amostrados 1:{SUBSAMPLE if len(tile_files) > 0 else 1})',
             fontsize=13, fontweight='bold')

plt.tight_layout()
f1 = gt_dir / 'groundtracks_overview.png'
plt.savefig(f1, dpi=300, bbox_inches='tight')
print(f"   ✓ {f1.name}")
plt.close()

# ============================================
# PLOT 2: POR ÉPOCA (ANO)
# ============================================

if has_time:
    print("   Plot 2: Groundtracks por época...")

    fig = plt.figure(figsize=(14, 12))
    ax, tr = make_map_ax(fig)

    t_valid = t_year[idx]
    t_finite = t_valid[np.isfinite(t_valid)]
    year_min = int(np.floor(np.nanmin(t_finite)))
    year_max = int(np.ceil(np.nanmax(t_finite)))

    cmap_year = plt.cm.viridis
    norm_year = mcolors.Normalize(vmin=year_min, vmax=year_max)

    if tr:
        sc = ax.scatter(lon[idx], lat[idx], c=t_year[idx],
                        cmap=cmap_year, norm=norm_year,
                        transform=tr, **kwargs_scatter)
    else:
        sc = ax.scatter(lon[idx], lat[idx], c=t_year[idx],
                        cmap=cmap_year, norm=norm_year,
                        **kwargs_scatter)

    plt.colorbar(sc, ax=ax, shrink=0.7, label='Ano')

    ax.set_title(f'{region_name} — Groundtracks por Época\n'
                 f'{year_min} a {year_max}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    f2 = gt_dir / 'groundtracks_by_year.png'
    plt.savefig(f2, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f2.name}")
    plt.close()

# ============================================
# PLOT 3: POR FEIXE (BEAM)
# ============================================

if has_beam:
    print("   Plot 3: Groundtracks por feixe...")

    fig = plt.figure(figsize=(14, 12))
    ax, tr = make_map_ax(fig)

    for b_id in sorted(beams_unique):
        b_int = int(b_id)
        if b_int not in beam_colors:
            continue

        b_mask = beam[idx] == b_id
        if np.sum(b_mask) == 0:
            continue

        label = beam_labels.get(b_int, f'Beam {b_int}')
        color = beam_colors[b_int]

        if tr:
            ax.scatter(lon[idx][b_mask], lat[idx][b_mask],
                       c=color, label=label, transform=tr,
                       s=0.3, alpha=0.2, rasterized=True)
        else:
            ax.scatter(lon[idx][b_mask], lat[idx][b_mask],
                       c=color, label=label,
                       s=0.3, alpha=0.2, rasterized=True)

    ax.legend(markerscale=30, fontsize=10, loc='upper right',
              framealpha=0.9, title='Feixes ICESat-2')

    ax.set_title(f'{region_name} — Groundtracks por Feixe\n'
                 f'6 feixes: 3 pares (strong/weak)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    f3 = gt_dir / 'groundtracks_by_beam.png'
    plt.savefig(f3, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f3.name}")
    plt.close()

# ============================================
# PLOT 4: POR CICLO
# ============================================

if has_cycle and len(cycles_unique) > 1:
    print("   Plot 4: Groundtracks por ciclo...")

    n_cycles = len(cycles_unique)
    if n_cycles > 6:
        step = max(1, n_cycles // 6)
        selected_cycles = cycles_unique[::step][:6]
    else:
        selected_cycles = cycles_unique

    n_sel = len(selected_cycles)
    n_cols = min(3, n_sel)
    n_rows = int(np.ceil(n_sel / n_cols))

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows + 1))

    for i, cyc in enumerate(selected_cycles):
        c_mask = cycle == cyc

        if HAS_CARTOPY:
            proj = ccrs.SouthPolarStereo()
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection=proj)
            if bbox:
                ax.set_extent([bbox['lon_min'] - 2, bbox['lon_max'] + 2,
                               bbox['lat_min'] - 1, bbox['lat_max'] + 1],
                              crs=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', linewidth=0.5)
            ax.scatter(lon[c_mask], lat[c_mask], c='steelblue',
                       s=0.1, alpha=0.3, rasterized=True,
                       transform=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.scatter(lon[c_mask], lat[c_mask], c='steelblue',
                       s=0.1, alpha=0.3, rasterized=True)
            lat_c = np.nanmean(lat)
            ax.set_aspect(1 / np.cos(np.radians(lat_c)))

        n_c = int(np.sum(c_mask))

        if has_time:
            t_c = t_year[c_mask]
            t_c_finite = t_c[np.isfinite(t_c)]
            epoch = f'{np.nanmean(t_c_finite):.1f}' if len(t_c_finite) > 0 else '?'
        else:
            epoch = '?'

        ax.set_title(f'Ciclo {int(cyc)} (~{epoch})\n{n_c:,} pts',
                     fontsize=10, fontweight='bold')

    fig.suptitle(f'{region_name} — Groundtracks por Ciclo ICESat-2',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    f4 = gt_dir / 'groundtracks_by_cycle.png'
    plt.savefig(f4, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f4.name}")
    plt.close()

# ============================================
# PLOT 5: DENSIDADE DE COBERTURA (HEATMAP)
# ============================================

print("   Plot 5: Densidade de cobertura...")

fig = plt.figure(figsize=(14, 12))
ax, tr = make_map_ax(fig)

n_bins = 200
lon_range = [np.nanmin(lon), np.nanmax(lon)]
lat_range = [np.nanmin(lat), np.nanmax(lat)]

H, xedges, yedges = np.histogram2d(lon, lat, bins=n_bins,
                                      range=[lon_range, lat_range])

H_log = np.log10(H.T + 1)
H_log[H_log == 0] = np.nan

lon_c = (xedges[:-1] + xedges[1:]) / 2
lat_c = (yedges[:-1] + yedges[1:]) / 2
Lon_h, Lat_h = np.meshgrid(lon_c, lat_c)

if tr:
    im = ax.pcolormesh(Lon_h, Lat_h, H_log, transform=tr,
                        cmap='hot_r', shading='auto', rasterized=True)
else:
    im = ax.pcolormesh(Lon_h, Lat_h, H_log,
                        cmap='hot_r', shading='auto', rasterized=True)

cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='log₁₀(N pontos + 1)')

ax.set_title(f'{region_name} — Densidade de Cobertura ICESat-2\n'
             f'Total: {n_pts:,} pontos',
             fontsize=13, fontweight='bold')

plt.tight_layout()
f5 = gt_dir / 'groundtracks_density.png'
plt.savefig(f5, dpi=300, bbox_inches='tight')
print(f"   ✓ {f5.name}")
plt.close()

# ============================================
# PLOT 6: ZOOM — TRACKS INDIVIDUAIS
# ============================================

print("   Plot 6: Zoom em tracks individuais...")

if bbox:
    lon_center = (bbox['lon_min'] + bbox['lon_max']) / 2
    lat_center = (bbox['lat_min'] + bbox['lat_max']) / 2
else:
    lon_center = np.nanmean(lon)
    lat_center = np.nanmean(lat)

d_lon = 2.0
d_lat = 0.5

zoom_mask = ((lon >= lon_center - d_lon) & (lon <= lon_center + d_lon) &
             (lat >= lat_center - d_lat) & (lat <= lat_center + d_lat))

n_zoom = int(np.sum(zoom_mask))

if n_zoom > 100:
    fig, ax = plt.subplots(figsize=(14, 8))

    if has_time:
        sc = ax.scatter(lon[zoom_mask], lat[zoom_mask],
                        c=t_year[zoom_mask], cmap='viridis',
                        s=2, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label='Ano', shrink=0.8)
    elif has_beam:
        for b_id in sorted(beams_unique):
            b_int = int(b_id)
            bm = zoom_mask & (beam == b_id)
            if np.sum(bm) > 0:
                label = beam_labels.get(b_int, f'Beam {b_int}')
                color = beam_colors.get(b_int, 'gray')
                ax.scatter(lon[bm], lat[bm], c=color, label=label,
                           s=3, alpha=0.6, rasterized=True)
        ax.legend(markerscale=5, fontsize=9)
    else:
        ax.scatter(lon[zoom_mask], lat[zoom_mask],
                   c='steelblue', s=2, alpha=0.5, rasterized=True)

    lat_c_zoom = np.nanmean(lat[zoom_mask])
    ax.set_aspect(1 / np.cos(np.radians(lat_c_zoom)))
    ax.set_xlabel('Longitude (°)', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_title(f'{region_name} — Zoom: Tracks Individuais\n'
                 f'Lon: [{lon_center-d_lon:.1f}°, {lon_center+d_lon:.1f}°] | '
                 f'Lat: [{lat_center-d_lat:.1f}°, {lat_center+d_lat:.1f}°] | '
                 f'{n_zoom:,} pontos',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    f6 = gt_dir / 'groundtracks_zoom.png'
    plt.savefig(f6, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f6.name}")
    plt.close()
else:
    print(f"   ⚠ Poucos pontos na janela de zoom ({n_zoom})")
    f6 = None

# ============================================
# PLOT 7: PADRÃO ORBITAL (ASCENDING/DESCENDING)
# ============================================

if has_time and n_pts > 1000:
    print("   Plot 7: Ascending vs Descending...")

    fig = plt.figure(figsize=(14, 12))
    ax, tr = make_map_ax(fig)

    lon_quant = np.round(lon * 10) % 2

    mask_a = lon_quant[idx] == 0
    mask_b = lon_quant[idx] == 1

    if tr:
        ax.scatter(lon[idx][mask_a], lat[idx][mask_a],
                   c='#e41a1c', s=0.1, alpha=0.15, rasterized=True,
                   transform=tr, label='Grupo A')
        ax.scatter(lon[idx][mask_b], lat[idx][mask_b],
                   c='#377eb8', s=0.1, alpha=0.15, rasterized=True,
                   transform=tr, label='Grupo B')
    else:
        ax.scatter(lon[idx][mask_a], lat[idx][mask_a],
                   c='#e41a1c', s=0.1, alpha=0.15, rasterized=True,
                   label='Grupo A')
        ax.scatter(lon[idx][mask_b], lat[idx][mask_b],
                   c='#377eb8', s=0.1, alpha=0.15, rasterized=True,
                   label='Grupo B')

    ax.legend(markerscale=50, fontsize=11)
    ax.set_title(f'{region_name} — Padrão de Cobertura Orbital\n'
                 f'Padrão de repetição do ICESat-2 (91 dias)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    f7 = gt_dir / 'groundtracks_orbital_pattern.png'
    plt.savefig(f7, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f7.name}")
    plt.close()

# ============================================
# RESUMO
# ============================================

print("\n" + "=" * 70)
print("GROUNDTRACKS PLOTADOS")
print("=" * 70)

all_plots = [
    ('groundtracks_overview.png', 'Overview geral'),
    ('groundtracks_by_year.png', 'Por época/ano'),
    ('groundtracks_by_beam.png', 'Por feixe (6 beams)'),
    ('groundtracks_by_cycle.png', 'Por ciclo orbital'),
    ('groundtracks_density.png', 'Densidade de cobertura'),
    ('groundtracks_zoom.png', 'Zoom em tracks individuais'),
    ('groundtracks_orbital_pattern.png', 'Padrão orbital'),
]

for fname, desc in all_plots:
    f = gt_dir / fname
    if f.exists():
        print(f"   ✓ {fname:40s} — {desc}")

print(f"\nTotal de pontos: {n_pts:,}")
if has_time:
    print(f"Período: {np.nanmin(t_year):.2f} — {np.nanmax(t_year):.2f}")
if has_cycle:
    print(f"Ciclos: {len(cycles_unique)}")
if has_beam:
    print(f"Feixes: {len(beams_unique)}")
print(f"\nDiretório: {gt_dir}")
print("=" * 70)