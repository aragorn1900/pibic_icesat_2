"""
Mapas com Projeção Polar Estereográfica da Antártica
Usa-se Cartopy para projeções polares
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

# Importar configurações
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

# Importar utilitários
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts\utils")
from io_utils import read_hdf5

print("=" * 70)
print("MAPAS COM PROJEÇÃO POLAR ESTEREOGRÁFICA")
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

# Nome da região
try:
    region_name = THWAITES_BBOX['name']
    bbox = THWAITES_BBOX
except (NameError, KeyError):
    region_name = 'Amundsen Sea Embayment'
    bbox = {'lon_min': -140, 'lon_max': -80, 'lat_min': -76, 'lat_max': -70}

# Criar diretório de mapas
maps_dir = FIGURES_DIR / 'maps'
maps_dir.mkdir(exist_ok=True, parents=True)

# ============================================
# VERIFICAR CARTOPY
# ============================================

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("\n✓ Cartopy disponível - usando projeção polar")
except ImportError:
    HAS_CARTOPY = False
    print("\n⚠ Cartopy não disponível - usando projeção simples")

# ============================================
# CARREGAR DADOS
# ============================================

print("\n1. Carregando dados...")

# Buscar grade automaticamente
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
            grid_file = sorted(h5_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]

if grid_file is None:
    print(f"\n✗ Grade não encontrada em: {grid_dir}")
    print("   Execute primeiro: 12_create_grid.py")
    sys.exit(1)

print(f"   Arquivo: {grid_file.name}")

grid_data = read_hdf5(grid_file)

grid_lon = grid_data['lon']
grid_lat = grid_data['lat']

# Escolher melhor versão de dh/dt
if 'dhdt_smooth' in grid_data:
    grid_dhdt = grid_data['dhdt_smooth']
elif 'dhdt_filled' in grid_data:
    grid_dhdt = grid_data['dhdt_filled']
elif 'dhdt' in grid_data:
    grid_dhdt = grid_data['dhdt']
else:
    print("✗ Nenhuma variável de dh/dt encontrada!")
    sys.exit(1)

# Incerteza
if 'dhdt_sigma' in grid_data:
    grid_std = grid_data['dhdt_sigma']
elif 'dhdt_std' in grid_data:
    grid_std = grid_data['dhdt_std']
else:
    grid_std = np.zeros_like(grid_dhdt)

print(f"   ✓ Grade carregada: {grid_dhdt.shape}")

# ============================================
# MAPA 1: PROJEÇÃO POLAR ESTEREOGRÁFICA
# ============================================

if HAS_CARTOPY:

    print("\n2. Criando Mapa 1: Projeção Polar Estereográfica...")

    fig = plt.figure(figsize=(14, 14))

    proj = ccrs.SouthPolarStereo()
    ax = plt.axes(projection=proj)

    ax.set_extent([bbox['lon_min'] - 2,
                   bbox['lon_max'] + 2,
                   bbox['lat_min'] - 1,
                   bbox['lat_max'] + 1],
                  crs=ccrs.PlateCarree())

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    im = ax.pcolormesh(grid_lon, grid_lat, grid_dhdt,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm,
                       shading='auto', rasterized=True)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        pad=0.05, shrink=0.8,
                        label='dh/dt (m/ano)', extend='both')
    cbar.ax.tick_params(labelsize=11)

    ax.coastlines(resolution='50m', color='black', linewidth=1.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)

    gl = ax.gridlines(draw_labels=True, linestyle='--',
                      color='black', alpha=0.5, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f'{region_name} - Taxa de Mudança de Elevação (dh/dt)\n'
                 f'Projeção Estereográfica Polar Sul\n'
                 f'Inverno Austral (JJA) - ICESat-2 ATL06',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    map1_file = maps_dir / 'amundsen_sea_dhdt_polar_stereo.png'
    plt.savefig(map1_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Mapa 1 salvo: {map1_file.name}")
    plt.close()

    # ============================================
    # MAPA 2: CONTEXTO REGIONAL
    # ============================================

    print("\n3. Criando Mapa 2: Contexto Regional...")

    fig = plt.figure(figsize=(16, 16))

    proj = ccrs.SouthPolarStereo()
    ax = plt.axes(projection=proj)

    ax.set_extent([-140, -70, -80, -70], crs=ccrs.PlateCarree())

    im = ax.pcolormesh(grid_lon, grid_lat, grid_dhdt,
                       transform=ccrs.PlateCarree(),
                       cmap='RdBu_r', vmin=-2, vmax=2,
                       shading='auto', rasterized=True, alpha=0.9)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.7,
                        label='dh/dt (m/ano)')

    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='tan', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

    gl = ax.gridlines(draw_labels=True, linestyle='--',
                      color='gray', alpha=0.5)

    # Destacar bounding box
    from matplotlib.patches import Rectangle

    rect = Rectangle((bbox['lon_min'], bbox['lat_min']),
                     bbox['lon_max'] - bbox['lon_min'],
                     bbox['lat_max'] - bbox['lat_min'],
                     linewidth=3, edgecolor='red', facecolor='none',
                     linestyle='--', transform=ccrs.PlateCarree())
    ax.add_patch(rect)

    ax.text(bbox['lon_min'] - 1, bbox['lat_max'] + 0.5,
            f'{region_name}',
            transform=ccrs.PlateCarree(),
            fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'{region_name} no Contexto do West Antarctic Ice Sheet\n'
                 f'Projeção Estereográfica Polar Sul',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()

    map2_file = maps_dir / 'amundsen_sea_regional_context_polar.png'
    plt.savefig(map2_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Mapa 2 salvo: {map2_file.name}")
    plt.close()

    # ============================================
    # MAPA 3: ZOOM COM CONTORNOS
    # ============================================

    print("\n4. Criando Mapa 3: Zoom de Alta Resolução...")

    fig = plt.figure(figsize=(12, 14))

    proj = ccrs.SouthPolarStereo()
    ax = plt.axes(projection=proj)

    ax.set_extent([bbox['lon_min'] - 0.5, bbox['lon_max'] + 0.5,
                   bbox['lat_min'] - 0.3, bbox['lat_max'] + 0.3],
                  crs=ccrs.PlateCarree())

    im = ax.pcolormesh(grid_lon, grid_lat, grid_dhdt,
                       transform=ccrs.PlateCarree(),
                       cmap='RdBu_r', vmin=-2, vmax=2,
                       shading='auto', rasterized=True)

    try:
        levels = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
        cs = ax.contour(grid_lon, grid_lat, grid_dhdt,
                        levels=levels,
                        transform=ccrs.PlateCarree(),
                        colors='black', linewidths=1, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')
    except Exception:
        pass

    cbar = plt.colorbar(im, ax=ax, label='dh/dt (m/ano)', shrink=0.8)

    ax.coastlines(resolution='10m', color='black', linewidth=2)

    gl = ax.gridlines(draw_labels=True, linestyle=':',
                      color='black', alpha=0.6, linewidth=0.5)

    ax.set_title(f'{region_name} - Zoom de Alta Resolução\n'
                 f'dh/dt com Contornos',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    map3_file = maps_dir / 'amundsen_sea_dhdt_zoom_polar.png'
    plt.savefig(map3_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Mapa 3 salvo: {map3_file.name}")
    plt.close()

# ============================================
# MAPA ALTERNATIVA SEM CARTOPY
# ============================================

if not HAS_CARTOPY:

    print("\n2. Criando mapas sem Cartopy (projeção simples)...")

    fig, ax = plt.subplots(figsize=(14, 10))

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    im = ax.pcolormesh(grid_lon, grid_lat, grid_dhdt,
                       cmap=cmap, norm=norm,
                       shading='auto', rasterized=True)

    cbar = plt.colorbar(im, ax=ax, label='dh/dt (m/ano)', extend='both')

    lat_center = (bbox['lat_min'] + bbox['lat_max']) / 2
    aspect_ratio = 1 / np.cos(np.radians(lat_center))
    ax.set_aspect(aspect_ratio)

    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title(f'{region_name} - dh/dt\n(Projeção Geográfica Simples)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    map_file = maps_dir / 'amundsen_sea_dhdt_simple.png'
    plt.savefig(map_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Mapa salvo: {map_file.name}")
    plt.close()

# ============================================
# RESUMO
# ============================================

print("\n" + "=" * 70)
print("MAPAS POLARES CRIADOS")
print("=" * 70)

if HAS_CARTOPY:
    print("\nMapas com Projeção Estereográfica Polar:")
    print(f"  1. amundsen_sea_dhdt_polar_stereo.png")
    print(f"  2. amundsen_sea_regional_context_polar.png")
    print(f"  3. amundsen_sea_dhdt_zoom_polar.png")
else:
    print("\nCartopy não disponível:")
    print(f"  - amundsen_sea_dhdt_simple.png")
    print("\nPara instalar Cartopy:")
    print("  conda install -c conda-forge cartopy")

print(f"\nDiretório: {maps_dir}")
print("=" * 70)

print("\n✓ Mapas polares concluídos!")
