"""
Criação de mapas de dh/dt
Paleta de cores:
  VERMELHO = afinamento / derretimento (dh/dt < 0)
  CINZA CLARO = estável / próximo de zero
  AZUL     = espessamento (dh/dt > 0)
Gera:
  1. Mapa dh/dt principal
  2. Mapa RMSE (erro de predição do interpgaus)
  3. Mapa de cobertura (N observações)
  4. Mapa de classificação de regimes
  5. Mapa de aceleração (d²h/dt²)
  6. Mapa de Δh acumulado
  7. Painel resumo (4-em-1)
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import sys

sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

print("=" * 70)
print("CRIAÇÃO DE MAPAS DE dh/dt")
print("Compatível com: Script 12 v3 (interpgaus)")
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
    bbox = THWAITES_BBOX
except (NameError, KeyError):
    region_name = 'Amundsen Sea Embayment'
    bbox = None

maps_dir = FIGURES_DIR / 'maps'
maps_dir.mkdir(exist_ok=True, parents=True)

# ============================================
# FUNÇÃO: LER HDF5 ROBUSTO
# ============================================

def read_h5_robust(filepath):
    """Lê datasets de HDF5, pula grupos e attrs."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            try:
                if isinstance(f[key], h5py.Group):
                    continue
                data[key] = f[key][:]
            except:
                pass
    return data


def get_var(data, *keys):
    """Busca variável por múltiplos nomes."""
    for key in keys:
        if key in data and isinstance(data[key], np.ndarray):
            return data[key]
    return None

# ============================================
# PALETA CUSTOMIZADA: Vermelho → Cinza claro → Azul
# ============================================

def make_rdgbu_cmap(name='RdGrayBu'):
    """
    Cria colormap: Vermelho escuro → Vermelho → Cinza claro → Azul → Azul escuro
    Centro (zero) = cinza claro (#D9D9D9) em vez de branco.
    """
    colors = [
        (0.00, '#67001F'),   # vermelho muito escuro (extremo negativo)
        (0.15, '#B2182B'),   # vermelho escuro
        (0.30, '#D6604D'),   # vermelho médio
        (0.42, '#F4A582'),   # salmão claro
        (0.50, '#D9D9D9'),   # CINZA CLARO (centro / zero)
        (0.58, '#92C5DE'),   # azul claro
        (0.70, '#4393C3'),   # azul médio
        (0.85, '#2166AC'),   # azul escuro
        (1.00, '#053061'),   # azul muito escuro (extremo positivo)
    ]
    positions = [c[0] for c in colors]
    hex_colors = [c[1] for c in colors]

    # Converter hex para RGB
    rgb_colors = [mcolors.hex2color(h) for h in hex_colors]

    return LinearSegmentedColormap.from_list(name, list(zip(positions, rgb_colors)), N=256)


# Criar colormaps
cmap_dhdt = make_rdgbu_cmap('RdGrayBu')
cmap_accel = make_rdgbu_cmap('RdGrayBu_accel')

# ============================================
# CARREGAR GRADE (Script 12 — interpgaus)
# ============================================

print("\n1. Carregando grade interpolada...")

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

if grid_file is None:
    print(f"✗ Grade não encontrada em: {grid_dir}")
    sys.exit(1)

print(f"   Arquivo: {grid_file.name}")
gd = read_h5_robust(grid_file)

# Resolver variáveis
grid_dhdt = get_var(gd, 'Z_pred', 'dhdt_smooth', 'dhdt_filled', 'dhdt')
grid_rmse = get_var(gd, 'Z_rmse', 'dhdt_sigma', 'dhdt_std')
grid_nobs = get_var(gd, 'Z_nobs', 'n_obs_per_cell', 'n_points')
grid_p2 = get_var(gd, 'Z_accel')
grid_rmse_fit = get_var(gd, 'Z_rmse_fit')
grid_lon = get_var(gd, 'lon')
grid_lat = get_var(gd, 'lat')

if grid_dhdt is None:
    print("✗ Variável de dh/dt não encontrada!")
    sys.exit(1)

if grid_rmse is None:
    grid_rmse = np.full_like(grid_dhdt, np.nan)
if grid_nobs is None:
    grid_nobs = np.full_like(grid_dhdt, np.nan)

has_p2 = grid_p2 is not None
has_rmse_fit = grid_rmse_fit is not None

print(f"   Shape: {grid_dhdt.shape}")
print(f"   Aceleração (p2): {'✓' if has_p2 else '✗'}")
print(f"   RMSE fit: {'✓' if has_rmse_fit else '✗'}")

del gd

# ============================================
# CARTOPY (OPCIONAL)
# ============================================

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("  Cartopy disponível")
except ImportError:
    HAS_CARTOPY = False
    print("  Cartopy não disponível — mapas sem projeção polar")

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def make_ax(fig, pos=111, extent=None):
    """Cria eixo com projeção polar ou simples."""
    if HAS_CARTOPY:
        proj = ccrs.SouthPolarStereo()
        ax = fig.add_subplot(pos, projection=proj)
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='black', linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        gl = ax.gridlines(draw_labels=True, linestyle='--',
                          color='gray', alpha=0.4, linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        return ax, ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(pos)
        lat_c = np.nanmean(grid_lat)
        ax.set_aspect(1 / np.cos(np.radians(lat_c)))
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3, linestyle='--')
        return ax, None


def plot_map(ax, data, tr, **kwargs):
    """pcolormesh compatível com/sem Cartopy."""
    if tr:
        return ax.pcolormesh(grid_lon, grid_lat, data, transform=tr,
                              shading='auto', rasterized=True, **kwargs)
    else:
        return ax.pcolormesh(grid_lon, grid_lat, data,
                              shading='auto', rasterized=True, **kwargs)


# Extensão
if bbox:
    extent = [bbox['lon_min'] - 2, bbox['lon_max'] + 2,
              bbox['lat_min'] - 1, bbox['lat_max'] + 1]
else:
    extent = [np.nanmin(grid_lon) - 1, np.nanmax(grid_lon) + 1,
              np.nanmin(grid_lat) - 1, np.nanmax(grid_lat) + 1]

# Norm padrão
norm_dhdt = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

# ============================================
# MAPA 1: dh/dt PRINCIPAL
# ============================================

print("\n2. Mapa 1: dh/dt principal...")

fig = plt.figure(figsize=(14, 12))
ax, tr = make_ax(fig, 111, extent)

im = plot_map(ax, grid_dhdt, tr, cmap=cmap_dhdt, norm=norm_dhdt)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='dh/dt (m/ano)', extend='both')
cbar.ax.tick_params(labelsize=10)

try:
    levels = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
    if tr:
        cs = ax.contour(grid_lon, grid_lat, grid_dhdt, levels=levels,
                        transform=tr, colors='black', linewidths=0.5, alpha=0.3)
    else:
        cs = ax.contour(grid_lon, grid_lat, grid_dhdt, levels=levels,
                        colors='black', linewidths=0.5, alpha=0.3)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
except Exception:
    pass

ax.set_title(f'{region_name} — Taxa de Mudança de Elevação (dh/dt)\n'
             f'Inverno Austral (JJA) — ICESat-2 ATL06\n'
             f'Vermelho = afinamento | Cinza = estável | Azul = espessamento',
             fontsize=13, fontweight='bold')

plt.tight_layout()
f1 = maps_dir / 'map_dhdt.png'
plt.savefig(f1, dpi=300, bbox_inches='tight')
print(f"   ✓ {f1.name}")
plt.close()

# ============================================
# MAPA 2: RMSE (Erro de Predição)
# ============================================

print("   Mapa 2: RMSE...")

if not np.all(np.isnan(grid_rmse)):
    fig = plt.figure(figsize=(14, 12))
    ax, tr = make_ax(fig, 111, extent)

    im = plot_map(ax, grid_rmse, tr, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label='RMSE (m/ano)')

    ax.set_title(f'{region_name} — Erro de Predição (RMSE)\n'
                 f'interpgaus — RSS(variabilidade + erro sistemático)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    f2 = maps_dir / 'map_rmse.png'
    plt.savefig(f2, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f2.name}")
    plt.close()
else:
    print("   RMSE não disponível")
    f2 = None

# ============================================
# MAPA 3: COBERTURA (N observações)
# ============================================

print("   Mapa 3: Cobertura...")

nobs_plot = np.where(grid_nobs > 0, grid_nobs, np.nan)

if not np.all(np.isnan(nobs_plot)):
    fig = plt.figure(figsize=(14, 12))
    ax, tr = make_ax(fig, 111, extent)

    im = plot_map(ax, nobs_plot, tr, cmap='viridis')
    plt.colorbar(im, ax=ax, shrink=0.8, label='N° observações usadas')

    ax.set_title(f'{region_name} — Cobertura de Dados\n'
                 f'Observações usadas por nó (4 quadrantes)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    f3 = maps_dir / 'map_coverage.png'
    plt.savefig(f3, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f3.name}")
    plt.close()
else:
    print("   Cobertura não disponível")
    f3 = None

# ============================================
# MAPA 4: CLASSIFICAÇÃO DE REGIMES
# ============================================

print("   Mapa 4: Classificação...")

fig = plt.figure(figsize=(14, 12))
ax, tr = make_ax(fig, 111, extent)

classification = np.full_like(grid_dhdt, np.nan)
classification[grid_dhdt < -1.5] = 1
classification[(grid_dhdt >= -1.5) & (grid_dhdt < -0.5)] = 2
classification[(grid_dhdt >= -0.5) & (grid_dhdt < -0.1)] = 3
classification[(grid_dhdt >= -0.1) & (grid_dhdt <= 0.1)] = 4
classification[(grid_dhdt > 0.1) & (grid_dhdt <= 0.5)] = 5
classification[grid_dhdt > 0.5] = 6

# Vermelho (afinamento) → Cinza claro (estável) → Azul (espessamento)
cmap_class = mcolors.ListedColormap(
    ['#8B0000',   # 1: afinamento rápido — vermelho escuro
     '#E03030',   # 2: afinamento moderado — vermelho
     '#F4A582',   # 3: afinamento lento — salmão claro
     '#D9D9D9',   # 4: estável — CINZA CLARO
     '#92C5DE',   # 5: espessamento lento — azul claro
     '#2166AC'])  # 6: espessamento forte — azul escuro
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm_class = mcolors.BoundaryNorm(bounds, cmap_class.N)

im = plot_map(ax, classification, tr, cmap=cmap_class, norm=norm_class)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[1, 2, 3, 4, 5, 6])
cbar.ax.set_yticklabels([
    'Afin. rápido\n(< -1.5)',
    'Afin. moderado\n(-1.5 a -0.5)',
    'Afin. lento\n(-0.5 a -0.1)',
    'Estável\n(±0.1)',
    'Esp. lento\n(0.1 a 0.5)',
    'Esp. forte\n(> 0.5)'
], fontsize=9)

valid = grid_dhdt[~np.isnan(grid_dhdt)].ravel()
thin_pct = 100 * np.sum(valid < 0) / len(valid)
ax.set_title(f'{region_name} — Classificação de Regimes\n'
             f'Adelgaçamento: {thin_pct:.1f}% da área | '
             f'Vermelho = afinamento | Cinza = estável | Azul = espessamento',
             fontsize=13, fontweight='bold')

plt.tight_layout()
f4 = maps_dir / 'map_classification.png'
plt.savefig(f4, dpi=300, bbox_inches='tight')
print(f"   ✓ {f4.name}")
plt.close()

# ============================================
# MAPA 5: ACELERAÇÃO (d²h/dt²)
# ============================================

f5_accel = None

if has_p2:
    print("   Mapa 5: Aceleração (p2)...")

    fig = plt.figure(figsize=(14, 12))
    ax, tr = make_ax(fig, 111, extent)

    norm_p2 = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = plot_map(ax, grid_p2, tr, cmap=cmap_accel, norm=norm_p2)
    plt.colorbar(im, ax=ax, shrink=0.8, label='d²h/dt² (m/ano²)', extend='both')

    try:
        levels_p2 = [-0.3, -0.1, 0, 0.1, 0.3]
        if tr:
            cs = ax.contour(grid_lon, grid_lat, grid_p2, levels=levels_p2,
                            transform=tr, colors='black', linewidths=0.5, alpha=0.3)
        else:
            cs = ax.contour(grid_lon, grid_lat, grid_p2, levels=levels_p2,
                            colors='black', linewidths=0.5, alpha=0.3)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    except Exception:
        pass

    p2_valid = grid_p2[~np.isnan(grid_p2)].ravel()
    ax.set_title(f'{region_name} — Aceleração (d²h/dt²)\n'
                 f'Média: {np.mean(p2_valid):+.4f} m/ano² | '
                 f'Vermelho = acelerando perda | Azul = desacelerando',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    f5_accel = maps_dir / 'map_acceleration.png'
    plt.savefig(f5_accel, dpi=300, bbox_inches='tight')
    print(f"   ✓ {f5_accel.name}")
    plt.close()
else:
    print("   Aceleração (p2) não disponível")

# ============================================
# MAPA 6: Δh ACUMULADO
# ============================================

print("   Mapa 6: Δh acumulado...")

mean_tspan = 4.5

try:
    dhdt_dir = RESULTS_DIR / 'dhdt_winter'
    for name in ['amundsen_sea_dhdt_winter_joined.h5']:
        jf = dhdt_dir / name
        if jf.exists():
            with h5py.File(jf, 'r') as f:
                for ts_key in ['tspan', 'time_span']:
                    if ts_key in f:
                        ts = f[ts_key][:]
                        ts_valid = ts[ts > 0]
                        if len(ts_valid) > 0:
                            mean_tspan = float(np.nanmean(ts_valid))
                        break
            break
except Exception:
    pass

print(f"   Time span médio: {mean_tspan:.2f} anos")

delta_h = grid_dhdt * mean_tspan

fig = plt.figure(figsize=(14, 12))
ax, tr = make_ax(fig, 111, extent)

norm_dh = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=5)
im = plot_map(ax, delta_h, tr, cmap=cmap_dhdt, norm=norm_dh)
plt.colorbar(im, ax=ax, shrink=0.8, label='Δh (metros)', extend='both')

ax.set_title(f'{region_name} — Mudança de Elevação Acumulada\n'
             f'Período: ~{mean_tspan:.1f} anos — ICESat-2\n'
             f'Vermelho = perda de gelo | Cinza = estável | Azul = ganho',
             fontsize=13, fontweight='bold')

plt.tight_layout()
f6 = maps_dir / 'map_delta_h.png'
plt.savefig(f6, dpi=300, bbox_inches='tight')
print(f"   ✓ {f6.name}")
plt.close()

# ============================================
# MAPA 7: PAINEL RESUMO (4-em-1)
# ============================================

print("   Mapa 7: Painel resumo (4-em-1)...")

fig = plt.figure(figsize=(20, 18))

# 7a: dh/dt
ax1, tr = make_ax(fig, 221, extent)
im1 = plot_map(ax1, grid_dhdt, tr, cmap=cmap_dhdt, norm=norm_dhdt)
plt.colorbar(im1, ax=ax1, shrink=0.7, label='dh/dt (m/ano)', extend='both')
ax1.set_title('(a) dh/dt\nVermelho = afinamento | Cinza = estável | Azul = espessamento',
              fontsize=10, fontweight='bold')

# 7b: Aceleração ou RMSE
ax2, tr = make_ax(fig, 222, extent)
if has_p2:
    norm_p2 = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im2 = plot_map(ax2, grid_p2, tr, cmap=cmap_accel, norm=norm_p2)
    plt.colorbar(im2, ax=ax2, shrink=0.7, label='d²h/dt² (m/ano²)', extend='both')
    ax2.set_title('(b) Aceleração (d²h/dt²)', fontsize=11, fontweight='bold')
else:
    im2 = plot_map(ax2, grid_rmse, tr, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, shrink=0.7, label='RMSE (m/ano)')
    ax2.set_title('(b) RMSE', fontsize=11, fontweight='bold')

# 7c: Cobertura
ax3, tr = make_ax(fig, 223, extent)
im3 = plot_map(ax3, nobs_plot, tr, cmap='viridis')
plt.colorbar(im3, ax=ax3, shrink=0.7, label='N° observações')
ax3.set_title('(c) Cobertura', fontsize=11, fontweight='bold')

# 7d: Classificação
ax4, tr = make_ax(fig, 224, extent)
im4 = plot_map(ax4, classification, tr, cmap=cmap_class, norm=norm_class)
cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.7, ticks=[1, 2, 3, 4, 5, 6])
cbar4.ax.set_yticklabels([
    '< -1.5', '-1.5 a -0.5', '-0.5 a -0.1',
    '±0.1', '0.1 a 0.5', '> 0.5'
], fontsize=8)
ax4.set_title('(d) Classificação', fontsize=11, fontweight='bold')

fig.suptitle(f'{region_name} — Resumo de dh/dt\n'
             f'Inverno Austral (JJA) — ICESat-2 ATL06 — interpgaus\n'
             f'Vermelho = afinamento | Cinza = estável | Azul = espessamento',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

f7 = maps_dir / 'panel_summary.png'
plt.savefig(f7, dpi=300, bbox_inches='tight')
print(f"   ✓ {f7.name}")
plt.close()

# ============================================
# RESUMO
# ============================================

print("\n" + "=" * 70)
print("MAPAS CRIADOS")
print("=" * 70)

all_maps = [
    ('map_dhdt.png', 'dh/dt principal'),
    ('map_rmse.png', 'Erro de predição (RMSE)'),
    ('map_coverage.png', 'Cobertura de dados'),
    ('map_classification.png', 'Classificação de regimes'),
    ('map_acceleration.png', 'Aceleração (d²h/dt²)'),
    ('map_delta_h.png', 'Δh acumulado'),
    ('panel_summary.png', 'Painel resumo (4-em-1)'),
]

for fname, desc in all_maps:
    f = maps_dir / fname
    if f.exists():
        print(f"   ✓ {fname:35s} — {desc}")

print(f"\nDiretório: {maps_dir}")
print("=" * 70)
print("\n✓ Mapas concluídos!")
print("\nPróximo passo: 15_plot_timeseries.py")
