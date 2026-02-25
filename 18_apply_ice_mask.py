"""
Aplicação de Máscara de Gelo para filtrar oceano
Remove pontos no oceano, mantém apenas gelo terrestre e plataforma de gelo

Baseado em: BedMachine Antarctica
Referência: Morlighem et al. (2020) - https://doi.org/10.1038/s41561-019-0510-8

MÁSCARA DE GELO (BedMachine):
    0 = Oceano (ocean)
    1 = Gelo terrestre (grounded ice)
    2 = Gelo flutuante/plataforma (floating ice / ice shelf)
    3 = Ilha rochosa (rock outcrop)
    4 = Lago subglacial (subglacial lake)
"""

import numpy as np
import sys
from pathlib import Path

# Importar configurações
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

# Importar utilitários
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts\utils")
from io_utils import read_hdf5, write_hdf5
from geodetic_utils import lonlat_to_xy

print("=" * 70)
print("APLICAÇÃO DE MÁSCARA DE GELO (SOMENTE BEDMACHINE)")
print("Remove oceano, mantém gelo terrestre e plataforma de gelo")
print("=" * 70)

# ============================================
# BEDMACHINE (OBRIGATÓRIO)
# ============================================

BEDMACHINE_FILE = DATA_DIR / "bedmachine" / "BedMachineAntarctica-v3.nc"

if not BEDMACHINE_FILE.exists():
    print("\n✗ BedMachine NÃO encontrado!")
    print(f"  Procurado em: {BEDMACHINE_FILE}")
    print("  Este script requer BedMachine. Instale/baixe o arquivo e tente novamente.")
    sys.exit(1)

print("\n✓ BedMachine encontrado. Usando máscara precisa...")

try:
    import netCDF4 as nc
    from scipy.interpolate import RegularGridInterpolator
except Exception as e:
    print("\n✗ Dependências faltando para usar BedMachine (netCDF4/scipy).")
    print(f"  Erro: {e}")
    sys.exit(1)

try:
    # Ler BedMachine
    print("  Carregando BedMachine...")
    ds = nc.Dataset(BEDMACHINE_FILE, "r")

    mask_bed = ds["mask"][:]  # 0=ocean, 1=grounded, 2=floating, 3=rock, 4=lake
    x_bed = ds["x"][:]        # coordenadas polares estereográficas
    y_bed = ds["y"][:]

    print(f"  BedMachine: {mask_bed.shape}")
    print(f"  Resolução: {x_bed[1] - x_bed[0]:.0f} m")

    # Criar interpolador
    print("  Criando interpolador...")
    interp_mask = RegularGridInterpolator(
        (y_bed, x_bed),
        mask_bed.astype(float),
        method="nearest",      # Nearest para máscara categórica
        bounds_error=False,
        fill_value=0,          # Oceano por padrão
    )

    ds.close()

except Exception as e:
    print("\n✗ Erro ao ler/interpolar BedMachine.")
    print(f"  Erro: {e}")
    try:
        ds.close()
    except Exception:
        pass
    sys.exit(1)

# ============================================
# PROCESSAR ARQUIVO MESCLADO
# ============================================

merged_file = RESULTS_DIR / "dhdt_winter" / "thwaites_dhdt_winter_merged.h5"

print(f"\n1. Processando arquivo mesclado...")
print(f"   Arquivo: {merged_file.name}")

if not merged_file.exists():
    print("\n✗ Arquivo mesclado não encontrado!")
    print("  Execute primeiro: 11_join_tiles.py")
    sys.exit(1)

data = read_hdf5(merged_file)

lon = data["longitude"]
lat = data["latitude"]
dhdt = data["dhdt"]

n_total = len(lon)
print(f"   Pontos totais: {n_total:,}")

# ============================================
# APLICAR MÁSCARA (BEDMACHINE)
# ============================================

print("\n2. Aplicando máscara de gelo (BedMachine Antarctica)...")

# Converter lon/lat para coordenadas polares (x, y)
x, y = lonlat_to_xy(lon, lat)

# Interpolar máscara nos pontos de dados
print("   Interpolando máscara (isso pode demorar ~30s)...")
points = np.column_stack([y, x])
mask_values = interp_mask(points)

# Manter: gelo terrestre (1) + gelo flutuante (2)
ice_mask = (mask_values >= 1) & (mask_values <= 2)

print(f"\n   Tipos de superfície encontrados:")
unique, counts = np.unique(mask_values.astype(int), return_counts=True)
labels = {
    0: "Oceano",
    1: "Gelo terrestre",
    2: "Gelo flutuante",
    3: "Rocha",
    4: "Lago subglacial",
}
for val, count in zip(unique, counts):
    label = labels.get(val, f"Desconhecido ({val})")
    pct = 100 * count / n_total
    print(f"      {label}: {count:,} ({pct:.1f}%)")

# ============================================
# ESTATÍSTICAS DA FILTRAGEM
# ============================================

n_ice = int(np.sum(ice_mask))
n_removed = n_total - n_ice
pct_kept = 100 * n_ice / n_total
pct_removed = 100 * n_removed / n_total

print(f"\n3. Resultado da filtragem:")
print(f"   Pontos mantidos (gelo): {n_ice:,} ({pct_kept:.1f}%)")
print(f"   Pontos removidos (oceano/outros): {n_removed:,} ({pct_removed:.1f}%)")

dhdt_before = dhdt[~np.isnan(dhdt)]
dhdt_after = dhdt[ice_mask & ~np.isnan(dhdt)]

print(f"\n4. Impacto nas estatísticas de dh/dt:")
print(f"\n   ANTES da máscara:")
print(f"      Média: {np.mean(dhdt_before):.4f} m/ano")
print(f"      Mediana: {np.median(dhdt_before):.4f} m/ano")
print(f"      Desvio padrão: {np.std(dhdt_before):.4f} m/ano")

print(f"\n   DEPOIS da máscara:")
print(f"      Média: {np.mean(dhdt_after):.4f} m/ano")
print(f"      Mediana: {np.median(dhdt_after):.4f} m/ano")
print(f"      Desvio padrão: {np.std(dhdt_after):.4f} m/ano")

delta_mean = np.mean(dhdt_after) - np.mean(dhdt_before)
delta_median = np.median(dhdt_after) - np.median(dhdt_before)

print(f"\n   MUDANÇA:")
print(f"      Δ Média: {delta_mean:.4f} m/ano ({delta_mean/np.abs(np.mean(dhdt_before))*100:+.1f}%)")
print(f"      Δ Mediana: {delta_median:.4f} m/ano")

# ============================================
# SALVAR DADOS FILTRADOS
# ============================================

print("\n5. Salvando dados filtrados...")

filtered_data = {}
for key, value in data.items():
    if isinstance(value, np.ndarray) and len(value) == n_total:
        filtered_data[key] = value[ice_mask]
    else:
        filtered_data[key] = value

filtered_data["n_points_original"] = n_total
filtered_data["n_points_filtered"] = n_ice
filtered_data["mask_type"] = "BedMachine"

output_file = RESULTS_DIR / "dhdt_winter" / "thwaites_dhdt_winter_ice_only.h5"
write_hdf5(output_file, filtered_data)

print(f"   ✓ Arquivo salvo: {output_file.name}")
print(f"   Tamanho: {output_file.stat().st_size / (1024**2):.1f} MB")

# ============================================
# CRIAR MAPA COMPARATIVO
# ============================================

print("\n6. Criando mapa comparativo...")

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(
        lon, lat, c=dhdt, cmap="RdBu_r", vmin=-2, vmax=2,
        s=0.5, alpha=0.5, rasterized=True
    )
    ax1.set_title(f"ANTES da Máscara\n{n_total:,} pontos", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_aspect("equal")
    plt.colorbar(sc1, ax=ax1, label="dh/dt (m/ano)")

    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(
        lon[ice_mask], lat[ice_mask], c=dhdt[ice_mask], cmap="RdBu_r",
        vmin=-2, vmax=2, s=0.5, alpha=0.5, rasterized=True
    )
    ax2.set_title(f"DEPOIS da Máscara\n{n_ice:,} pontos", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("Latitude (°)")
    ax2.set_aspect("equal")
    plt.colorbar(sc2, ax=ax2, label="dh/dt (m/ano)")

    ax3 = fig.add_subplot(gs[0, 2])
    removed_mask = ~ice_mask
    ax3.scatter(
        lon[removed_mask], lat[removed_mask],
        c="red", s=1, alpha=0.3, rasterized=True, label="Removido"
    )
    ax3.scatter(
        lon[ice_mask], lat[ice_mask],
        c="blue", s=0.3, alpha=0.2, rasterized=True, label="Mantido"
    )
    ax3.set_title(f"Pontos Removidos (vermelho)\n{n_removed:,} pontos", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Longitude (°)")
    ax3.set_ylabel("Latitude (°)")
    ax3.set_aspect("equal")
    ax3.legend(loc="upper right")

    fig.suptitle("Comparação: Antes vs Depois da Máscara de Gelo", fontsize=16, fontweight="bold")

    comparison_file = FIGURES_DIR / "maps" / "ice_mask_comparison.png"
    plt.savefig(comparison_file, dpi=200, bbox_inches="tight")
    print(f"   ✓ Mapa comparativo: {comparison_file.name}")
    plt.close()

except Exception as e:
    print(f"   ⚠ Erro ao criar mapa: {e}")

# ============================================
# RESUMO FINAL
# ============================================

print("\n" + "=" * 70)
print("RESUMO DA APLICAÇÃO DE MÁSCARA")
print("=" * 70)
print("\nMétodo: BedMachine Antarctica")
print(f"\nDados:")
print(f"  Original: {n_total:,} pontos")
print(f"  Filtrado: {n_ice:,} pontos ({pct_kept:.1f}%)")
print(f"  Removido: {n_removed:,} pontos ({pct_removed:.1f}%)")

print(f"\nEstatísticas de dh/dt:")
print(f"  Média: {np.mean(dhdt_before):.3f} → {np.mean(dhdt_after):.3f} m/ano ({delta_mean:+.3f})")
print(f"  Mediana: {np.median(dhdt_before):.3f} → {np.median(dhdt_after):.3f} m/ano ({delta_median:+.3f})")

print(f"\nArquivo de saída:")
print(f"  {output_file}")
print("\n" + "=" * 70)
print("\n✓ Máscara de gelo aplicada com sucesso!")
