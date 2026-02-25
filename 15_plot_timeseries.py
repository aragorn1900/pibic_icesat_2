"""
Análise de séries temporais e gráficos complementares

- Histograma/Boxplot: usa produto final de dh/dt (results/dhdt_winter/*.h5)
- Série temporal de anomalia de elevação: usa observações ATL06 (Data/*atl06*.h5)
  Leitura em chunks (sem loop por célula)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import h5py
import gc

# Importar configurações
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

# Importar utilitários
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts\utils")
from io_utils import read_hdf5

print("=" * 60)
print("ANÁLISE DE SÉRIES TEMPORAIS E GRÁFICOS")
print("=" * 60)

# ============================================
# RESOLVER DIRETÓRIOS
# ============================================

BASE_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")

try:
    _results_dir = RESULTS_DIR
except NameError:
    _results_dir = BASE_DIR / "results"
RESULTS_DIR = _results_dir

try:
    _figures_dir = FIGURES_DIR
except NameError:
    _figures_dir = BASE_DIR / "figures"
FIGURES_DIR = _figures_dir

try:
    _data_dir = DATA_DIR
except NameError:
    _data_dir = BASE_DIR / "Data"
DATA_DIR = Path(_data_dir)

try:
    region_name = THWAITES_BBOX["name"]
except (NameError, KeyError):
    region_name = "Amundsen Sea Embayment"

try:
    grid_res = GRID_RESOLUTION
except NameError:
    grid_res = 2000

plots_dir = FIGURES_DIR / "plots"
plots_dir.mkdir(exist_ok=True, parents=True)

# ============================================
# PARTE A: DHDT FINAL (HISTOGRAMA + BOXPLOT)
# ============================================

print("\n1. Carregando dh/dt final (para histograma/boxplot)...")

dhdt_dir = RESULTS_DIR / "dhdt_winter"
merged_file = None

if dhdt_dir.exists():
    for name in [
        "amundsen_sea_dhdt_winter_joined.h5",
        "amundsen_sea_dhdt_winter_gridded.h5",
        "thwaites_dhdt_winter_ice_only.h5",
        "thwaites_dhdt_winter_merged.h5",
    ]:
        candidate = dhdt_dir / name
        if candidate.exists():
            merged_file = candidate
            break
    if merged_file is None:
        h5_files = [f for f in dhdt_dir.glob("*.h5") if "tile_" not in f.name]
        if h5_files:
            merged_file = sorted(
                h5_files, key=lambda f: f.stat().st_mtime, reverse=True
            )[0]

if merged_file is None:
    print(f"\n✗ Nenhum arquivo de dh/dt encontrado em: {dhdt_dir}")
    sys.exit(1)

print(f"   Arquivo dh/dt: {merged_file.name}")

data_dhdt = read_hdf5(merged_file)
dhdt = data_dhdt["dhdt"]
valid = ~np.isnan(dhdt)
dhdt = dhdt[valid]

print(f"   ✓ dh/dt carregado: {len(dhdt):,} pontos válidos")

# ============================================
# GRÁFICO 1: HISTOGRAMA
# ============================================

print("\n2. Criando histograma de dh/dt...")

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.arange(-3, 2, 0.1)
ax.hist(dhdt, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero (estável)")
ax.axvline(np.mean(dhdt), color="orange", linestyle="--", linewidth=2,
           label=f"Média ({np.mean(dhdt):.3f} m/ano)")
ax.axvline(np.median(dhdt), color="green", linestyle="--", linewidth=2,
           label=f"Mediana ({np.median(dhdt):.3f} m/ano)")
ax.set_xlabel("dh/dt (m/ano)", fontsize=12)
ax.set_ylabel("Frequência", fontsize=12)
ax.set_title(f"Distribuição de dh/dt - {region_name} (Inverno)",
             fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
hist_file = plots_dir / "dhdt_histogram.png"
plt.savefig(hist_file, dpi=300, bbox_inches="tight")
print(f"   ✓ Histograma salvo: {hist_file}")
plt.close()

# ============================================
# GRÁFICO 2: BOX PLOT
# ============================================

print("\n3. Criando box plot...")

fig, ax = plt.subplots(figsize=(8, 10))
bp = ax.boxplot([dhdt], vert=True, patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor("lightblue")
bp["medians"][0].set_color("red")
bp["medians"][0].set_linewidth(2)
ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_ylabel("dh/dt (m/ano)", fontsize=12)
ax.set_title(f"Distribuição de dh/dt - {region_name}\n(Inverno Austral)",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
stats_text = (
    f"n = {len(dhdt):,}\n"
    f"Média = {np.mean(dhdt):.3f} m/ano\n"
    f"Mediana = {np.median(dhdt):.3f} m/ano\n"
    f"σ = {np.std(dhdt):.3f} m/ano"
)
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.tight_layout()
box_file = plots_dir / "dhdt_boxplot.png"
plt.savefig(box_file, dpi=300, bbox_inches="tight")
print(f"   ✓ Box plot salvo: {box_file}")
plt.close()

# ============================================
# PARTE B: SÉRIE TEMPORAL — ANOMALIA DE ELEVAÇÃO
#   100% vetorizado com numpy (sem dicionários por célula)
#
#   Estratégia:
#   1) Passada 1: descobrir extensão x,y e anos disponíveis
#   2) Criar arrays 2D: soma[cell_idx, year_idx] e count[cell_idx, year_idx]
#   3) Passada 2: acumular com np.add.at (vetorizado, sem loop)
#   4) Filtrar células com dados em todos os anos
#   5) Calcular anomalia corrigida
# ============================================

print("\n4. Série temporal de anomalia de elevação (vetorizado)...")

candidate_obs_files = [
    DATA_DIR / "amundsen_sea_atl06_winter_JJA.h5",
    DATA_DIR / "amundsen_sea_embayment_atl06_winter_JJA.h5",
    DATA_DIR / "amundsen_sea_atl06_merged.h5",
    DATA_DIR / "amundsen_sea_embayment_atl06_merged.h5",
]

obs_file = next((f for f in candidate_obs_files if f.exists()), None)
if obs_file is None:
    for pattern in ["*winter*.h5", "*merged*.h5"]:
        candidates = sorted(DATA_DIR.glob(pattern))
        if candidates:
            obs_file = candidates[-1]
            break

ts_file = None

if obs_file is None:
    print("   ⚠ Nenhum arquivo observacional encontrado em Data/.")

else:
    print(f"   Arquivo: {obs_file.name}")
    file_size_gb = obs_file.stat().st_size / (1024**3)
    print(f"   Tamanho: {file_size_gb:.2f} GB")

    try:
        sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts\utils")
        from geodetic_utils import lonlat_to_xy

        with h5py.File(obs_file, "r") as f:

            has_t_year = "t_year" in f
            has_h_li = "h_li" in f
            has_x = "x" in f and "y" in f
            has_lonlat = "longitude" in f and "latitude" in f

            if not has_t_year or not has_h_li:
                print("   ⚠ 't_year' ou 'h_li' não encontrados.")
            elif not has_x and not has_lonlat:
                print("   ⚠ Coordenadas não encontradas.")
            else:
                n_total = f["t_year"].shape[0]
                print(f"   Total de observações: {n_total:,}")
                print(f"   Resolução de binning: {grid_res/1000:.0f} km")

                CHUNK = 10_000_000
                n_chunks = (n_total + CHUNK - 1) // CHUNK

                # --------------------------------------------------
                # PASSADA 1 (rápida): Descobrir extensão e anos
                # --------------------------------------------------
                print("   Passada 1: descobrindo extensão e anos...")

                x_min_g = np.inf
                x_max_g = -np.inf
                y_min_g = np.inf
                y_max_g = -np.inf
                all_years = set()

                for i_c in range(n_chunks):
                    s = i_c * CHUNK
                    e = min(s + CHUNK, n_total)

                    t_chunk = f["t_year"][s:e]
                    ok = ~np.isnan(t_chunk)
                    yrs = np.floor(t_chunk[ok]).astype(np.int32)
                    all_years.update(np.unique(yrs).tolist())

                    if has_x:
                        xc = f["x"][s:e]
                        yc = f["y"][s:e]
                    else:
                        lon_c = f["longitude"][s:e]
                        lat_c = f["latitude"][s:e]
                        xc, yc = lonlat_to_xy(lon_c, lat_c)
                        del lon_c, lat_c

                    ok2 = ~np.isnan(xc)
                    x_min_g = min(x_min_g, float(np.min(xc[ok2])))
                    x_max_g = max(x_max_g, float(np.max(xc[ok2])))
                    y_min_g = min(y_min_g, float(np.min(yc[ok2])))
                    y_max_g = max(y_max_g, float(np.max(yc[ok2])))

                    del t_chunk, xc, yc

                all_years = sorted(all_years)
                n_years = len(all_years)
                year_to_idx = {yr: i for i, yr in enumerate(all_years)}

                x_origin = np.floor(x_min_g / grid_res) * grid_res
                y_origin = np.floor(y_min_g / grid_res) * grid_res
                nx = int(np.ceil((x_max_g - x_origin) / grid_res)) + 1
                ny = int(np.ceil((y_max_g - y_origin) / grid_res)) + 1
                n_cells = nx * ny

                print(f"   Anos: {all_years}")
                print(f"   Grade: {nx} × {ny} = {n_cells:,} células")
                print(f"   Origem: x={x_origin/1000:.0f} km, y={y_origin/1000:.0f} km")

                # Memória necessária: 2 arrays (soma, count) × n_cells × n_years × 8 bytes
                mem_mb = 2 * n_cells * n_years * 8 / (1024**2)
                print(f"   RAM para acumuladores: {mem_mb:.0f} MB")

                if mem_mb > 4000:
                    print(f"   ⚠ Usando grade mais grossa (5 km) para caber na RAM...")
                    grid_res_ts = 5000
                    nx = int(np.ceil((x_max_g - x_origin) / grid_res_ts)) + 1
                    ny = int(np.ceil((y_max_g - y_origin) / grid_res_ts)) + 1
                    n_cells = nx * ny
                    mem_mb = 2 * n_cells * n_years * 8 / (1024**2)
                    print(f"   Grade ajustada: {nx} × {ny} = {n_cells:,} ({mem_mb:.0f} MB)")
                else:
                    grid_res_ts = grid_res

                # --------------------------------------------------
                # Criar acumuladores: soma e count por (célula, ano)
                # --------------------------------------------------
                cell_sum = np.zeros((n_cells, n_years), dtype=np.float64)
                cell_count = np.zeros((n_cells, n_years), dtype=np.int64)

                # --------------------------------------------------
                # PASSADA 2: Acumular (100% vetorizado)
                # --------------------------------------------------
                print("   Passada 2: acumulando h_li por célula por ano (vetorizado)...")

                for i_c in range(n_chunks):
                    s = i_c * CHUNK
                    e = min(s + CHUNK, n_total)

                    t_chunk = f["t_year"][s:e]
                    h_chunk = f["h_li"][s:e]

                    if has_x:
                        xc = f["x"][s:e]
                        yc = f["y"][s:e]
                    else:
                        lon_c = f["longitude"][s:e]
                        lat_c = f["latitude"][s:e]
                        xc, yc = lonlat_to_xy(lon_c, lat_c)
                        del lon_c, lat_c

                    # Filtrar válidos
                    ok = (~np.isnan(h_chunk) & ~np.isnan(t_chunk)
                          & ~np.isnan(xc) & ~np.isnan(yc))
                    t_ok = t_chunk[ok]
                    h_ok = h_chunk[ok]
                    x_ok = xc[ok]
                    y_ok = yc[ok]
                    del t_chunk, h_chunk, xc, yc

                    # Índices de célula
                    ci = ((x_ok - x_origin) / grid_res_ts).astype(np.int32)
                    cj = ((y_ok - y_origin) / grid_res_ts).astype(np.int32)
                    np.clip(ci, 0, nx - 1, out=ci)
                    np.clip(cj, 0, ny - 1, out=cj)
                    cell_idx = ci * ny + cj

                    # Índice de ano
                    years_chunk = np.floor(t_ok).astype(np.int32)
                    year_idx = np.array([year_to_idx.get(int(y), -1) for y in years_chunk],
                                       dtype=np.int32)
                    valid_yr = year_idx >= 0

                    cell_idx = cell_idx[valid_yr]
                    year_idx = year_idx[valid_yr]
                    h_valid = h_ok[valid_yr]

                    del t_ok, x_ok, y_ok, ci, cj, years_chunk

                    # Acumular com np.add.at (vetorizado!)
                    np.add.at(cell_sum, (cell_idx, year_idx), h_valid)
                    np.add.at(cell_count, (cell_idx, year_idx), 1)

                    del cell_idx, year_idx, h_valid
                    gc.collect()

                    if (i_c + 1) % 3 == 0 or (i_c + 1) == n_chunks:
                        pct = 100 * (i_c + 1) / n_chunks
                        print(f"      Chunk {i_c+1}/{n_chunks} ({pct:.0f}%)")

                # --------------------------------------------------
                # Filtrar células com dados em TODOS os anos
                # --------------------------------------------------
                print("   Filtrando células com cobertura completa...")

                has_all_years = np.all(cell_count > 0, axis=1)
                n_complete = int(np.sum(has_all_years))
                print(f"   Células com TODOS os anos: {n_complete:,} / {n_cells:,}")

                if n_complete == 0 or n_years <= 1:
                    print("   ⚠ Dados insuficientes para série temporal.")
                else:
                    # Média por célula por ano (só células completas)
                    sum_complete = cell_sum[has_all_years]    # (n_complete, n_years)
                    cnt_complete = cell_count[has_all_years]  # (n_complete, n_years)
                    mean_complete = sum_complete / cnt_complete  # (n_complete, n_years)

                    # Liberar
                    del cell_sum, cell_count, sum_complete, cnt_complete
                    gc.collect()

                    # Estatísticas regionais por ano
                    sorted_years = np.array(all_years)
                    year_means = np.mean(mean_complete, axis=0)
                    year_stds = np.std(mean_complete, axis=0)
                    year_sems = year_stds / np.sqrt(n_complete)

                    # Anomalia
                    ref_mean = year_means[0]
                    anomalies = year_means - ref_mean

                    # Tendência linear
                    x_fit = sorted_years.astype(float)
                    coeffs = np.polyfit(x_fit, anomalies, 1)
                    trend_slope = coeffs[0]
                    trend_line = np.polyval(coeffs, x_fit)

                    # Tendência quadrática
                    if len(sorted_years) >= 4:
                        coeffs2 = np.polyfit(x_fit, anomalies, 2)
                        accel = 2 * coeffs2[0]
                        has_accel = True
                    else:
                        has_accel = False

                    # Bandas de incerteza
                    band_1sigma = year_sems
                    band_2sigma = 2 * year_sems

                    # Imprimir tabela
                    ref_year = sorted_years[0]
                    print(f"\n   Estatísticas por ano (corrigidas por cobertura):")
                    print(f"   {'Ano':>6s}  {'h_li médio':>12s}  {'Δh (m)':>10s}  {'±SEM':>8s}")
                    print(f"   {'-'*42}")
                    for i, yr in enumerate(sorted_years):
                        print(f"   {yr:6d}  {year_means[i]:12.2f}  {anomalies[i]:+10.3f}  {year_sems[i]:8.3f}")

                    print(f"\n   Células usadas: {n_complete:,}")
                    print(f"   Tendência linear: {trend_slope:+.3f} m/ano")
                    if has_accel:
                        print(f"   Aceleração: {accel:+.3f} m/ano²")
                    total_change = anomalies[-1]
                    total_yrs = sorted_years[-1] - sorted_years[0]
                    print(f"   Mudança total ({ref_year}-{sorted_years[-1]}): {total_change:+.3f} m em {total_yrs} anos")

                    # --------------------------------------------------
                    # GRÁFICO —
                    # --------------------------------------------------

                    fig, ax = plt.subplots(figsize=(12, 7))

                    # Banda 2σ
                    ax.fill_between(
                        sorted_years,
                        anomalies - band_2sigma,
                        anomalies + band_2sigma,
                        alpha=0.15, color="steelblue", label="Incerteza 2σ",
                    )

                    # Banda 1σ
                    ax.fill_between(
                        sorted_years,
                        anomalies - band_1sigma,
                        anomalies + band_1sigma,
                        alpha=0.30, color="steelblue", label="Incerteza 1σ",
                    )

                    # Linha de anomalia
                    ax.plot(
                        sorted_years, anomalies, "o-",
                        color="darkblue", linewidth=2.5, markersize=9,
                        zorder=5, label="Anomalia de elevação",
                    )

                    # Tendência linear
                    ax.plot(
                        sorted_years, trend_line, "--",
                        color="red", linewidth=2,
                        label=f"Tendência: {trend_slope:+.3f} m/ano",
                    )

                    # Tendência quadrática
                    if has_accel:
                        x_smooth = np.linspace(sorted_years[0], sorted_years[-1], 100)
                        y_smooth = np.polyval(coeffs2, x_smooth)
                        ax.plot(
                            x_smooth, y_smooth, "-",
                            color="darkred", linewidth=1.5, alpha=0.7,
                            label=f"Quadrático (acel: {accel:+.3f} m/ano²)",
                        )

                    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)

                    ax.set_xlabel("Ano", fontsize=14)
                    ax.set_ylabel("Anomalia de elevação Δh (m)", fontsize=14)
                    ax.set_title(
                        f"Mudança de Elevação — {region_name}\n"
                        f"Inverno Austral (JJA) · Referência: {ref_year} · "
                        f"{n_complete:,} células com cobertura completa",
                        fontsize=14, fontweight="bold",
                    )
                    ax.set_xticks(sorted_years)
                    ax.legend(loc="lower left", fontsize=11)
                    ax.grid(True, alpha=0.3)

                    stats_box = (
                        f"Tendência: {trend_slope:+.3f} m/ano\n"
                        f"Mudança total: {total_change:+.3f} m\n"
                        f"Período: {ref_year}–{sorted_years[-1]}\n"
                        f"Células: {n_complete:,}"
                    )
                    if has_accel:
                        stats_box += f"\nAceleração: {accel:+.3f} m/ano²"

                    ax.text(
                        0.98, 0.97, stats_box, transform=ax.transAxes,
                        fontsize=10, verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.5",
                                  facecolor="white", edgecolor="gray", alpha=0.9),
                    )

                    plt.tight_layout()

                    ts_file = plots_dir / "elevation_anomaly_timeseries.png"
                    plt.savefig(ts_file, dpi=300, bbox_inches="tight")
                    print(f"\n   ✓ Série temporal salva: {ts_file}")
                    plt.close()

                    del mean_complete

    except Exception as e:
        print(f"   ✗ Erro ao processar série temporal: {e}")
        import traceback
        traceback.print_exc()
        ts_file = None

# ============================================
# RESUMO FINAL
# ============================================

print("\n" + "=" * 60)
print("GRÁFICOS CRIADOS COM SUCESSO")
print("=" * 60)
print(f"Diretório: {plots_dir}")
print("\nArquivos:")
print(f"  1. {hist_file.name}")
print(f"  2. {box_file.name}")
if ts_file:
    print(f"  3. {Path(ts_file).name}")
print("=" * 60)

print("\n✓ PROJETO CONCLUÍDO!")
print(f"\nResultados finais em:")
print(f"  - Grades:   {RESULTS_DIR / 'grids'}")
print(f"  - Mapas:    {FIGURES_DIR / 'maps'}")

print(f"  - Gráficos: {plots_dir}")
