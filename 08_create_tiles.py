"""
Criação de tiles espaciais para processamento paralelo
Equivalente a: tile.py do CAPTOOLKIT
Versão 5.0 - Leitura 100% sequencial (sem fancy indexing)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import h5py
import gc
import shutil

# ============================================
# DETECTAR DIRETÓRIO AUTOMATICAMENTE
# ============================================

if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).parent
else:
    SCRIPT_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")

sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    THWAITES_BBOX,
    TILE_SIZE,
    DATA_DIR,
    TILES_WINTER_DIR,
    LOGS_DIR
)

sys.path.insert(0, str(SCRIPT_DIR / 'utils'))
from io_utils import write_hdf5
from geodetic_utils import lonlat_to_xy, xy_to_lonlat

print("=" * 70)
print("CRIAÇÃO DE TILES ESPACIAIS (v5.0 - LEITURA SEQUENCIAL)")
print("Otimizado para 8 GB RAM — Zero fancy indexing")
print("=" * 70)

# ============================================
# PARÂMETROS
# ============================================

CHUNK_SIZE = 20_000_000  # ~150 MB por array float64

print(f"\nParâmetros:")
print(f"  Tamanho do tile: {TILE_SIZE/1000:.0f} km × {TILE_SIZE/1000:.0f} km")
print(f"  Chunk de leitura: {CHUNK_SIZE:,} pontos")

# ============================================
# LOCALIZAR ARQUIVO
# ============================================

print(f"\n1. Localizando arquivo de inverno...")

winter_files = list(DATA_DIR.glob("*winter*.h5"))
if len(winter_files) == 0:
    print(f"\n✗ Nenhum arquivo de inverno encontrado em: {DATA_DIR}")
    sys.exit(1)

winter_file = sorted(winter_files)[-1]
file_size_gb = winter_file.stat().st_size / (1024**3)
print(f"   Arquivo: {winter_file.name}")
print(f"   Tamanho: {file_size_gb:.2f} GB")

# ============================================
# INSPECIONAR ARQUIVO
# ============================================

print(f"\n2. Inspecionando arquivo...")

with h5py.File(winter_file, 'r') as f:
    available_vars = list(f.keys())
    n_points = f['latitude'].shape[0]
    has_x = 'x' in available_vars
    has_y = 'y' in available_vars

    data_var_names = []
    meta_var_names = []
    for var in available_vars:
        ds = f[var]
        if ds.shape and len(ds.shape) >= 1 and ds.shape[0] == n_points:
            data_var_names.append(var)
        else:
            meta_var_names.append(var)

print(f"   Total de pontos: {n_points:,}")
print(f"   Variáveis de dados: {len(data_var_names)}")
print(f"   Variáveis de metadata: {len(meta_var_names)}")
print(f"   Tem x,y: {has_x and has_y}")

n_chunks = int(np.ceil(n_points / CHUNK_SIZE))
print(f"   Chunks necessários: {n_chunks}")

# ============================================
# PASSADA 1: DESCOBRIR EXTENSÃO + TILE KEYS
# ============================================

print(f"\n3. PASSADA 1 — Extensão espacial e tile keys...")

# 3a. Extensão
x_global_min = np.inf
x_global_max = -np.inf
y_global_min = np.inf
y_global_max = -np.inf

with h5py.File(winter_file, 'r') as f:
    for start in tqdm(range(0, n_points, CHUNK_SIZE), desc="   Extensão", total=n_chunks):
        end = min(start + CHUNK_SIZE, n_points)
        if has_x and has_y:
            xc = f['x'][start:end]
            yc = f['y'][start:end]
        else:
            lon_c = f['longitude'][start:end]
            lat_c = f['latitude'][start:end]
            xc, yc = lonlat_to_xy(lon_c, lat_c)
            del lon_c, lat_c
        x_global_min = min(x_global_min, float(np.min(xc)))
        x_global_max = max(x_global_max, float(np.max(xc)))
        y_global_min = min(y_global_min, float(np.min(yc)))
        y_global_max = max(y_global_max, float(np.max(yc)))
        del xc, yc

gc.collect()

print(f"   X: {x_global_min/1000:.1f} a {x_global_max/1000:.1f} km")
print(f"   Y: {y_global_min/1000:.1f} a {y_global_max/1000:.1f} km")

# Grade
x_min_tile = np.floor(x_global_min / TILE_SIZE) * TILE_SIZE
x_max_tile = np.ceil(x_global_max / TILE_SIZE) * TILE_SIZE
y_min_tile = np.floor(y_global_min / TILE_SIZE) * TILE_SIZE
y_max_tile = np.ceil(y_global_max / TILE_SIZE) * TILE_SIZE

x_edges = np.arange(x_min_tile, x_max_tile + TILE_SIZE, TILE_SIZE)
y_edges = np.arange(y_min_tile, y_max_tile + TILE_SIZE, TILE_SIZE)

n_tiles_x = len(x_edges) - 1
n_tiles_y = len(y_edges) - 1
n_tiles_total = n_tiles_x * n_tiles_y

print(f"   Grade: {n_tiles_x} × {n_tiles_y} = {n_tiles_total} tiles possíveis")

# 3b. Tile keys em memmap
print("   Computando tile_key por ponto...")

temp_dir = TILES_WINTER_DIR / '_temp'
temp_dir.mkdir(exist_ok=True, parents=True)

tile_keys_path = temp_dir / 'tile_keys.dat'
tile_keys_mmap = np.memmap(tile_keys_path, dtype=np.int32, mode='w+', shape=(n_points,))

# Com 180 tiles, int32 é mais que suficiente e usa metade da memória
with h5py.File(winter_file, 'r') as f:
    for start in tqdm(range(0, n_points, CHUNK_SIZE), desc="   Tile keys", total=n_chunks):
        end = min(start + CHUNK_SIZE, n_points)
        if has_x and has_y:
            xc = f['x'][start:end]
            yc = f['y'][start:end]
        else:
            lon_c = f['longitude'][start:end]
            lat_c = f['latitude'][start:end]
            xc, yc = lonlat_to_xy(lon_c, lat_c)
            del lon_c, lat_c

        ti = np.floor((xc - x_min_tile) / TILE_SIZE).astype(np.int32)
        tj = np.floor((yc - y_min_tile) / TILE_SIZE).astype(np.int32)
        np.clip(ti, 0, n_tiles_x - 1, out=ti)
        np.clip(tj, 0, n_tiles_y - 1, out=tj)

        tile_keys_mmap[start:end] = ti * n_tiles_y + tj
        del xc, yc, ti, tj

tile_keys_mmap.flush()
del tile_keys_mmap
gc.collect()

print("   ✓ Tile keys salvos em disco")

# ============================================
# PASSADA 2: CONTAR PONTOS POR TILE
# ============================================

print(f"\n4. PASSADA 2 — Contando pontos por tile...")

tile_keys = np.memmap(tile_keys_path, dtype=np.int32, mode='r', shape=(n_points,))

# Contar pontos por tile (histograma — super rápido)
tile_counts = np.zeros(n_tiles_total, dtype=np.int64)

for start in tqdm(range(0, n_points, CHUNK_SIZE), desc="   Contando", total=n_chunks):
    end = min(start + CHUNK_SIZE, n_points)
    chunk = np.array(tile_keys[start:end])
    counts_chunk = np.bincount(chunk, minlength=n_tiles_total)
    tile_counts += counts_chunk
    del chunk, counts_chunk

# Identificar tiles com dados
active_tiles = np.where(tile_counts > 0)[0]
n_tiles_with_data = len(active_tiles)

print(f"   Tiles com dados: {n_tiles_with_data}")
print(f"   Tiles vazios: {n_tiles_total - n_tiles_with_data}")

# Mostrar distribuição
active_counts = tile_counts[active_tiles]
print(f"   Pontos por tile — Min: {active_counts.min():,}  "
      f"Média: {active_counts.mean():.0f}  Max: {active_counts.max():,}")

# ============================================
# PASSADA 3: CRIAR TILES TEMPORÁRIOS (APPEND)
# Percorre o HDF5 sequencialmente UMA VEZ,
# distribui pontos nos arquivos de tile
# ============================================

print(f"\n5. PASSADA 3 — Distribuindo dados nos tiles (leitura sequencial)...")

TILES_WINTER_DIR.mkdir(exist_ok=True, parents=True)

# Pré-criar arquivos HDF5 para cada tile ativo com tamanho conhecido
tile_temp_dir = temp_dir / 'tile_data'
tile_temp_dir.mkdir(exist_ok=True, parents=True)

# Pré-criar arquivos temporários e rastrear posição de escrita
tile_write_pos = {}  # tile_key -> posição atual de escrita

print("   Criando arquivos temporários para tiles...")

# Detectar dtypes das variáveis
var_dtypes = {}
with h5py.File(winter_file, 'r') as f:
    for var in data_var_names:
        var_dtypes[var] = f[var].dtype

# Variáveis que precisamos garantir
need_xy_calc = ('x' not in data_var_names) and ('longitude' in data_var_names)
if need_xy_calc:
    var_dtypes['x'] = np.float64
    var_dtypes['y'] = np.float64

all_var_names = list(var_dtypes.keys())

for tile_key in tqdm(active_tiles, desc="   Pré-criando"):
    n_pts = int(tile_counts[tile_key])
    tile_h5_path = tile_temp_dir / f'temp_tile_{tile_key:06d}.h5'

    with h5py.File(tile_h5_path, 'w') as tf:
        for var in all_var_names:
            tf.create_dataset(var, shape=(n_pts,), dtype=var_dtypes[var])

    tile_write_pos[tile_key] = 0

print("   ✓ Arquivos temporários criados")

# Agora percorrer o HDF5 original sequencialmente e distribuir
print("   Distribuindo pontos (leitura sequencial do HDF5)...")

# Para evitar abrir/fechar milhares de arquivos por chunk,
# agrupamos as escritas: lemos um chunk, separamos por tile,
# e escrevemos em lote

with h5py.File(winter_file, 'r') as f:
    for start in tqdm(range(0, n_points, CHUNK_SIZE), desc="   Distribuindo", total=n_chunks):
        end = min(start + CHUNK_SIZE, n_points)
        chunk_size = end - start

        # Ler tile_keys deste chunk
        keys_chunk = np.array(tile_keys[start:end])

        # Ler todas as variáveis deste chunk (leitura sequencial = rápida)
        chunk_data = {}
        for var in data_var_names:
            chunk_data[var] = f[var][start:end]

        # Calcular x, y se necessário
        if need_xy_calc:
            cx, cy = lonlat_to_xy(chunk_data['longitude'], chunk_data['latitude'])
            chunk_data['x'] = cx
            chunk_data['y'] = cy
            del cx, cy

        # Encontrar quais tiles estão neste chunk
        unique_in_chunk = np.unique(keys_chunk)

        # Para cada tile neste chunk, extrair e escrever
        for tk in unique_in_chunk:
            mask = keys_chunk == tk
            n_pts_this = int(np.sum(mask))

            if n_pts_this == 0:
                continue

            tile_h5_path = tile_temp_dir / f'temp_tile_{tk:06d}.h5'
            wp = tile_write_pos[tk]

            with h5py.File(tile_h5_path, 'a') as tf:
                for var in all_var_names:
                    if var in chunk_data:
                        tf[var][wp:wp + n_pts_this] = chunk_data[var][mask]

            tile_write_pos[tk] = wp + n_pts_this

        del keys_chunk, chunk_data, unique_in_chunk
        gc.collect()

del tile_keys
gc.collect()

print("   ✓ Dados distribuídos!")

# ============================================
# PASSADA 4: CONVERTER TILES TEMP → TILES FINAIS
# ============================================

print(f"\n6. PASSADA 4 — Convertendo para tiles finais...")

# Ler metadata
meta_data = {}
with h5py.File(winter_file, 'r') as f:
    for var in meta_var_names:
        try:
            meta_data[var] = f[var][:]
        except Exception:
            continue

n_tiles_created = 0
tile_stats = []

for tile_key in tqdm(active_tiles, desc="   Finalizando"):
    i = int(tile_key // n_tiles_y)
    j = int(tile_key % n_tiles_y)
    n_pts_tile = int(tile_counts[tile_key])

    if n_pts_tile == 0:
        continue

    # Limites do tile
    x_tile_min = x_edges[i]
    x_tile_max = x_edges[i + 1]
    y_tile_min = y_edges[j]
    y_tile_max = y_edges[j + 1]
    x_center = (x_tile_min + x_tile_max) / 2
    y_center = (y_tile_min + y_tile_max) / 2
    lon_center, lat_center = xy_to_lonlat(x_center, y_center)

    # Ler do temporário
    tile_h5_path = tile_temp_dir / f'temp_tile_{tile_key:06d}.h5'
    tile_data = {}

    with h5py.File(tile_h5_path, 'r') as tf:
        for var in tf.keys():
            tile_data[var] = tf[var][:]

    # Metadata
    for var, val in meta_data.items():
        tile_data[var] = val

    # Metadados do tile
    tile_data['tile_id'] = f"{i:04d}_{j:04d}"
    tile_data['tile_i'] = i
    tile_data['tile_j'] = j
    tile_data['tile_x_min'] = x_tile_min
    tile_data['tile_x_max'] = x_tile_max
    tile_data['tile_y_min'] = y_tile_min
    tile_data['tile_y_max'] = y_tile_max
    tile_data['tile_x_center'] = x_center
    tile_data['tile_y_center'] = y_center
    tile_data['tile_lon_center'] = lon_center
    tile_data['tile_lat_center'] = lat_center
    tile_data['n_points_tile'] = n_pts_tile

    # Salvar tile final
    tile_filepath = TILES_WINTER_DIR / f"tile_{i:04d}_{j:04d}.h5"

    try:
        write_hdf5(tile_filepath, tile_data)
        n_tiles_created += 1
        tile_stats.append({
            'tile_id': f"{i:04d}_{j:04d}",
            'tile_i': i,
            'tile_j': j,
            'n_points': n_pts_tile,
            'x_center': x_center / 1000,
            'y_center': y_center / 1000,
            'lon_center': lon_center,
            'lat_center': lat_center
        })
    except Exception as e:
        print(f"\n✗ Erro tile {i:04d}_{j:04d}: {e}")
        continue

    del tile_data

# ============================================
# LIMPEZA
# ============================================

print("\n7. Limpando arquivos temporários...")

try:
    shutil.rmtree(temp_dir)
    print("   ✓ Temporários removidos")
except Exception as e:
    print(f"   ⚠ Remoção parcial: {e}")
    print(f"   Delete manualmente: {temp_dir}")

# ============================================
# LOG
# ============================================

if len(tile_stats) > 0:
    df_tiles = pd.DataFrame(tile_stats)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOGS_DIR / 'tiles_winter_log.csv'
    df_tiles.to_csv(log_file, index=False)
    print(f"\n   ✓ Log salvo: {log_file}")

# ============================================
# RESUMO
# ============================================

print("\n" + "=" * 70)
print("RESUMO DA CRIAÇÃO DE TILES")
print("=" * 70)
print(f"Pontos de entrada: {n_points:,}")
print(f"Tiles possíveis: {n_tiles_total}")
print(f"Tiles com dados: {n_tiles_with_data}")
print(f"Tiles criados: {n_tiles_created}")

if len(tile_stats) > 0:
    df = pd.DataFrame(tile_stats)
    print(f"\nEstatísticas dos tiles:")
    print(f"  Pontos por tile - Mínimo: {df['n_points'].min():,}")
    print(f"  Pontos por tile - Média: {df['n_points'].mean():.0f}")
    print(f"  Pontos por tile - Máximo: {df['n_points'].max():,}")
    total_pts = df['n_points'].sum()
    print(f"\n  Total de pontos nos tiles: {total_pts:,}")
    print(f"  Cobertura: {100*total_pts/n_points:.1f}%")

print(f"\nDiretório: {TILES_WINTER_DIR}")
print("=" * 70)
print("\n✓ Criação de tiles concluída!")
print("\nPróximo passo: 09_apply_corrections.py")