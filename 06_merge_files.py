"""
Mesclagem de arquivos ATL06 - VERSÃO MEMORY-SAFE
Processa em chunks para evitar problemas de RAM
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import h5py
from tqdm import tqdm
from datetime import datetime

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
    PROCESSED_DIR,
    DATA_DIR
)

sys.path.insert(0, str(SCRIPT_DIR / 'utils'))
from io_utils import read_hdf5

print("="*70)
print("MESCLAGEM DE ARQUIVOS ATL06 - VERSÃO MEMORY-SAFE")
print("="*70)

# ============================================
# CONFIGURAÇÃO
# ============================================

# Processar em lotes para controlar uso de RAM
CHUNK_SIZE = 300  # Arquivos por chunk (ajuste conforme RAM disponível)

# ============================================
# LISTAR ARQUIVOS
# ============================================

print("\n1. Listando arquivos filtrados...")
processed_files = sorted(list(PROCESSED_DIR.glob("*.h5")))
n_files = len(processed_files)
print(f"   Encontrados: {n_files} arquivos")

if n_files == 0:
    print("\n✗ Nenhum arquivo encontrado!")
    sys.exit(1)

# ============================================
# DETERMINAR VARIÁVEIS
# ============================================

print("\n2. Determinando estrutura de dados...")

first_data = None
for filepath in processed_files[:20]:
    try:
        first_data = read_hdf5(filepath)
        if first_data and 'latitude' in first_data:
            lat = np.atleast_1d(first_data['latitude'])
            if len(lat) > 1:
                break
    except:
        continue

if first_data is None:
    print("✗ Não foi possível ler nenhum arquivo!")
    sys.exit(1)

# Identificar variáveis array (excluir as que deram erro de RAM)
SKIP_VARS = ['cycle', 'delta_time']  # Variáveis que deram erro

array_vars = []
for key, value in first_data.items():
    if isinstance(value, np.ndarray):
        arr = np.atleast_1d(value)
        if arr.size > 0 and key not in SKIP_VARS:
            array_vars.append(key)

print(f"   Variáveis a mesclar ({len(array_vars)}):")
for var in sorted(array_vars):
    print(f"      - {var}")

# ============================================
# ARQUIVO DE SAÍDA
# ============================================

region_name = THWAITES_BBOX['name'].lower().replace(' ', '_')
output_file = DATA_DIR / f'{region_name}_atl06_merged.h5'
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Remover se já existir
if output_file.exists():
    output_file.unlink()
    print(f"\n   ✓ Arquivo anterior removido")

print(f"\n   Arquivo de saída: {output_file.name}")

# ============================================
# PROCESSAR EM CHUNKS - SALVAR DIRETO EM HDF5
# ============================================

print(f"\n3. Mesclando em chunks de {CHUNK_SIZE} arquivos...")
print("   (Evita problemas de RAM)")

n_chunks = (n_files + CHUNK_SIZE - 1) // CHUNK_SIZE
n_total_points = 0
n_files_read = 0
n_files_error = 0

# Criar arquivo HDF5 de saída com datasets extensíveis
with h5py.File(output_file, 'w') as f_out:
    
    datasets_created = False
    
    for chunk_idx in range(n_chunks):
        
        # Arquivos deste chunk
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_files)
        chunk_files = processed_files[start:end]
        
        print(f"\n   Chunk {chunk_idx+1}/{n_chunks}: arquivos {start+1}-{end}")
        
        # Acumular dados do chunk na RAM
        chunk_data = {var: [] for var in array_vars}
        
        for filepath in tqdm(chunk_files, desc=f"  Chunk {chunk_idx+1}", leave=False):
            try:
                data = read_hdf5(filepath, variables=array_vars)
                
                if not data or 'latitude' not in data:
                    continue
                
                lat = np.atleast_1d(data['latitude'])
                if len(lat) == 0:
                    continue
                
                for var in array_vars:
                    if var in data:
                        val = np.atleast_1d(data[var])
                        if val.size > 0:
                            chunk_data[var].append(val)
                
                n_files_read += 1
                
            except Exception as e:
                n_files_error += 1
                continue
        
        # Concatenar chunk
        chunk_arrays = {}
        n_chunk_points = 0
        
        for var in array_vars:
            if len(chunk_data[var]) > 0:
                try:
                    chunk_arrays[var] = np.concatenate(chunk_data[var])
                    if var == 'latitude':
                        n_chunk_points = len(chunk_arrays[var])
                except:
                    chunk_arrays[var] = np.array([])
            else:
                chunk_arrays[var] = np.array([])
        
        if n_chunk_points == 0:
            print(f"   ⚠ Chunk {chunk_idx+1} vazio, pulando...")
            continue
        
        print(f"   → {n_chunk_points:,} pontos neste chunk")
        
        # Salvar chunk no HDF5
        if not datasets_created:
            # Criar datasets extensíveis na primeira vez
            for var, arr in chunk_arrays.items():
                if arr.size > 0:
                    maxshape = (None,) + arr.shape[1:]
                    f_out.create_dataset(
                        var,
                        data=arr,
                        maxshape=maxshape,
                        chunks=True,
                        compression='gzip',
                        compression_opts=4
                    )
            datasets_created = True
            
        else:
            # Adicionar (extend) aos datasets existentes
            for var, arr in chunk_arrays.items():
                if arr.size == 0:
                    continue
                    
                if var not in f_out:
                    # Dataset novo - criar
                    maxshape = (None,) + arr.shape[1:]
                    f_out.create_dataset(
                        var,
                        data=arr,
                        maxshape=maxshape,
                        chunks=True,
                        compression='gzip',
                        compression_opts=4
                    )
                else:
                    # Estender dataset existente
                    current_size = f_out[var].shape[0]
                    new_size = current_size + arr.shape[0]
                    f_out[var].resize(new_size, axis=0)
                    f_out[var][current_size:new_size] = arr
        
        n_total_points += n_chunk_points
        print(f"   Total acumulado: {n_total_points:,} pontos")
        
        # Limpar memória
        del chunk_data, chunk_arrays
        import gc
        gc.collect()

# ============================================
# ADICIONAR METADADOS E COORDENADAS
# ============================================

print(f"\n4. Adicionando metadados e coordenadas...")

with h5py.File(output_file, 'a') as f_out:
    
    n_pts = f_out['latitude'].shape[0]
    
    # Calcular ano e mês a partir de t_year
    if 't_year' in f_out and 'year' not in f_out:
        print("   Calculando ano e mês...")
        
        t_year = f_out['t_year'][:]
        year = np.floor(t_year).astype(int)
        year_frac = t_year - year
        month = np.clip(np.round(year_frac * 12 + 0.5).astype(int), 1, 12)
        
        f_out.create_dataset('year', data=year, compression='gzip')
        f_out.create_dataset('month', data=month, compression='gzip')
        print("   ✓ year, month calculados")
    
    # Calcular coordenadas projetadas em chunks (evita erro de RAM)
    if 'x' not in f_out and 'latitude' in f_out:
        print("   Calculando coordenadas projetadas em chunks...")
        
        sys.path.insert(0, str(SCRIPT_DIR / 'utils'))
        from geodetic_utils import lonlat_to_xy
        
        n_pts = f_out['latitude'].shape[0]
        
        # Criar datasets vazios para x e y
        f_out.create_dataset('x', shape=(n_pts,), dtype='float64',
                            maxshape=(None,), chunks=True, compression='gzip')
        f_out.create_dataset('y', shape=(n_pts,), dtype='float64',
                            maxshape=(None,), chunks=True, compression='gzip')
        
        # Processar em chunks de 1 milhão de pontos
        COORD_CHUNK = 1_000_000
        n_coord_chunks = (n_pts + COORD_CHUNK - 1) // COORD_CHUNK
        
        for i in tqdm(range(n_coord_chunks), desc="   Coordenadas"):
            s = i * COORD_CHUNK
            e = min(s + COORD_CHUNK, n_pts)
            
            lon_chunk = f_out['longitude'][s:e]
            lat_chunk = f_out['latitude'][s:e]
            
            x_chunk, y_chunk = lonlat_to_xy(lon_chunk, lat_chunk)
            
            f_out['x'][s:e] = x_chunk
            f_out['y'][s:e] = y_chunk
        
        print("   ✓ Coordenadas x, y calculadas (EPSG:3031)")
    
    # Metadados gerais
    f_out.attrs['n_points'] = n_total_points
    f_out.attrs['n_files_merged'] = n_files_read
    f_out.attrs['n_files_error'] = n_files_error
    f_out.attrs['region'] = THWAITES_BBOX['name']
    f_out.attrs['bbox_lon_min'] = THWAITES_BBOX['lon_min']
    f_out.attrs['bbox_lon_max'] = THWAITES_BBOX['lon_max']
    f_out.attrs['bbox_lat_min'] = THWAITES_BBOX['lat_min']
    f_out.attrs['bbox_lat_max'] = THWAITES_BBOX['lat_max']
    f_out.attrs['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ============================================
# RESUMO
# ============================================

file_size_gb = output_file.stat().st_size / (1024**3)

print("\n" + "="*70)
print("RESUMO DA MESCLAGEM")
print("="*70)
print(f"Arquivos totais: {n_files}")
print(f"Arquivos lidos: {n_files_read} ({100*n_files_read/n_files:.1f}%)")
print(f"Arquivos com erro: {n_files_error}")
print(f"Total de pontos: {n_total_points:,}")
print(f"\nArquivo: {output_file}")
print(f"Tamanho: {file_size_gb:.2f} GB")

# Ler estatísticas do arquivo final
with h5py.File(output_file, 'r') as f:
    
    print(f"\nVariáveis no arquivo:")
    for key in sorted(f.keys()):
        shape = f[key].shape
        dtype = f[key].dtype
        print(f"  {key:30s} {str(shape):20s} {dtype}")
    
    if 'year' in f:
        years = f['year'][:]
        print(f"\nDistribuição por ano:")
        for yr in sorted(np.unique(years)):
            n = np.sum(years == yr)
            print(f"  {yr}: {n:,} pontos ({100*n/n_total_points:.1f}%)")
    
    if 'month' in f:
        months = f['month'][:]
        month_names = {6: 'Jun', 7: 'Jul', 8: 'Ago',
                       1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr',
                       5: 'Mai', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
        print(f"\nDistribuição por mês:")
        for mo in sorted(np.unique(months)):
            n = np.sum(months == mo)
            print(f"  {month_names.get(mo, mo)}: {n:,} pontos ({100*n/n_total_points:.1f}%)")

print("="*70)
print("\n✓ Mesclagem concluída!")
print("\nPróximo passo: 07_separate_winter.py")
