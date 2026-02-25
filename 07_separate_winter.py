"""
Separação de dados de inverno austral
"""

import numpy as np
import h5py
from pathlib import Path
import sys
from tqdm import tqdm

# ============================================
# DETECTAR DIRETÓRIO AUTOMATICAMENTE
# ============================================

if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).parent
else:
    SCRIPT_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")

sys.path.insert(0, str(SCRIPT_DIR))

# Importar configurações - APENAS O QUE EXISTE NO CONFIG
from config import (
    THWAITES_BBOX,
    WINTER_MONTHS,    # [6, 7, 8] ← existe no config
    START_YEAR,       # 2019 ← existe no config
    END_YEAR,         # 2024 ← existe no config
    DATA_DIR,
    LOGS_DIR
)

# Importar utilitários
sys.path.insert(0, str(SCRIPT_DIR / 'utils'))
from io_utils import read_hdf5, write_hdf5
from filter_utils import filter_by_month, apply_mask

# ============================================
# DEFINIR WINTER_LABEL LOCALMENTE
# (não existe no config, definir aqui)
# ============================================

WINTER_LABEL = "JJA"  # Junho, Julho, Agosto

month_names = {
    1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr',
    5:'Mai', 6:'Jun', 7:'Jul', 8:'Ago',
    9:'Set', 10:'Out', 11:'Nov', 12:'Dez'
}

print("="*70)
print("SEPARAÇÃO DE DADOS DE INVERNO AUSTRAL")
print("Baseado em: filtst.py do CAPTOOLKIT")
print("="*70)

# ============================================
# DEFINIÇÃO DE INVERNO
# ============================================

print(f"\nEstação: Inverno Austral ({WINTER_LABEL})")
print(f"Meses selecionados: {WINTER_MONTHS}")
for m in WINTER_MONTHS:
    print(f"  → Mês {m} ({month_names.get(m, m)})")
print(f"Período: {START_YEAR} a {END_YEAR}")

# ============================================
# ARQUIVO DE ENTRADA (MESCLADO)
# ============================================

print("\n1. Localizando arquivo mesclado...")

# Buscar qualquer arquivo *merged*.h5
merged_files = list(DATA_DIR.glob("*merged*.h5"))

if len(merged_files) == 0:
    print(f"\n✗ Nenhum arquivo mesclado encontrado em: {DATA_DIR}")
    print("   Execute primeiro: 06_merge_files.py")
    sys.exit(1)

# Usar o mais recente
input_file = sorted(merged_files)[-1]
print(f"   Arquivo: {input_file.name}")

file_size_gb = input_file.stat().st_size / (1024**3)
print(f"   Tamanho: {file_size_gb:.2f} GB")

# ============================================
# ARQUIVO DE SAÍDA
# ============================================

region_name = THWAITES_BBOX['name'].lower().replace(' ', '_')
output_file = DATA_DIR / f'{region_name}_atl06_winter_{WINTER_LABEL}.h5'
print(f"\n   Saída: {output_file.name}")

# ============================================
# VERIFICAR VARIÁVEIS DISPONÍVEIS
# ============================================

with h5py.File(input_file, 'r') as f:
    available_vars = list(f.keys())
    n_total = f['latitude'].shape[0]
    
    has_month  = 'month'  in available_vars
    has_t_year = 't_year' in available_vars
    has_year   = 'year'   in available_vars
    
    print(f"\n2. Arquivo de entrada:")
    print(f"   Total de pontos: {n_total:,}")
    print(f"   Variáveis: {len(available_vars)}")
    print(f"   Tem 'month':  {has_month}")
    print(f"   Tem 't_year': {has_t_year}")
    print(f"   Tem 'year':   {has_year}")

if not has_month and not has_t_year:
    print("\n✗ Arquivo não tem variável temporal!")
    sys.exit(1)

# ============================================
# PROCESSAR EM CHUNKS - MEMORY SAFE
# ============================================

CHUNK_SIZE = 5_000_000  # 5 milhões de pontos por chunk
n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"\n3. Filtrando inverno em {n_chunks} chunks de {CHUNK_SIZE:,} pontos...")

# ============================================
# PASSAGEM 1: CONTAR PONTOS DE INVERNO
# ============================================

print("\n   Pré-contagem de pontos de inverno...")

n_winter_total = 0

with h5py.File(input_file, 'r') as f_in:
    for chunk_idx in tqdm(range(n_chunks), desc="   Contando"):
        s = chunk_idx * CHUNK_SIZE
        e = min(s + CHUNK_SIZE, n_total)
        
        # Obter meses do chunk
        if has_month:
            month_chunk = f_in['month'][s:e]
        else:
            # Calcular de t_year
            t_chunk = f_in['t_year'][s:e]
            year_frac = t_chunk - np.floor(t_chunk)
            month_chunk = np.clip(
                np.floor(year_frac * 12 + 0.5).astype(int), 1, 12
            )
        
        # Máscara de inverno
        winter_mask = np.zeros(len(month_chunk), dtype=bool)
        for m in WINTER_MONTHS:
            winter_mask |= (month_chunk == m)
        
        n_winter_total += int(np.sum(winter_mask))

print(f"   Pontos de inverno: {n_winter_total:,}")
print(f"   ({100*n_winter_total/n_total:.1f}% do total)")

if n_winter_total == 0:
    print(f"\n✗ Nenhum ponto de inverno! Verifique WINTER_MONTHS={WINTER_MONTHS}")
    sys.exit(1)

# ============================================
# PASSAGEM 2: SALVAR PONTOS DE INVERNO
# ============================================

print("\n4. Salvando dados de inverno...")

# Remover arquivo anterior se existir
if output_file.exists():
    output_file.unlink()

datasets_created = False

with h5py.File(input_file, 'r') as f_in:
    with h5py.File(output_file, 'w') as f_out:
        
        for chunk_idx in tqdm(range(n_chunks), desc="   Filtrando"):
            s = chunk_idx * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, n_total)
            
            # Obter meses do chunk
            if has_month:
                month_chunk = f_in['month'][s:e]
            else:
                t_chunk = f_in['t_year'][s:e]
                year_frac = t_chunk - np.floor(t_chunk)
                month_chunk = np.clip(
                    np.floor(year_frac * 12 + 0.5).astype(int), 1, 12
                )
            
            # Máscara de inverno
            winter_mask = np.zeros(len(month_chunk), dtype=bool)
            for m in WINTER_MONTHS:
                winter_mask |= (month_chunk == m)
            
            # Pular se não tem dados de inverno neste chunk
            if not np.any(winter_mask):
                continue
            
            # Ler e filtrar TODAS as variáveis do chunk
            chunk_data = {}
            for var in available_vars:
                try:
                    var_chunk = f_in[var][s:e]
                    chunk_data[var] = var_chunk[winter_mask]
                except Exception:
                    continue
            
            # Salvar chunk no HDF5 de saída
            if not datasets_created:
                # Criar datasets extensíveis na primeira vez
                for var, arr in chunk_data.items():
                    if arr.size > 0:
                        f_out.create_dataset(
                            var,
                            data=arr,
                            maxshape=(None,) + arr.shape[1:],
                            chunks=True,
                            compression='gzip',
                            compression_opts=4
                        )
                datasets_created = True
            else:
                # Estender datasets existentes
                for var, arr in chunk_data.items():
                    if arr.size == 0:
                        continue
                    if var not in f_out:
                        f_out.create_dataset(
                            var,
                            data=arr,
                            maxshape=(None,) + arr.shape[1:],
                            chunks=True,
                            compression='gzip',
                            compression_opts=4
                        )
                    else:
                        cur = f_out[var].shape[0]
                        new = cur + arr.shape[0]
                        f_out[var].resize(new, axis=0)
                        f_out[var][cur:new] = arr
        
        # Metadados
        f_out.attrs['n_points']   = n_winter_total
        f_out.attrs['season']     = WINTER_LABEL
        f_out.attrs['months']     = str(WINTER_MONTHS)
        f_out.attrs['year_start'] = START_YEAR
        f_out.attrs['year_end']   = END_YEAR
        f_out.attrs['region']     = THWAITES_BBOX['name']

# ============================================
# VERIFICAR RESULTADO
# ============================================

print("\n5. Verificando arquivo de saída...")

with h5py.File(output_file, 'r') as f:
    n_saved = f['latitude'].shape[0]
    
    print(f"   ✓ Pontos salvos: {n_saved:,}")
    
    # Distribuição por mês
    if 'month' in f:
        months_arr = f['month'][:]
        print(f"\n   Distribuição por mês:")
        for m in sorted(np.unique(months_arr)):
            n = int(np.sum(months_arr == m))
            print(f"   → {month_names.get(int(m), m)}: {n:,} ({100*n/n_saved:.1f}%)")
    
    # Distribuição por ano
    if 'year' in f:
        years_arr = f['year'][:]
        print(f"\n   Distribuição por ano:")
        for yr in sorted(np.unique(years_arr)):
            n = int(np.sum(years_arr == yr))
            print(f"   → {int(yr)}: {n:,} ({100*n/n_saved:.1f}%)")

# ============================================
# RESUMO
# ============================================

file_size_mb = output_file.stat().st_size / (1024**2)

print("\n" + "="*70)
print("RESUMO DA SEPARAÇÃO SAZONAL")
print("="*70)
print(f"Arquivo entrada: {input_file.name}")
print(f"Pontos totais:   {n_total:,}")
print(f"Pontos inverno:  {n_winter_total:,} ({100*n_winter_total/n_total:.1f}%)")
print(f"\nArquivo saída:   {output_file.name}")
print(f"Tamanho:         {file_size_mb:.1f} MB")
print("="*70)

print("\n✓ Separação de inverno concluída!")

print("\nPróximo passo: 08_create_tiles.py")
