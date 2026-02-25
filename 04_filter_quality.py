"""
Filtragem de dados por qualidade
Equivalente a: filtnan.py do CAPTOOLKIT
Versão 2.0 - Box Expandida
"""

import numpy as np
import pandas as pd
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

# Importar configurações
from config import (
    PROCESSED_DIR,
    QUALITY_FILTERS,  # ← CORRETO (não QUALITY_FLAGS)
    LOGS_DIR
)

# Importar utils
sys.path.insert(0, str(SCRIPT_DIR / 'utils'))
from io_utils import read_hdf5, write_hdf5
from filter_utils import filter_quality, apply_mask

print("="*70)
print("FILTRAGEM DE QUALIDADE - ATL06")
print("="*70)

# ============================================
# CONFIGURAÇÕES DE QUALIDADE
# ============================================

print("\nCritérios de qualidade:")
print(f"  - atl06_quality_summary ≤ {QUALITY_FILTERS['atl06_quality_summary']}")  # ← CORRIGIDO
print(f"  - h_li_sigma < {QUALITY_FILTERS['h_li_sigma']} m")  # ← CORRIGIDO
print(f"  - h_robust_sprd < {QUALITY_FILTERS['h_robust_sprd']} m")  # ← CORRIGIDO
h_min, h_max = QUALITY_FILTERS['height_range']  # ← CORRIGIDO
print(f"  - Altura entre {h_min} e {h_max} m")
print(f"  - Sem NaNs em coordenadas e tempo")

# ============================================
# LISTAR ARQUIVOS PROCESSADOS
# ============================================

print("\n1. Listando arquivos processados...")
processed_files = sorted(list(PROCESSED_DIR.glob("*.h5")))
print(f"   Encontrados: {len(processed_files)} arquivos")

if len(processed_files) == 0:
    print("\n✗ Nenhum arquivo processado encontrado!")
    print(f"   Diretório: {PROCESSED_DIR}")
    print("   Execute primeiro: 03_read_atl06.py")
    sys.exit(1)

# ============================================
# FILTRAR ARQUIVOS
# ============================================

print("\n2. Aplicando filtros de qualidade...")
print("   (Isso pode demorar ~2-3 horas para box grande!)")

n_files_processed = 0
n_files_empty = 0
n_points_before = 0
n_points_after = 0
filter_log = []

for filepath in tqdm(processed_files, desc="Filtrando"):
    
    try:
        # Ler dados
        data = read_hdf5(filepath)
        
        # Verificar se tem dados
        if 'latitude' not in data or len(data['latitude']) == 0:
            # Arquivo já vazio
            filepath.unlink()  # Deletar
            n_files_empty += 1
            continue
        
        n_before = len(data['latitude'])
        n_points_before += n_before
        
        # Aplicar filtro de qualidade
        mask = filter_quality(
            data,
            max_quality=QUALITY_FILTERS['atl06_quality_summary'],  # ← CORRIGIDO
            max_sigma=QUALITY_FILTERS['h_li_sigma'],  # ← CORRIGIDO
            height_range=QUALITY_FILTERS['height_range']  # ← CORRIGIDO
        )
        
        # Aplicar máscara
        filtered_data = apply_mask(data, mask)
        
        n_after = len(filtered_data['latitude'])
        n_points_after += n_after
        
        # Percentual mantido
        pct = 100 * n_after / n_before if n_before > 0 else 0
        
        # Salvar ou deletar
        if n_after > 0:
            # Sobrescrever arquivo com dados filtrados
            write_hdf5(filepath, filtered_data)
            status = 'OK'
        else:
            # Arquivo ficou vazio - deletar
            filepath.unlink()
            status = 'VAZIO - DELETADO'
            n_files_empty += 1
        
        # Log
        filter_log.append({
            'file': filepath.name,
            'n_before': n_before,
            'n_after': n_after,
            'pct_kept': pct,
            'status': status
        })
        
        n_files_processed += 1
        
    except Exception as e:
        print(f"\n✗ Erro ao filtrar {filepath.name}: {e}")
        filter_log.append({
            'file': filepath.name,
            'n_before': 0,
            'n_after': 0,
            'pct_kept': 0,
            'status': f'ERROR: {str(e)[:50]}'
        })
        continue

# ============================================
# SALVAR LOG
# ============================================

if len(filter_log) > 0:
    df_log = pd.DataFrame(filter_log)
    
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOGS_DIR / 'atl06_quality_filter_log.csv'
    df_log.to_csv(log_file, index=False)
    
    print(f"\n   ✓ Log salvo: {log_file}")

# ============================================
# RESUMO
# ============================================

print("\n" + "="*70)
print("RESUMO DA FILTRAGEM")
print("="*70)
print(f"Arquivos processados: {n_files_processed}")
print(f"Arquivos deletados (vazios): {n_files_empty}")

print(f"\nPontos ANTES dos filtros: {n_points_before:,}")
print(f"Pontos DEPOIS dos filtros: {n_points_after:,}")

if n_points_before > 0:
    n_removed = n_points_before - n_points_after
    pct_removed = 100 * n_removed / n_points_before
    pct_kept = 100 * n_points_after / n_points_before
    
    print(f"Pontos removidos: {n_removed:,} ({pct_removed:.1f}%)")
    print(f"Pontos mantidos: {n_points_after:,} ({pct_kept:.1f}%)")

# Arquivos restantes
remaining_files = list(PROCESSED_DIR.glob("*.h5"))
print(f"\nArquivos válidos restantes: {len(remaining_files)}")

print(f"\nDiretório: {PROCESSED_DIR}")
print("="*70)

print("\n✓ Filtragem concluída!")
print("\nPróximo passo: 05_filter_region.py")