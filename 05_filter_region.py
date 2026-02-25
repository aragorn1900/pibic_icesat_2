"""
Filtragem por região geográfica
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
    THWAITES_BBOX,
    PROCESSED_DIR,
    LOGS_DIR
)

# Importar utilitários
sys.path.insert(0, str(SCRIPT_DIR / 'utils'))
from io_utils import read_hdf5, write_hdf5
from filter_utils import filter_region, apply_mask

print("="*70)
print("FILTRAGEM POR REGIÃO GEOGRÁFICA")
print("Baseado em: filtmask.py do CAPTOOLKIT")
print("="*70)

# ============================================
# REGIÃO DE INTERESSE
# ============================================

print(f"\nRegião: {THWAITES_BBOX['name']}")
print(f"Bounding Box:")
print(f"  Longitude: {THWAITES_BBOX['lon_min']}° a {THWAITES_BBOX['lon_max']}°")
print(f"  Latitude: {THWAITES_BBOX['lat_min']}° a {THWAITES_BBOX['lat_max']}°")

area_lon = THWAITES_BBOX['lon_max'] - THWAITES_BBOX['lon_min']
area_lat = THWAITES_BBOX['lat_max'] - THWAITES_BBOX['lat_min']
print(f"  Área: {area_lon}° × {area_lat}°")

# ============================================
# LISTAR ARQUIVOS
# ============================================

print("\n1. Listando arquivos filtrados por qualidade...")
processed_files = sorted(list(PROCESSED_DIR.glob("*.h5")))
print(f"   Encontrados: {len(processed_files)} arquivos")

if len(processed_files) == 0:
    print("\n✗ Nenhum arquivo encontrado!")
    print(f"   Diretório: {PROCESSED_DIR}")
    print("   Execute primeiro: 04_filter_quality.py")
    sys.exit(1)

# ============================================
# FILTRAR POR REGIÃO
# ============================================

print("\n2. Filtrando por bounding box...")

n_files_processed = 0
n_files_empty = 0
n_points_before = 0
n_points_after = 0
region_log = []

for filepath in tqdm(processed_files, desc="Filtrando região"):
    
    try:
        # Ler dados
        data = read_hdf5(filepath)
        
        # Verificar se tem dados
        if 'latitude' not in data:
            # Arquivo sem dados de coordenadas
            filepath.unlink()
            n_files_empty += 1
            continue
        
        # Garantir que latitude é array
        lat = np.atleast_1d(data['latitude'])
        
        if len(lat) == 0:
            # Arquivo vazio
            filepath.unlink()
            n_files_empty += 1
            continue
        
        n_before = len(lat)
        n_points_before += n_before
        
        # Aplicar filtro de região (bounding box)
        mask = filter_region(data, THWAITES_BBOX)
        
        # Contar pontos dentro da região
        n_inside = np.sum(mask)
        
        # Aplicar máscara
        if n_inside > 0:
            filtered_data = apply_mask(data, mask)
            
            # Verificar se ainda tem dados após aplicar máscara
            if 'latitude' in filtered_data:
                lat_filtered = np.atleast_1d(filtered_data['latitude'])
                n_after = len(lat_filtered)
            else:
                n_after = 0
            
            if n_after > 0:
                n_points_after += n_after
                
                # Salvar (sobrescrever)
                write_hdf5(filepath, filtered_data)
                status = 'OK'
            else:
                # Filtrado ficou vazio
                filepath.unlink()
                n_files_empty += 1
                n_after = 0
                status = 'VAZIO - DELETADO'
        else:
            # Totalmente fora da região
            filepath.unlink()
            n_files_empty += 1
            n_after = 0
            status = 'FORA DA REGIÃO - DELETADO'
        
        # Percentual mantido
        pct = 100 * n_after / n_before if n_before > 0 else 0
        
        # Log
        region_log.append({
            'file': filepath.name,
            'n_before': n_before,
            'n_after': n_after,
            'n_removed': n_before - n_after,
            'pct_kept': pct,
            'status': status
        })
        
        n_files_processed += 1
        
    except Exception as e:
        print(f"\n✗ Erro ao filtrar {filepath.name}: {e}")
        region_log.append({
            'file': filepath.name,
            'n_before': 0,
            'n_after': 0,
            'n_removed': 0,
            'pct_kept': 0,
            'status': f'ERROR: {str(e)[:50]}'
        })
        continue

# ============================================
# SALVAR LOG
# ============================================

if len(region_log) > 0:
    df_log = pd.DataFrame(region_log)
    
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    log_file = LOGS_DIR / 'atl06_region_filter_log.csv'
    df_log.to_csv(log_file, index=False)
    
    print(f"\n   ✓ Log salvo: {log_file}")

# ============================================
# RESUMO
# ============================================

print("\n" + "="*70)
print("RESUMO DA FILTRAGEM REGIONAL")
print("="*70)
print(f"Arquivos processados: {n_files_processed}")
print(f"Arquivos deletados (vazios/fora): {n_files_empty}")
print(f"Arquivos mantidos: {n_files_processed - n_files_empty}")

print(f"\nPontos ANTES: {n_points_before:,}")
print(f"Pontos DEPOIS: {n_points_after:,}")

if n_points_before > 0:
    n_removed = n_points_before - n_points_after
    pct_removed = 100 * n_removed / n_points_before
    pct_kept = 100 * n_points_after / n_points_before
    
    print(f"Pontos removidos: {n_removed:,} ({pct_removed:.1f}%)")
    print(f"Pontos mantidos: {n_points_after:,} ({pct_kept:.1f}%)")

# Arquivos finais
remaining_files = list(PROCESSED_DIR.glob("*.h5"))
print(f"\nArquivos válidos restantes: {len(remaining_files)}")

print(f"\nDiretório: {PROCESSED_DIR}")
print("="*70)

print("\n✓ Filtragem regional concluída!")

print("\nPróximo passo: 06_merge_files.py")
