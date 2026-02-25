"""
Aplicação de correções geofísicas aos dados
Equivalente a: corrslope.py, corrtide.py, corribe.py do CAPTOOLKIT
VERSÃO FINAL CORRIGIDA
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

# Importar configurações
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import *

# Importar utilitários
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts\utils")
from io_utils import read_hdf5, write_hdf5

print("="*60)
print("APLICAÇÃO DE CORREÇÕES GEOFÍSICAS")
print("Baseado em: corrslope.py, corrtide.py, corribe.py")
print("="*60)

# ============================================
# RESOLVER DIRETÓRIO DE TILES
# ============================================

# Tentar usar TILES_WINTER_DIR do config; se não existir, buscar automaticamente
try:
    _tiles_dir = TILES_WINTER_DIR
except NameError:
    # Buscar qualquer pasta que contenha "winter" e tenha tiles dentro
    BASE_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")
    candidates = []

    # Procurar em locais comuns
    for search_root in [BASE_DIR / 'data', BASE_DIR / 'tiles', BASE_DIR, DATA_DIR if 'DATA_DIR' in dir() else BASE_DIR / 'data']:
        if search_root.exists():
            for d in search_root.rglob("*winter*"):
                if d.is_dir() and list(d.glob("tile_*.h5")):
                    candidates.append(d)

    if candidates:
        _tiles_dir = sorted(candidates, key=lambda p: len(list(p.glob("tile_*.h5"))), reverse=True)[0]
        print(f"   ⚠ TILES_WINTER_DIR não definido no config.py")
        print(f"   ✓ Diretório detectado automaticamente: {_tiles_dir}")
    else:
        print("✗ Não foi possível encontrar diretório de tiles!")
        print("  Defina TILES_WINTER_DIR no config.py ou verifique se 08_create_tiles.py foi executado.")
        sys.exit(1)

TILES_WINTER_DIR = _tiles_dir

# ============================================
# CONFIGURAÇÕES DE CORREÇÕES
# ============================================

APPLY_TIDE_CHECK = True
APPLY_IBE = False
APPLY_SLOPE = False

print("\nCorreções a aplicar:")
print(f"  Verificação de marés: {'SIM' if APPLY_TIDE_CHECK else 'NÃO'}")
print(f"  IBE (Inverse Barometer): {'SIM' if APPLY_IBE else 'NÃO'}")
print(f"  Inclinação (Slope): {'SIM' if APPLY_SLOPE else 'NÃO'}")

# ============================================
# LISTAR TILES
# ============================================

print("\n1. Listando tiles...")
tile_files = sorted(list(TILES_WINTER_DIR.glob("tile_*.h5")))
print(f"   Encontrados: {len(tile_files)} tiles")

if len(tile_files) == 0:
    print("\n✗ Nenhum tile encontrado!")
    print("   Execute primeiro: 08_create_tiles.py")
    sys.exit(1)

# ============================================
# VERIFICAR CORREÇÕES JÁ APLICADAS
# ============================================

print("\n2. Verificando correções já aplicadas no ATL06...")

first_tile = read_hdf5(tile_files[0])

tide_vars = ['tide_earth', 'tide_load', 'tide_ocean', 'tide_pole']
has_tide_vars = all(var in first_tile for var in tide_vars)
dac_var = 'dac' in first_tile

print(f"   Variáveis de marés: {'PRESENTES' if has_tide_vars else 'AUSENTES'}")
print(f"   Variável DAC: {'PRESENTE' if dac_var else 'AUSENTE'}")

if has_tide_vars:
    print("\n   ✓ ATL06 já contém correções de marés aplicadas pela NASA:")
    for var in tide_vars:
        if var in first_tile:
            print(f"      - {var}")

# ============================================
# PROCESSAR TILES
# ============================================

print("\n3. Processando tiles...")

n_tiles_processed = 0
n_tiles_error = 0
any_corrections_applied = False

for tile_file in tqdm(tile_files, desc="Aplicando correções"):
    
    try:
        # Ler tile
        tile_data = read_hdf5(tile_file)
        
        # Verificar se tem h_li
        if 'h_li' not in tile_data or len(tile_data.get('h_li', [])) == 0:
            continue
        
        # Iniciar com altura original
        h_corrected = tile_data['h_li'].copy()
        
        # Para este projeto, NÃO aplicamos correções adicionais
        # (ATL06 já vem com marés e DAC aplicados)
        
        # Salvar altura "corrigida" (igual à original)
        tile_data['h_li_corrected'] = h_corrected
        tile_data['corrections_applied'] = 'none'
        
        # Salvar tile atualizado
        write_hdf5(tile_file, tile_data)
        
        n_tiles_processed += 1
        
    except Exception as e:
        print(f"\n✗ Erro ao processar {tile_file.name}: {e}")
        n_tiles_error += 1
        continue

# ============================================
# RESUMO
# ============================================

print("\n" + "="*60)
print("RESUMO DA APLICAÇÃO DE CORREÇÕES")
print("="*60)
print(f"Tiles processados: {n_tiles_processed}/{len(tile_files)}")
print(f"Tiles com erro: {n_tiles_error}")

print("\nCorreções aplicadas:")
if APPLY_SLOPE:
    print("  ✓ Inclinação (Slope)")
    any_corrections_applied = True
else:
    print("  ✗ Inclinação (não aplicada)")

if APPLY_IBE:
    print("  ✓ IBE (Inverse Barometer)")
    any_corrections_applied = True
else:
    print("  ✗ IBE (não aplicada)")

print("\nCorreções do ATL06 original (já aplicadas pela NASA):")
if has_tide_vars:
    print("  ✓ Marés oceânicas (tide_ocean)")
    print("  ✓ Marés de carga (tide_load)")
    print("  ✓ Marés terrestres (tide_earth)")
    print("  ✓ Marés polares (tide_pole)")
else:
    print("  ⚠ Correções de marés não encontradas")

if dac_var:
    print("  ✓ DAC (Dynamic Atmospheric Correction)")

print("\nNOTA IMPORTANTE:")
print("  - Para análise de dh/dt, o mais importante é usar a MESMA")
print("    referência em todos os tempos (ex: sempre elipsoidal)")
print("  - Correções de marés e DAC já foram aplicadas pela NASA no ATL06")
print("  - IBE tem efeito pequeno (~1-2 cm) para gelo terrestre")
print("  - Correção de slope ideal requer DEM de alta resolução (REMA)")

print("="*60)

print("\n✓ Aplicação de correções concluída!")
print("\nPróximo passo: 10_calculate_dhdt.py (CORE do projeto)")

print("\nVARIÁVEL A USAR PARA dh/dt:")
if any_corrections_applied:
    print("  → h_li_corrected (com correções aplicadas)")
else:
    print("  → h_li_corrected (= h_li, sem correções adicionais)")