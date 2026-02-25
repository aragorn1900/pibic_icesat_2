"""
Verificação de integridade dos arquivos ATL06 baixados
"""

import h5py
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Importar configurações
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import RAW_DIR, LOGS_DIR

print("="*70)
print("VERIFICAÇÃO DE INTEGRIDADE - ATL06 DOWNLOADS")
print("="*70)

# ============================================
# 1. CONTAR ARQUIVOS
# ============================================

print("\n1. Contando arquivos baixados...")

atl06_files = list(RAW_DIR.glob("ATL06*.h5"))
n_files = len(atl06_files)

print(f"   Total de arquivos .h5: {n_files}")

if n_files == 0:
    print("\n❌ ERRO: Nenhum arquivo ATL06 encontrado!")
    print(f"   Procurado em: {RAW_DIR}")
    sys.exit(1)

# ============================================
# 2. CALCULAR TAMANHO TOTAL
# ============================================

print("\n2. Calculando tamanho total...")

total_size = 0
file_sizes = []

for f in atl06_files:
    size = f.stat().st_size
    total_size += size
    file_sizes.append(size)

total_size_gb = total_size / (1024**3)
avg_size_mb = np.mean(file_sizes) / (1024**2)
min_size_mb = np.min(file_sizes) / (1024**2)
max_size_mb = np.max(file_sizes) / (1024**2)

print(f"   Tamanho total: {total_size_gb:.2f} GB")
print(f"   Tamanho médio: {avg_size_mb:.1f} MB")
print(f"   Tamanho mínimo: {min_size_mb:.1f} MB")
print(f"   Tamanho máximo: {max_size_mb:.1f} MB")

# ============================================
# 3. VERIFICAR INTEGRIDADE
# ============================================

print("\n3. Verificando integridade dos arquivos...")
print("   (Isso pode demorar alguns minutos...)")

corrupted_files = []
valid_files = []
file_info = []

for fpath in tqdm(atl06_files, desc="Verificando"):
    
    try:
        # Tentar abrir arquivo
        with h5py.File(fpath, 'r') as f:
            
            # Verificar estrutura básica
            has_gt1l = 'gt1l' in f
            has_gt1r = 'gt1r' in f
            has_any_gt = any(k.startswith('gt') for k in f.keys())
            
            if not has_any_gt:
                corrupted_files.append((fpath.name, "No ground tracks"))
                continue
            
            # Tentar ler uma variável de teste
            if has_gt1l:
                test_data = f['gt1l/land_ice_segments/h_li']
                n_points = len(test_data)
            elif has_gt1r:
                test_data = f['gt1r/land_ice_segments/h_li']
                n_points = len(test_data)
            else:
                # Procurar primeiro ground track disponível
                for key in f.keys():
                    if key.startswith('gt'):
                        test_data = f[f'{key}/land_ice_segments/h_li']
                        n_points = len(test_data)
                        break
            
            # Arquivo válido
            valid_files.append(fpath.name)
            file_info.append({
                'name': fpath.name,
                'size_mb': fpath.stat().st_size / (1024**2),
                'n_points': n_points,
                'status': 'OK'
            })
            
    except Exception as e:
        corrupted_files.append((fpath.name, str(e)))
        file_info.append({
            'name': fpath.name,
            'size_mb': fpath.stat().st_size / (1024**2),
            'n_points': 0,
            'status': f'ERROR: {str(e)[:50]}'
        })

# ============================================
# 4. RESUMO
# ============================================

print("\n" + "="*70)
print("RESUMO DA VERIFICAÇÃO")
print("="*70)

print(f"\nTotal de arquivos: {n_files}")
print(f"Arquivos válidos: {len(valid_files)} ({100*len(valid_files)/n_files:.1f}%)")
print(f"Arquivos corrompidos: {len(corrupted_files)} ({100*len(corrupted_files)/n_files:.1f}%)")

print(f"\nTamanho total: {total_size_gb:.2f} GB")
print(f"Tamanho médio: {avg_size_mb:.1f} MB")

# Estatísticas de pontos
if file_info:
    total_points = sum(f['n_points'] for f in file_info if f['n_points'] > 0)
    print(f"\nPontos totais estimados: {total_points:,}")

# ============================================
# 5. ARQUIVOS CORROMPIDOS
# ============================================

if corrupted_files:
    print("\n" + "="*70)
    print("⚠️  ARQUIVOS CORROMPIDOS ENCONTRADOS")
    print("="*70)
    
    for fname, error in corrupted_files[:20]:  # Mostrar primeiros 20
        print(f"   - {fname}")
        print(f"     Erro: {error[:80]}")
    
    if len(corrupted_files) > 20:
        print(f"\n   ... e mais {len(corrupted_files)-20} arquivos")
    
    # Salvar lista completa
    corrupted_file = LOGS_DIR / 'corrupted_files.txt'
    with open(corrupted_file, 'w') as f:
        for fname, error in corrupted_files:
            f.write(f"{fname}\t{error}\n")
    
    print(f"\n   Lista completa salva em: {corrupted_file}")
    print("\n   RECOMENDAÇÃO: Re-baixar arquivos corrompidos")
    print("   Execute novamente: 01_download_atl06.py")

# ============================================
# 6. SALVAR RELATÓRIO
# ============================================

print("\n4. Salvando relatório de verificação...")

# Relatório CSV
import pandas as pd

df = pd.DataFrame(file_info)
report_file = LOGS_DIR / 'download_verification_report.csv'
df.to_csv(report_file, index=False)

print(f"   ✓ Relatório salvo: {report_file}")

# Relatório texto
report_txt = LOGS_DIR / 'download_verification_summary.txt'
with open(report_txt, 'w') as f:
    f.write("="*70 + "\n")
    f.write("RELATÓRIO DE VERIFICAÇÃO - ATL06 DOWNLOADS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Data da verificação: {pd.Timestamp.now()}\n\n")
    f.write(f"Total de arquivos: {n_files}\n")
    f.write(f"Arquivos válidos: {len(valid_files)} ({100*len(valid_files)/n_files:.1f}%)\n")
    f.write(f"Arquivos corrompidos: {len(corrupted_files)} ({100*len(corrupted_files)/n_files:.1f}%)\n\n")
    f.write(f"Tamanho total: {total_size_gb:.2f} GB\n")
    f.write(f"Tamanho médio: {avg_size_mb:.1f} MB\n")
    f.write(f"Tamanho mínimo: {min_size_mb:.1f} MB\n")
    f.write(f"Tamanho máximo: {max_size_mb:.1f} MB\n\n")
    
    if file_info:
        total_points = sum(f['n_points'] for f in file_info if f['n_points'] > 0)
        f.write(f"Pontos totais estimados: {total_points:,}\n\n")
    
    if corrupted_files:
        f.write("\n" + "="*70 + "\n")
        f.write("ARQUIVOS CORROMPIDOS:\n")
        f.write("="*70 + "\n\n")
        for fname, error in corrupted_files:
            f.write(f"{fname}\n  Erro: {error}\n\n")

print(f"   ✓ Resumo salvo: {report_txt}")

# ============================================
# 7. DECISÃO FINAL
# ============================================

print("\n" + "="*70)

if len(corrupted_files) == 0:
    print("✓ VERIFICAÇÃO COMPLETA - TODOS OS ARQUIVOS VÁLIDOS!")
    print("="*70)
    print("\nPróximo passo: 03_read_atl06.py")
    
elif len(corrupted_files) / n_files < 0.05:  # < 5% corrompidos
    print("⚠️  VERIFICAÇÃO COMPLETA COM AVISOS")
    print("="*70)
    print(f"\n{len(corrupted_files)} arquivo(s) corrompido(s) ({100*len(corrupted_files)/n_files:.1f}%)")
    print("\nOPÇÕES:")
    print("  1. Continuar processamento (arquivos válidos são suficientes)")
    print("  2. Re-executar Script 01 para baixar arquivos faltantes")
    print("\nRecomendação: CONTINUAR se < 5% corrompidos")
    
else:  # >= 5% corrompidos
    print("❌ VERIFICAÇÃO FALHOU - MUITOS ARQUIVOS CORROMPIDOS")
    print("="*70)
    print(f"\n{len(corrupted_files)} arquivo(s) corrompido(s) ({100*len(corrupted_files)/n_files:.1f}%)")
    print("\nRECOMENDAÇÃO: Re-executar Script 01 para re-baixar arquivos")
    print("\nNÃO prossiga para Script 03 até resolver arquivos corrompidos!")


print("="*70)
