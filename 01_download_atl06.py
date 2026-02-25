"""
Download de granules ICESat-2 ATL06
Versão 2.0 - Com checkpoint e retry para box grande
"""

import earthaccess
import sys
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import time
import shutil

# Importar configurações
sys.path.insert(0, r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")
from config import (
    THWAITES_BBOX, 
    START_YEAR, 
    END_YEAR, 
    WINTER_MONTHS,
    LOGS_DIR,
    RAW_DIR
)

print("="*70)
print("DOWNLOAD ICESat-2 ATL06 - VERSÃO ROBUSTA")
print(f"Região: {THWAITES_BBOX['name']}")
print(f"Box: {THWAITES_BBOX['lon_min']}° a {THWAITES_BBOX['lon_max']}° (lon)")
print(f"     {THWAITES_BBOX['lat_min']}° a {THWAITES_BBOX['lat_max']}° (lat)")
print(f"Área: ~875,000 km²")
print("="*70)

# ============================================
# CHECKPOINT SYSTEM
# ============================================

CHECKPOINT_FILE = LOGS_DIR / 'download_checkpoint.json'

def save_checkpoint(downloaded_files, total_granules):
    """Salva progresso do download"""
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'downloaded': len(downloaded_files),
        'total': total_granules,
        'files': downloaded_files,
        'percentage': 100 * len(downloaded_files) / total_granules if total_granules > 0 else 0
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    return checkpoint

def load_checkpoint():
    """Carrega progresso anterior"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

# ============================================
# CONFIGURAÇÃO
# ============================================

print("\n1. Configuração da busca...")
print(f"   Período: {START_YEAR}-{END_YEAR}")
print(f"   Meses: {WINTER_MONTHS} (inverno JJA)")

# Verificar espaço em disco
total, used, free = shutil.disk_usage("D:\\")
free_gb = free // (2**30)

print(f"\n⚠️  VERIFICAÇÃO DE ESPAÇO:")
print(f"   Espaço livre: {free_gb} GB")
print(f"   Necessário: ~150 GB")

if free_gb < 150:
    print(f"\n❌ ERRO: Espaço insuficiente!")
    print(f"   Precisa de pelo menos 150 GB")
    print(f"   Você tem apenas {free_gb} GB")
    response = input("\n   Continuar mesmo assim? (s/n): ")
    if response.lower() != 's':
        sys.exit(1)
else:
    print(f"   ✓ Espaço suficiente!")

# ============================================
# AUTENTICAÇÃO
# ============================================

print("\n2. Autenticando na NASA Earthdata...")
try:
    auth = earthaccess.login()
    print("   ✓ Autenticação bem-sucedida!")
except Exception as e:
    print(f"\n❌ ERRO na autenticação: {e}")
    print("\nCrie uma conta em: https://urs.earthdata.nasa.gov/users/new")
    sys.exit(1)

# ============================================
# BUSCAR GRANULES
# ============================================

print("\n3. Buscando granules ATL06...")
print("   (Isso pode demorar 5-10 minutos para box grande...)")

try:
    results = earthaccess.search_data(
        short_name='ATL06',
        version='006',
        bounding_box=(
            THWAITES_BBOX['lon_min'],
            THWAITES_BBOX['lat_min'],
            THWAITES_BBOX['lon_max'],
            THWAITES_BBOX['lat_max']
        ),
        temporal=(f'{START_YEAR}-01-01', f'{END_YEAR}-12-31')
    )
    
    print(f"   ✓ Busca concluída!")
    print(f"   Total de granules encontrados: {len(results)}")
    
except Exception as e:
    print(f"\n❌ ERRO na busca: {e}")
    sys.exit(1)

if len(results) == 0:
    print("\n❌ Nenhum granule encontrado para esta região/período!")
    sys.exit(1)

# ============================================
# FILTRAR POR MÊS (INVERNO)
# ============================================

print("\n4. Filtrando por inverno austral (JJA)...")

winter_results = []
for granule in results:
    try:
        # Extrair data do nome do arquivo
        filename = granule['umm']['RelatedUrls'][0]['URL'].split('/')[-1]
        # ATL06_20190615123456_12345678_006_01.h5
        date_str = filename.split('_')[1][:8]  # YYYYMMDD
        month = int(date_str[4:6])
        
        if month in WINTER_MONTHS:
            winter_results.append(granule)
    except:
        continue

print(f"   ✓ Granules de inverno: {len(winter_results)}")

if len(winter_results) == 0:
    print("\n❌ Nenhum granule de inverno encontrado!")
    sys.exit(1)

# Salvar lista de granules
granules_file = LOGS_DIR / 'granules_list.json'
granule_names = [g['umm']['RelatedUrls'][0]['URL'].split('/')[-1] 
                 for g in winter_results]
with open(granules_file, 'w') as f:
    json.dump(granule_names, f, indent=2)

print(f"   Lista salva em: {granules_file}")

# ============================================
# VERIFICAR CHECKPOINT
# ============================================

print("\n5. Verificando downloads anteriores...")

checkpoint = load_checkpoint()
downloaded_files = set()

if checkpoint:
    print(f"   ✓ Checkpoint encontrado!")
    print(f"   Progresso anterior: {checkpoint['percentage']:.1f}%")
    print(f"   Arquivos baixados: {checkpoint['downloaded']}/{checkpoint['total']}")
    
    # Verificar quais arquivos ainda existem
    for fname in checkpoint['files']:
        fpath = RAW_DIR / fname
        if fpath.exists():
            downloaded_files.add(fname)
    
    print(f"   Arquivos válidos: {len(downloaded_files)}")
    
    # Confirmar retomada
    response = input("\n   Retomar download anterior? (s/n): ")
    if response.lower() != 's':
        downloaded_files = set()
        print("   Reiniciando download do zero...")
else:
    print("   Nenhum checkpoint encontrado. Iniciando do zero...")

# ============================================
# ESTIMAR TEMPO
# ============================================

print("\n6. Estimativas de download...")

total_granules = len(winter_results)
remaining = total_granules - len(downloaded_files)

# Assumir média de 100 MB por granule
avg_size_mb = 100
total_size_gb = (total_granules * avg_size_mb) / 1024
remaining_size_gb = (remaining * avg_size_mb) / 1024

print(f"   Total de granules: {total_granules}")
print(f"   Já baixados: {len(downloaded_files)}")
print(f"   Restantes: {remaining}")
print(f"   Tamanho estimado total: ~{total_size_gb:.1f} GB")
print(f"   Tamanho restante: ~{remaining_size_gb:.1f} GB")

# Estimar tempo (assumir 5 MB/s)
speed_mbs = 5  # MB/s (conservador)
time_hours = (remaining_size_gb * 1024) / (speed_mbs * 3600)
time_days = time_hours / 24

print(f"\n   ⏱️  Tempo estimado (a {speed_mbs} MB/s):")
print(f"      {time_hours:.1f} horas ({time_days:.1f} dias)")

# Confirmar antes de começar
print(f"\n{'='*70}")
print("⚠️  CONFIRMAÇÃO FINAL:")
print(f"   Você está prestes a baixar ~{total_size_gb:.0f} GB")
print(f"   Tempo estimado: ~{time_days:.1f} dias")
print(f"   Arquivos: {total_granules} granules")
print(f"{'='*70}")

response = input("\n   Iniciar download? (s/n): ")
if response.lower() != 's':
    print("\n❌ Download cancelado pelo usuário")
    sys.exit(0)

# ============================================
# DOWNLOAD COM RETRY
# ============================================

print("\n7. Iniciando download...")
print(f"   Diretório: {RAW_DIR}")

# Criar pasta se não existe
RAW_DIR.mkdir(exist_ok=True, parents=True)

# Filtrar granules já baixados
granules_to_download = [
    g for g in winter_results 
    if g['umm']['RelatedUrls'][0]['URL'].split('/')[-1] not in downloaded_files
]

print(f"\n   Baixando {len(granules_to_download)} granules...")

# Log de progresso
log_file = LOGS_DIR / f'download_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def log_message(msg):
    """Escreve no log e na tela"""
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")

# Baixar com retry
MAX_RETRIES = 3
downloaded_count = len(downloaded_files)
failed_granules = []

for i, granule in enumerate(tqdm(granules_to_download, desc="Downloading")):
    
    filename = granule['umm']['RelatedUrls'][0]['URL'].split('/')[-1]
    
    # Tentar baixar com retry
    success = False
    for attempt in range(MAX_RETRIES):
        try:
            files = earthaccess.download(
                [granule],
                str(RAW_DIR)
            )
            
            if files and len(files) > 0:
                downloaded_files.add(filename)
                downloaded_count += 1
                success = True
                
                # Salvar checkpoint a cada 10 arquivos
                if downloaded_count % 10 == 0:
                    save_checkpoint(list(downloaded_files), total_granules)
                
                break  # Sucesso, sair do loop de retry
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                log_message(f"   ⚠️  Erro no arquivo {filename}, tentativa {attempt+1}/{MAX_RETRIES}")
                time.sleep(5)  # Esperar antes de tentar novamente
            else:
                log_message(f"   ❌ Falha após {MAX_RETRIES} tentativas: {filename}")
                failed_granules.append(filename)
    
    # Log de progresso a cada 50 arquivos
    if (i + 1) % 50 == 0:
        progress = 100 * downloaded_count / total_granules
        log_message(f"\n   Progresso: {downloaded_count}/{total_granules} ({progress:.1f}%)")

# Salvar checkpoint final
checkpoint = save_checkpoint(list(downloaded_files), total_granules)

# ============================================
# RESUMO FINAL
# ============================================

print("\n" + "="*70)
print("RESUMO DO DOWNLOAD")
print("="*70)
print(f"Total de granules: {total_granules}")
print(f"Baixados com sucesso: {len(downloaded_files)}")
print(f"Falharam: {len(failed_granules)}")
print(f"Taxa de sucesso: {100*len(downloaded_files)/total_granules:.1f}%")

if failed_granules:
    print(f"\n⚠️  Arquivos que falharam:")
    for fname in failed_granules[:10]:  # Mostrar primeiros 10
        print(f"   - {fname}")
    if len(failed_granules) > 10:
        print(f"   ... e mais {len(failed_granules)-10} arquivos")
    
    # Salvar lista de falhas
    failed_file = LOGS_DIR / 'failed_downloads.txt'
    with open(failed_file, 'w') as f:
        f.write("\n".join(failed_granules))
    print(f"\n   Lista completa salva em: {failed_file}")

# Tamanho total baixado
try:
    total_size = sum(f.stat().st_size for f in RAW_DIR.glob("*.h5"))
    total_size_gb = total_size / (1024**3)
    print(f"\nTamanho total baixado: {total_size_gb:.2f} GB")
except:
    print(f"\nNão foi possível calcular tamanho total")

print(f"Localização: {RAW_DIR}")
print(f"Log: {log_file}")
print(f"Checkpoint: {CHECKPOINT_FILE}")

print("\n" + "="*70)

if len(downloaded_files) == total_granules:
    print("✓ DOWNLOAD COMPLETO COM SUCESSO!")
elif len(downloaded_files) > 0:
    print("⚠️  Download parcial - alguns arquivos falharam")
    print("   Você pode re-executar o script para tentar baixar os que falharam")
else:
    print("❌ Download falhou completamente")

print("="*70)

print("\nPróximo passo: 02_verify_download.py")