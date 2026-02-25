"""
Leitura de arquivos ATL06 
"""

import numpy as np
import h5py
import sys
import os
from pathlib import Path
from tqdm import tqdm
from astropy.time import Time

# ============================================
# DETECTAR DIRETÓRIO AUTOMATICAMENTE
# ============================================

# Se rodar como script
if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).parent
else:
    # Se rodar no Spyder console
    SCRIPT_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter\scripts")

sys.path.insert(0, str(SCRIPT_DIR))

# Importar configurações
from config import (
    THWAITES_BBOX,
    RAW_DIR,
    PROCESSED_DIR,
    QUALITY_FILTERS,
    SEGMENT_FILTER
)

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def gps2dyr(time):
    """Converte GPS time para anos decimais"""
    try:
        time_obj = Time(time, format='gps')
        time_decimal = Time(time_obj, format='decimalyear').value
        return time_decimal
    except:
        # Fallback: conversão manual
        # GPS epoch: 1980-01-06 00:00:00
        gps_epoch = 315964800.0  # Unix timestamp da época GPS
        unix_time = time + gps_epoch
        
        # Converter para anos decimais (aproximado)
        seconds_per_year = 365.25 * 24 * 3600
        year = 1970 + (unix_time / seconds_per_year)
        return year


def segDifferenceFilter(dh_fit_dx, h_li, tol=2):
    """
    Filtro de diferença de segmentos
    Remove pontos com grandes diferenças entre segmentos adjacentes
    
    Baseado em: Smith et al. (2019) - ATL06 Algorithm
    """
    dAT = SEGMENT_FILTER['dAT']  # 20.0 metros
    tol = SEGMENT_FILTER['tolerance']  # 2.0 sigma

    if h_li.shape[0] < 3:
        mask = np.ones_like(h_li, dtype=bool)
        return mask

    # Expected elevation plus/minus
    EPplus  = h_li + dAT * dh_fit_dx
    EPminus = h_li - dAT * dh_fit_dx

    # Calcular diferenças
    segDiff = np.zeros_like(h_li)
    segDiff[0:-1] = np.abs(EPplus[0:-1] - h_li[1:])
    segDiff[1:] = np.maximum(segDiff[1:], np.abs(h_li[0:-1] - EPminus[1:]))

    # Máscara
    mask = segDiff < tol

    return mask


def track_type(time, lat, tmax=1):
    """
    Determina tracks ascendentes e descendentes
    """
    tracks = np.zeros(lat.shape)
    
    # Identificar onde latitude atinge máximo (mudança de direção)
    idx_max = np.argmax(np.abs(lat))
    tracks[0:idx_max] = 1

    i_asc = np.zeros(tracks.shape, dtype=bool)

    for track in np.unique(tracks):
        i_track, = np.where(track == tracks)

        if len(i_track) < 2:
            continue

        i_min = time[i_track].argmin()
        i_max = time[i_track].argmax()
        lat_diff = lat[i_track][i_max] - lat[i_track][i_min]

        # Ascendente se latitude aumenta com tempo
        if lat_diff > 0:
            i_asc[i_track] = True

    return i_asc, np.invert(i_asc)


# ============================================
# FUNÇÃO PRINCIPAL DE LEITURA
# ============================================

def read_atl06_file(ifile, bbox=None, apply_quality=True):
    """
    Lê arquivo ATL06 completo
        
    Parâmetros
    ----------
    ifile : str
        Caminho do arquivo ATL06
    bbox : tuple, opcional
        (lon_min, lon_max, lat_min, lat_max)
    apply_quality : bool
        Se True, aplica filtros de qualidade
    
    Retorna
    -------
    data_all : dict
        Dados de todos os ground tracks processados
    """
    
    # Ground tracks (6 beams do ICESat-2)
    groups = ['/gt1l', '/gt1r', '/gt2l', '/gt2r', '/gt3l', '/gt3r']
    
    data_all = {}
    
    # Loop sobre ground tracks
    for gt in groups:
        
        try:
            with h5py.File(ifile, 'r') as fi:
                
                # Verificar se GT existe
                if gt not in fi.keys():
                    continue
                
                # Caminho base
                base = gt + '/land_ice_segments'
                
                # Verificar se land_ice_segments existe
                if base not in fi:
                    continue
                
                # ============================================
                # VARIÁVEIS PRINCIPAIS
                # ============================================
                
                # Coordenadas
                lat = fi[base + '/latitude'][:]
                lon = fi[base + '/longitude'][:]
                
                # Altura
                h_li = fi[base + '/h_li'][:]
                s_li = fi[base + '/h_li_sigma'][:]
                
                # Tempo
                t_dt = fi[base + '/delta_time'][:]
                
                # Época de referência GPS
                if isinstance(fi['/ancillary_data/atlas_sdp_gps_epoch'], h5py.Dataset):
                    tref = fi['/ancillary_data/atlas_sdp_gps_epoch'][()]
                else:
                    tref = fi['/ancillary_data/atlas_sdp_gps_epoch'][0]
                
                # Quality flags
                flag = fi[base + '/atl06_quality_summary'][:]
                
                # ============================================
                # FIT STATISTICS
                # ============================================
                
                fit_stats = base + '/fit_statistics'
                
                # dh_fit_dx (inclinação da superfície)
                dh_fit_dx = fi[fit_stats + '/dh_fit_dx'][:]
                
                # Outras métricas de qualidade
                h_rb = fi[fit_stats + '/h_robust_sprd'][:]
                
                try:
                    s_fg = fi[fit_stats + '/signal_selection_source'][:]
                except:
                    s_fg = np.zeros_like(lat)
                
                try:
                    snr = fi[fit_stats + '/snr_significance'][:]
                except:
                    snr = np.zeros_like(lat)
                
                # ============================================
                # CORREÇÕES GEOFÍSICAS
                # ============================================
                
                geophys = base + '/geophysical'
                
                # Marés (já aplicadas no ATL06)
                tide_earth = fi[geophys + '/tide_earth'][:]
                tide_load = fi[geophys + '/tide_load'][:]
                tide_ocean = fi[geophys + '/tide_ocean'][:]
                tide_pole = fi[geophys + '/tide_pole'][:]
                
                # DAC (Dynamic Atmospheric Correction)
                dac = fi[geophys + '/dac'][:]
                
                # Snow confidence
                try:
                    f_sn = fi[geophys + '/bsnow_conf'][:]
                except:
                    f_sn = np.zeros_like(lat)
                
                # ============================================
                # ORBIT INFO
                # ============================================
                
                # RGT (Reference Ground Track)
                if isinstance(fi['/orbit_info/rgt'], h5py.Dataset):
                    rgt_val = fi['/orbit_info/rgt'][()]
                else:
                    rgt_val = fi['/orbit_info/rgt'][0]
                
                rgt = rgt_val * np.ones(len(lat))
                
                # Cycle number
                try:
                    if isinstance(fi['/orbit_info/cycle_number'], h5py.Dataset):
                        cycle_val = fi['/orbit_info/cycle_number'][()]
                    else:
                        cycle_val = fi['/orbit_info/cycle_number'][0]
                    cycle = cycle_val * np.ones(len(lat))
                except:
                    cycle = np.zeros_like(lat)
                
        except Exception as e:
            # print(f"✗ Erro ao ler {gt} de {Path(ifile).name}: {e}")
            continue
        
        # Verificar se tem dados
        if len(lat) == 0:
            continue
        
        # ============================================
        # APLICAR FILTROS
        # ============================================
        
        # Converter dh_fit_dx para float64 (evita problemas numéricos)
        dh_fit_dx = np.float64(dh_fit_dx)
        
        # Filtro 1: Bounding box
        if bbox is not None:
            lonmin, lonmax, latmin, latmax = bbox
            ibox = (
                (lon >= lonmin) & (lon <= lonmax) & 
                (lat >= latmin) & (lat <= latmax)
            )
        else:
            ibox = np.ones(lat.shape, dtype=bool)
        
        # Filtro 2: Segment difference filter
        mask_seg = segDifferenceFilter(dh_fit_dx, h_li, tol=2)
        
        # Filtro 3: Quality filter
        if apply_quality:
            h_min, h_max = QUALITY_FILTERS['height_range']
            
            mask_quality = (
                (flag <= QUALITY_FILTERS['atl06_quality_summary']) &  # Qualidade boa
                (s_li < QUALITY_FILTERS['h_li_sigma']) &              # Incerteza < 1m
                (h_rb < QUALITY_FILTERS['h_robust_sprd']) &           # Robust spread < 1m
                (h_li > h_min) & (h_li < h_max) &                     # Altura razoável
                mask_seg &                                             # Segment filter
                ibox                                                   # Dentro da região
            )
        else:
            mask_quality = ibox & mask_seg
        
        # Aplicar máscara
        if not np.any(mask_quality):
            continue
        
        # Filtrar arrays
        lat = lat[mask_quality]
        lon = lon[mask_quality]
        h_li = h_li[mask_quality]
        s_li = s_li[mask_quality]
        t_dt = t_dt[mask_quality]
        flag = flag[mask_quality]
        dh_fit_dx = dh_fit_dx[mask_quality]
        h_rb = h_rb[mask_quality]
        s_fg = s_fg[mask_quality]
        snr = snr[mask_quality]
        f_sn = f_sn[mask_quality]
        tide_earth = tide_earth[mask_quality]
        tide_load = tide_load[mask_quality]
        tide_ocean = tide_ocean[mask_quality]
        tide_pole = tide_pole[mask_quality]
        dac = dac[mask_quality]
        rgt = rgt[mask_quality]
        cycle = cycle[mask_quality]
        
        # Verificar se ainda tem dados após filtros
        if len(lat) == 0:
            continue
        
        # ============================================
        # PROCESSAR TEMPO
        # ============================================
        
        # GPS time (segundos desde época GPS)
        t_gps = t_dt + tref
        
        # Decimal year (anos decimais)
        t_year = gps2dyr(t_gps)
        
        # Determinar track type (asc/desc)
        i_asc, i_des = track_type(t_year, lat)
        
        # ============================================
        # ARMAZENAR DADOS
        # ============================================
        
        data_all[gt] = {
            # Coordenadas
            'latitude': lat,
            'longitude': lon,
            
            # Altura
            'h_li': h_li,
            'h_li_sigma': s_li,
            'dh_fit_dx': dh_fit_dx,
            
            # Tempo
            'delta_time': t_dt,
            't_gps': t_gps,
            't_year': t_year,
            
            # Quality
            'atl06_quality_summary': flag,
            'h_robust_sprd': h_rb,
            'signal_selection_source': s_fg,
            'snr_significance': snr,
            'bsnow_conf': f_sn,
            
            # Correções geofísicas
            'tide_earth': tide_earth,
            'tide_load': tide_load,
            'tide_ocean': tide_ocean,
            'tide_pole': tide_pole,
            'dac': dac,
            
            # Orbit info
            'rgt': rgt,
            'cycle': cycle,
            'track_type': i_asc.astype(int),  # 1=asc, 0=desc
            
            # Metadata
            'ground_track': gt
        }
    
    return data_all


# ============================================
# SCRIPT PRINCIPAL
# ============================================

if __name__ == '__main__':
    
    print("="*70)
    print("LEITURA DE ARQUIVOS ATL06 - BASEADO NO CAPTOOLKIT")
    print("Box:", THWAITES_BBOX['name'])
    print("="*70)
    
    # Listar arquivos
    print("\n1. Listando arquivos ATL06...")
    atl06_files = sorted(list(RAW_DIR.glob("ATL06*.h5")))
    print(f"   Encontrados: {len(atl06_files)} arquivos")
    
    if len(atl06_files) == 0:
        print("\n✗ Nenhum arquivo encontrado!")
        print(f"   Diretório: {RAW_DIR}")
        sys.exit(1)
    
    # Bounding box
    bbox = (
        THWAITES_BBOX['lon_min'],
        THWAITES_BBOX['lon_max'],
        THWAITES_BBOX['lat_min'],
        THWAITES_BBOX['lat_max']
    )
    
    print(f"\n2. Bounding box:")
    print(f"   Longitude: {bbox[0]}° a {bbox[1]}°")
    print(f"   Latitude: {bbox[2]}° a {bbox[3]}°")
    
    # Criar diretório de saída
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    
    # Processar arquivos
    print("\n3. Processando arquivos...")
    print("   (Isso vai demorar ~10-12 horas para box grande!)")
    
    n_files_processed = 0
    n_files_failed = 0
    n_total_points = 0
    n_ground_tracks = 0
    
    for ifile in tqdm(atl06_files, desc="Lendo ATL06"):
        
        try:
            # Ler arquivo
            data_all = read_atl06_file(str(ifile), bbox=bbox, apply_quality=True)
            
            # Verificar se tem dados
            if len(data_all) == 0:
                continue
            
            # Salvar cada ground track
            for gt, data in data_all.items():
                
                # Nome de saída
                # ATL06_20190615123456_12345678_006_01.h5 -> ATL06_...gt1l.h5
                base_name = ifile.stem + f'_{gt[1:]}.h5'  # Remove '/' do gt
                output_file = PROCESSED_DIR / base_name
                
                # Salvar HDF5
                with h5py.File(output_file, 'w') as fo:
                    for key, value in data.items():
                        if isinstance(value, np.ndarray):
                            fo.create_dataset(key, data=value, compression='gzip')
                        elif isinstance(value, str):
                            fo.attrs[key] = value
                        else:
                            try:
                                fo.attrs[key] = value
                            except:
                                pass
                
                n_total_points += len(data['latitude'])
                n_ground_tracks += 1
            
            n_files_processed += 1
            
        except Exception as e:
            n_files_failed += 1
            # print(f"\n✗ Erro: {ifile.name}: {e}")
            continue
    
    # ============================================
    # RESUMO
    # ============================================
    
    print("\n" + "="*70)
    print("RESUMO DA LEITURA")
    print("="*70)
    print(f"Arquivos ATL06 encontrados: {len(atl06_files)}")
    print(f"Arquivos processados: {n_files_processed}")
    print(f"Arquivos com erro: {n_files_failed}")
    print(f"Taxa de sucesso: {100*n_files_processed/len(atl06_files):.1f}%")
    
    print(f"\nGround tracks extraídos: {n_ground_tracks}")
    print(f"Total de pontos: {n_total_points:,}")
    
    # Contar arquivos de saída
    output_files = list(PROCESSED_DIR.glob("*.h5"))
    print(f"Arquivos salvos: {len(output_files)}")
    
    print(f"\nDiretório de saída: {PROCESSED_DIR}")
    print("="*70)
    
    print("\n✓ Leitura concluída!")

    print("\nPróximo passo: 04_filter_quality.py")
