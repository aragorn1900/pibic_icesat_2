"""
Utilitários para filtragem de dados
Versão 2.1 - Corrigido para tratar escalares
"""

import numpy as np
import sys
from pathlib import Path


# ============================================
# FILTROS DE QUALIDADE
# ============================================

def filter_quality(data, max_quality=0, max_sigma=1.0, height_range=(-500, 5000)):
    """
    Aplica filtros de qualidade aos dados ATL06
    
    CORRIGIDO: Trata escalares e arrays de forma robusta
    
    Parâmetros
    ----------
    data : dict
        Dicionário com dados ATL06
    max_quality : int
        Máximo valor de atl06_quality_summary (0 = melhor)
    max_sigma : float
        Máxima incerteza em h_li_sigma (metros)
    height_range : tuple
        (min, max) altura válida (metros)
    
    Retorna
    -------
    mask : array booleano
        True = manter, False = remover
    """
    
    # VERIFICAR SE TEM DADOS VÁLIDOS
    if 'latitude' not in data:
        return np.array([], dtype=bool)
    
    # TRATAR ESCALARES VS ARRAYS
    lat = np.atleast_1d(data['latitude'])
    
    # Se latitude é escalar único, arquivo tem problema
    if lat.shape == (1,) and not isinstance(data.get('latitude'), np.ndarray):
        # É um escalar disfarçado de array - retornar máscara rejeitando
        return np.array([False], dtype=bool)
    
    n_points = len(lat)
    
    # Se não tem pontos, retornar máscara vazia
    if n_points == 0:
        return np.array([], dtype=bool)
    
    # Criar máscara inicial (todos True)
    mask = np.ones(n_points, dtype=bool)
    
    # Filtro 1: Quality summary
    if 'atl06_quality_summary' in data:
        qual = np.atleast_1d(data['atl06_quality_summary'])
        if len(qual) == n_points:
            mask &= (qual <= max_quality)
        elif len(qual) == 1:
            # Escalar aplicado a todos os pontos
            mask &= (qual[0] <= max_quality)
    
    # Filtro 2: Incerteza (sigma)
    if 'h_li_sigma' in data:
        sigma = np.atleast_1d(data['h_li_sigma'])
        if len(sigma) == n_points:
            mask &= (sigma < max_sigma)
            mask &= (sigma > 0)
        elif len(sigma) == 1:
            mask &= (sigma[0] < max_sigma)
            mask &= (sigma[0] > 0)
    
    # Filtro 3: Robust spread
    if 'h_robust_sprd' in data:
        sprd = np.atleast_1d(data['h_robust_sprd'])
        if len(sprd) == n_points:
            mask &= (sprd < max_sigma)
            mask &= (sprd > 0)
        elif len(sprd) == 1:
            mask &= (sprd[0] < max_sigma)
            mask &= (sprd[0] > 0)
    
    # Filtro 4: Altura válida
    if 'h_li' in data:
        h = np.atleast_1d(data['h_li'])
        if len(h) == n_points:
            h_min, h_max = height_range
            mask &= (h > h_min)
            mask &= (h < h_max)
            mask &= ~np.isnan(h)
        elif len(h) == 1:
            h_min, h_max = height_range
            mask &= (h[0] > h_min)
            mask &= (h[0] < h_max)
            mask &= ~np.isnan(h[0])
    
    # Filtro 5: Coordenadas válidas
    if 'longitude' in data:
        lon = np.atleast_1d(data['longitude'])
        if len(lon) == n_points:
            mask &= ~np.isnan(lon)
        elif len(lon) == 1:
            mask &= ~np.isnan(lon[0])
    
    # Latitude já foi verificada
    mask &= ~np.isnan(lat)
    
    # Filtro 6: Tempo válido
    if 't_year' in data:
        t = np.atleast_1d(data['t_year'])
        if len(t) == n_points:
            mask &= ~np.isnan(t)
            mask &= (t > 2018)
            mask &= (t < 2030)
        elif len(t) == 1:
            mask &= ~np.isnan(t[0])
            mask &= (t[0] > 2018)
            mask &= (t[0] < 2030)
    
    return mask


# ============================================
# FILTROS ESPACIAIS
# ============================================

def filter_region(data, bbox):
    """
    Filtra dados por região (bounding box)
    
    Parâmetros
    ----------
    data : dict
        Dicionário com dados
    bbox : dict ou tuple
        Se dict: {'lon_min', 'lon_max', 'lat_min', 'lat_max'}
        Se tuple: (lon_min, lon_max, lat_min, lat_max)
    
    Retorna
    -------
    mask : array booleano
        True = dentro da região
    """
    # Extrair limites
    if isinstance(bbox, dict):
        lon_min = bbox['lon_min']
        lon_max = bbox['lon_max']
        lat_min = bbox['lat_min']
        lat_max = bbox['lat_max']
    else:
        lon_min, lon_max, lat_min, lat_max = bbox
    
    # Verificar se tem dados
    if 'longitude' not in data or 'latitude' not in data:
        return np.array([], dtype=bool)
    
    # Criar arrays
    lon = np.atleast_1d(data['longitude'])
    lat = np.atleast_1d(data['latitude'])
    
    # Verificar tamanhos
    if len(lon) == 0 or len(lat) == 0:
        return np.array([], dtype=bool)
    
    if len(lon) != len(lat):
        # Tamanhos inconsistentes
        return np.zeros(max(len(lon), len(lat)), dtype=bool)
    
    # Criar máscara
    mask = (
        (lon >= lon_min) & (lon <= lon_max) &
        (lat >= lat_min) & (lat <= lat_max)
    )
    
    return mask


def filter_temporal(data, season_months=None, year_range=None):
    """
    Filtra dados por período temporal
    
    Parâmetros
    ----------
    data : dict
        Dicionário com dados (deve ter 't_year')
    season_months : list, opcional
        Lista de meses (1-12) para manter
        Ex: [6, 7, 8] = inverno austral (JJA)
    year_range : tuple, opcional
        (ano_min, ano_max) para filtrar
    
    Retorna
    -------
    mask : array booleano
        True = manter
    """
    if 'latitude' not in data:
        return np.array([], dtype=bool)
    
    n_points = len(np.atleast_1d(data['latitude']))
    mask = np.ones(n_points, dtype=bool)
    
    if 't_year' not in data:
        return mask
    
    t_year = np.atleast_1d(data['t_year'])
    
    if len(t_year) != n_points:
        # Tamanho inconsistente
        return mask
    
    # Filtro por meses (se especificado)
    if season_months is not None:
        # Extrair mês da parte decimal do ano
        year_frac = t_year - np.floor(t_year)
        month = np.floor(year_frac * 12) + 1
        month = np.clip(month, 1, 12).astype(int)
        
        # Criar máscara de meses
        month_mask = np.zeros(n_points, dtype=bool)
        for m in season_months:
            month_mask |= (month == m)
        
        mask &= month_mask
    
    # Filtro por range de anos
    if year_range is not None:
        year_min, year_max = year_range
        mask &= (t_year >= year_min)
        mask &= (t_year <= year_max)
    
    return mask


# ============================================
# APLICAÇÃO DE MÁSCARAS
# ============================================

def apply_mask(data, mask):
    """
    Aplica máscara booleana aos dados
    
    CORRIGIDO: Trata escalares e arrays de forma robusta
    
    Parâmetros
    ----------
    data : dict
        Dicionário com dados
    mask : array booleano
        Máscara (True = manter)
    
    Retorna
    -------
    filtered_data : dict
        Dados filtrados
    """
    filtered_data = {}
    
    # Se máscara está vazia, retornar dict vazio
    if len(mask) == 0:
        return filtered_data
    
    for key, value in data.items():
        # Verificar se é array numpy
        if isinstance(value, np.ndarray):
            # Converter para array 1D se necessário
            value_1d = np.atleast_1d(value)
            
            # Verificar se tamanho corresponde à máscara
            if len(value_1d) == len(mask):
                # Aplicar máscara
                filtered_value = value_1d[mask]
                
                # Restaurar shape original se era escalar
                if value.shape == ():
                    # Era escalar, manter como escalar se só 1 elemento
                    if len(filtered_value) == 1:
                        filtered_data[key] = filtered_value[0]
                    else:
                        filtered_data[key] = filtered_value
                else:
                    filtered_data[key] = filtered_value
            else:
                # Tamanho diferente - manter como está (metadata)
                filtered_data[key] = value
        else:
            # Não é array - metadata, escalar, ou string - manter
            filtered_data[key] = value
    
    return filtered_data
def filter_by_month(data, months, time_key='t_year'):
    """
    Filtra dados por mês
    Alias para filter_temporal com season_months
    
    Parâmetros
    ----------
    data : dict
        Dicionário com dados
    months : list
        Lista de meses (1-12). Ex: [6, 7, 8] = inverno austral JJA
    time_key : str
        Variável de tempo a usar ('t_year', 'month', 'delta_time')
    
    Retorna
    -------
    mask : array booleano
        True = manter
    """
    # Se já tem variável 'month' calculada, usar diretamente
    if 'month' in data:
        month_arr = np.atleast_1d(data['month'])
        n_points = len(np.atleast_1d(data.get('latitude', [])))
        
        if len(month_arr) == n_points and n_points > 0:
            mask = np.zeros(n_points, dtype=bool)
            for m in months:
                mask |= (month_arr == m)
            return mask
    
    # Caso contrário, usar filter_temporal
    return filter_temporal(data, season_months=months)

def combine_masks(*masks, operation='and'):
    """
    Combina múltiplas máscaras
    
    Parâmetros
    ----------
    *masks : arrays booleanos
        Múltiplas máscaras
    operation : str
        'and' (intersection) ou 'or' (union)
    
    Retorna
    -------
    combined : array booleano
    """
    if len(masks) == 0:
        return np.array([], dtype=bool)
    
    if len(masks) == 1:
        return masks[0]
    
    # Verificar que todas têm mesmo tamanho
    sizes = [len(m) for m in masks]
    if len(set(sizes)) > 1:
        raise ValueError(f"Máscaras têm tamanhos diferentes: {sizes}")
    
    combined = masks[0].copy()
    
    for mask in masks[1:]:
        if operation == 'and':
            combined &= mask
        elif operation == 'or':
            combined |= mask
        else:
            raise ValueError(f"Operação desconhecida: {operation}")
    
    return combined