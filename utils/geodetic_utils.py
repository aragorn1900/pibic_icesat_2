"""
Funções geodésicas e transformações de coordenadas
"""

import numpy as np
import pyproj
from datetime import datetime, timedelta

# Sistemas de referência
WGS84 = pyproj.CRS("EPSG:4326")
ANTARCTIC_STEREO = pyproj.CRS("EPSG:3031")

TRANSFORMER_TO_STEREO = pyproj.Transformer.from_crs(WGS84, ANTARCTIC_STEREO, always_xy=True)
TRANSFORMER_TO_WGS84 = pyproj.Transformer.from_crs(ANTARCTIC_STEREO, WGS84, always_xy=True)


def lonlat_to_xy(lon, lat):
    """Converte lon/lat para coordenadas polares estereográficas"""
    x, y = TRANSFORMER_TO_STEREO.transform(lon, lat)
    return x, y


def xy_to_lonlat(x, y):
    """Converte coordenadas polares para lon/lat"""
    lon, lat = TRANSFORMER_TO_WGS84.transform(x, y)
    return lon, lat


def delta_time_to_datetime(delta_time, epoch='2018-01-01'):
    """
    Converte delta_time (segundos desde epoch) para datetime
    
    ATL06 usa segundos desde 2018-01-01 12:00:00 UTC
    
    Parâmetros
    ----------
    delta_time : array-like
        Segundos desde epoch
    epoch : str
        Data de referência (default: '2018-01-01')
    
    Retorna
    -------
    datetimes : array de datetime
    """
    
    # Epoch ICESat-2: 2018-01-01 12:00:00 UTC
    base = datetime(2018, 1, 1, 12, 0, 0)
    
    # Converter para array numpy se necessário
    delta_time = np.atleast_1d(delta_time)
    
    # Calcular datetimes
    datetimes = np.array([base + timedelta(seconds=float(dt)) for dt in delta_time])
    
    return datetimes


def datetime_to_decimal_year(datetimes):
    """
    Converte datetime para ano decimal
    
    Parâmetros
    ----------
    datetimes : array de datetime
        Datas
    
    Retorna
    -------
    decimal_years : array
        Anos decimais (ex: 2020.5 = meio de 2020)
    """
    
    datetimes = np.atleast_1d(datetimes)
    
    decimal_years = np.zeros(len(datetimes))
    
    for i, dt in enumerate(datetimes):
        year = dt.year
        start_of_year = datetime(year, 1, 1)
        start_of_next_year = datetime(year + 1, 1, 1)
        
        year_duration = (start_of_next_year - start_of_year).total_seconds()
        elapsed = (dt - start_of_year).total_seconds()
        
        decimal_years[i] = year + (elapsed / year_duration)
    
    return decimal_years


def calculate_distance(lon1, lat1, lon2, lat2):
    """
    Calcula distância entre pontos (haversine)
    
    Retorna
    -------
    distance : float ou array
        Distância em metros
    """
    
    # Raio da Terra (m)
    R = 6371000
    
    # Converter para radianos
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance = R * c
    
    return distance


def calculate_azimuth(lon1, lat1, lon2, lat2):
    """
    Calcula azimute entre dois pontos
    
    Retorna
    -------
    azimuth : float ou array
        Azimute em graus (0-360)
    """
    
    # Converter para radianos
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    azimuth = np.degrees(np.arctan2(y, x))
    azimuth = (azimuth + 360) % 360  # Normalizar 0-360
    
    return azimuth