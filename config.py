"""
Configuração para Análise ICESat-2 ATL06
Região: Amundsen Sea Embayment (West Antarctic Ice Sheet)
"""

from pathlib import Path
import numpy as np

# ============================================
# INFORMAÇÕES DO PROJETO
# ============================================

PROJECT_NAME = "Amundsen Sea Embayment - Winter dh/dt Analysis"
SENSOR = "ICESat-2 ATL06"
REGION = "Amundsen Sea Embayment, West Antarctic Ice Sheet"

# ============================================
# REGIÃO ESPACIAL - BOX COMPLETA
# ============================================

THWAITES_BBOX = {
    'name': 'Amundsen Sea Embayment',
    'lon_min': -135.0,  # Longitude oeste
    'lon_max': -85.0,   # Longitude leste
    'lat_min': -79.0,   # Latitude sul
    'lat_max': -72.0    # Latitude norte
}

# Estatísticas da box
BOX_STATS = {
    'lon_span': 50.0,        # graus
    'lat_span': 7.0,         # graus
    'approx_width': 1250,    # km
    'approx_height': 775,    # km
    'approx_area': 875000    # km²
}

# ============================================
# PERÍODO TEMPORAL
# ============================================

START_YEAR = 2019
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))

# Estação
SEASON = 'Winter Austral (JJA)'
WINTER_MONTHS = [6, 7, 8]  # Junho, Julho, Agosto

# ============================================
# PARÂMETROS DE QUALIDADE ATL06
# ============================================

QUALITY_FILTERS = {
    'atl06_quality_summary': 0,      # Apenas melhor qualidade
    'h_li_sigma': 1.0,                # Incerteza máxima (metros)
    'h_robust_sprd': 1.0,             # Spread máximo (metros)
    'height_range': (-500, 5000)      # Altura válida (metros)
}

# Filtro de diferença entre segmentos
SEGMENT_FILTER = {
    'dAT': 20.0,        # Distância along-track (metros)
    'tolerance': 2.0    # Tolerância em desvios padrão
}

# ============================================
# TILING ESPACIAL - AJUSTADO PARA BOX GRANDE
# ============================================

TILE_SIZE = 100000  # metros (100 km × 100 km)

# Tile area = 10,000 km² (área 4× maior que config anterior)
# Número estimado de tiles: ~90-100 tiles

# ============================================
# FITSEC - CÁLCULO DE dh/dt
# AJUSTADO PARA ÁREA GRANDE
# ============================================

FITSEC_PARAMS = {
    # SEARCH RADIUS: Aumentado para área maior
    'search_radius': 1500,      # metros (1.5 km)
    # Razão: Densidade de pontos pode variar, raio maior garante vizinhos
    
    # POLYNOMIAL ORDER: Quadrático
    'poly_order': 2,            
    # Razão: Captura curvatura topográfica, funciona bem
    
    # ITERAÇÕES: Padrão CAPTOOLKIT
    'max_iterations': 5,        
    # Razão: Área enorme = mais variabilidade = precisa limpeza rigorosa
    
    # SIGMA THRESHOLD: Padrão
    'sigma_threshold': 3,       
    # Razão: 3-sigma é padrão científico
    
    # MÍNIMO DE PONTOS: Aumentado
    'min_points': 15            
    # Razão: Com raio 1.5 km, exigir mais pontos para robustez
}

# Modelo de superfície
SURFACE_MODEL = {
    'type': 'polynomial',
    'equation': 'h(x,y,t) = a₀ + a₁x + a₂y + a₃t + a₄x² + a₅y² + a₆xy',
    'dh/dt_coefficient': 'a₃'
}

# ============================================
# GRIDDING - INTERPOLAÇÃO
# RESOLUÇÃO AJUSTADA PARA BOX GRANDE
# ============================================

# OPÇÃO 1: Resolução ALTA (Detalhada)
GRID_RESOLUTION_HIGH = 1500      # metros (1.5 km)
INTERP_RADIUS_HIGH = 7500        # metros (7.5 km)
GRID_CELLS_HIGH = '~390,000'     # células

# OPÇÃO 2: Resolução MÉDIA (Balanceada)
GRID_RESOLUTION_MED = 2000       # metros (2 km)
INTERP_RADIUS_MED = 10000        # metros (10 km)
GRID_CELLS_MED = '~220,000'      # células

# OPÇÃO 3: Resolução REGIONAL (Overview)
GRID_RESOLUTION_LOW = 3000       # metros (3 km)
INTERP_RADIUS_LOW = 15000        # metros (15 km)
GRID_CELLS_LOW = '~100,000'      # células

# CONFIGURAÇÃO ATIVA (ESCOLHER UMA)
# ============================================
# ALTERE AQUI PARA TROCAR A RESOLUÇÃO
# ============================================

GRID_RESOLUTION = GRID_RESOLUTION_MED   # 2 km
INTERP_RADIUS = INTERP_RADIUS_MED       # 10 km

# Para mudar resolução, descomente a linha desejada:
# GRID_RESOLUTION = GRID_RESOLUTION_HIGH  # Alta: 1.5 km
# INTERP_RADIUS = INTERP_RADIUS_HIGH
# 
# GRID_RESOLUTION = GRID_RESOLUTION_LOW   # Baixa: 3 km
# INTERP_RADIUS = INTERP_RADIUS_LOW

# Suavização (Gaussian smoothing)
SMOOTHING_SIGMA = 2.5  # Aumentado para box grande

# Outlier removal na grade
OUTLIER_THRESHOLD = 3  # n-sigma

# ============================================
# MÁSCARA DE GELO - MUITO RIGOROSA
# ============================================

USE_ICE_MASK = True

ICE_MASK_CONFIG = {
    'dataset': 'BedMachine Antarctica v3',
    'resolution': 500,  # metros
    
    # Tipos de superfície a MANTER
    'keep_types': [1, 2],  # 1=grounded ice, 2=floating ice
    
    # Buffer para oceano (remover pontos próximos)
    'ocean_buffer': 3000,  # metros (3 km)
    # Razão: Box tem MUITO oceano, buffer ajuda
    
    # Espessura mínima de gelo
    'min_ice_thickness': 10,  # metros
    # Razão: Remove gelo marinho fino e debris
    
    # Qualidade
    'quality': 'high'
}

# ============================================
# CORREÇÕES GEOFÍSICAS
# ============================================

APPLY_CORRECTIONS = {
    'tide': False,      # Já aplicado no ATL06
    'ibe': False,       # Efeito pequeno em gelo
    'slope': False,     # Simplificado (sem DEM externo)
    'gia': False,       # GIA - adicionar futuramente
    'firn': False       # Densidade firn - adicionar futuramente
}

# ============================================
# DIRETÓRIOS DO PROJETO
# ============================================

BASE_DIR = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")

# Estrutura de diretórios
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'atl06'
PROCESSED_DIR = DATA_DIR / 'processed'
TILES_DIR = DATA_DIR / 'tiles'
TILES_WINTER_DIR = DATA_DIR / 'tiles_winter'

# Resultados
RESULTS_DIR = BASE_DIR / 'results'
DHDT_DIR = RESULTS_DIR / 'dhdt'
DHDT_WINTER_DIR = RESULTS_DIR / 'dhdt_winter'
GRIDS_DIR = RESULTS_DIR / 'grids'
TIMESERIES_DIR = RESULTS_DIR / 'timeseries'

# Figuras
FIGURES_DIR = BASE_DIR / 'figures'
MAPS_DIR = FIGURES_DIR / 'maps'
PLOTS_DIR = FIGURES_DIR / 'plots'

# Logs
LOGS_DIR = BASE_DIR / 'logs'

# BedMachine
BEDMACHINE_DIR = DATA_DIR / 'bedmachine'

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, TILES_DIR, 
                  TILES_WINTER_DIR, RESULTS_DIR, DHDT_DIR, 
                  DHDT_WINTER_DIR, GRIDS_DIR, TIMESERIES_DIR,
                  FIGURES_DIR, MAPS_DIR, PLOTS_DIR, LOGS_DIR,
                  BEDMACHINE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ============================================
# PROJEÇÃO CARTOGRÁFICA
# ============================================

PROJECTION = {
    'name': 'Antarctic Polar Stereographic',
    'epsg': 3031,
    'proj4': '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
}

# ============================================
# CONVERSÃO DE TEMPO
# ============================================

# Época ICESat-2
ICESAT2_EPOCH = (2018, 1, 1, 0, 0, 0)

# Época Unix (para conversões)
UNIX_EPOCH = (1970, 1, 1, 0, 0, 0)

# GPS time (para ATL06)
GPS_EPOCH = (1980, 1, 6, 0, 0, 0)

# ============================================
# CONSTANTES FÍSICAS
# ============================================

CONSTANTS = {
    'gravity': 9.81,              # m/s² (aceleração da gravidade)
    'ice_density': 917,           # kg/m³ (densidade do gelo)
    'seawater_density': 1027,     # kg/m³ (densidade água do mar)
    'earth_radius': 6371000,      # metros (raio da Terra)
}

# ============================================
# PARÂMETROS DE VISUALIZAÇÃO
# ============================================

PLOT_PARAMS = {
    'dh/dt_range': (-3, 2),       # m/ano (range colorbar)
    'dh/dt_cmap': 'RdBu_r',       # colormap
    'figure_dpi': 300,            # DPI para salvar figuras
    'map_projection': 'south_polar_stereo'
}

# ============================================
# FUNÇÃO AUXILIAR: Trocar Resolução
# ============================================

def set_resolution(level='medium'):
    """
    Troca a resolução da grade facilmente
    
    Parameters:
    -----------
    level : str
        'high' (1.5 km), 'medium' (2 km), 'low' (3 km)
    """
    global GRID_RESOLUTION, INTERP_RADIUS
    
    if level == 'high':
        GRID_RESOLUTION = GRID_RESOLUTION_HIGH
        INTERP_RADIUS = INTERP_RADIUS_HIGH
        print(f"✓ Resolução ALTA: {GRID_RESOLUTION}m, ~390k células")
    elif level == 'medium':
        GRID_RESOLUTION = GRID_RESOLUTION_MED
        INTERP_RADIUS = INTERP_RADIUS_MED
        print(f"✓ Resolução MÉDIA: {GRID_RESOLUTION}m, ~220k células")
    elif level == 'low':
        GRID_RESOLUTION = GRID_RESOLUTION_LOW
        INTERP_RADIUS = INTERP_RADIUS_LOW
        print(f"✓ Resolução BAIXA: {GRID_RESOLUTION}m, ~100k células")
    else:
        print("❌ Opções: 'high', 'medium', 'low'")

# ============================================
# EXPORT
# ============================================

__all__ = [
    'THWAITES_BBOX',
    'WINTER_MONTHS',
    'QUALITY_FILTERS',
    'SEGMENT_FILTER',
    'TILE_SIZE',
    'FITSEC_PARAMS',
    'GRID_RESOLUTION',
    'INTERP_RADIUS',
    'SMOOTHING_SIGMA',
    'ICE_MASK_CONFIG',
    'BASE_DIR',
    'DATA_DIR',
    'RESULTS_DIR',
    'FIGURES_DIR',
    'LOGS_DIR',
    'PROJECTION',
    'METADATA',
    'set_resolution'

]
