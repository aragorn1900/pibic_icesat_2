"""
Funções de interpolação espacial
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter


def interpolate_gaussian(x, y, z, grid_x, grid_y, search_radius=5000, sigma=None):
    """
    Interpolação usando kernel Gaussiano
    
      
    Parâmetros
    ----------
    x, y : array
        Coordenadas dos dados (metros)
    z : array
        Valores a interpolar (ex: dh/dt)
    grid_x, grid_y : 2D arrays
        Grade regular de saída
    search_radius : float
        Raio de busca em metros (default: 5000)
    sigma : float, opcional
        Desvio padrão do kernel Gaussiano
        Se None, usa sigma = search_radius / 3
    
    Retorna
    -------
    grid_z : 2D array
        Valores interpolados na grade
    grid_std : 2D array
        Desvio padrão em cada célula
    grid_count : 2D array
        Número de pontos usados em cada célula
    """
    
    if sigma is None:
        sigma = search_radius / 3.0
    
    # Criar KDTree para busca eficiente
    tree = cKDTree(np.column_stack([x, y]))
    
    # Inicializar grades de saída
    grid_z = np.full(grid_x.shape, np.nan)
    grid_std = np.full(grid_x.shape, np.nan)
    grid_count = np.zeros(grid_x.shape, dtype=int)
    
    # Loop sobre cada ponto da grade
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            
            # Coordenadas do ponto da grade
            xg = grid_x[i, j]
            yg = grid_y[i, j]
            
            # Buscar pontos dentro do raio
            indices = tree.query_ball_point([xg, yg], search_radius)
            
            if len(indices) < 3:
                # Mínimo de 3 pontos para interpolar
                continue
            
            # Dados vizinhos
            x_neigh = x[indices]
            y_neigh = y[indices]
            z_neigh = z[indices]
            
            # Calcular distâncias
            distances = np.sqrt((x_neigh - xg)**2 + (y_neigh - yg)**2)
            
            # Pesos Gaussianos
            weights = np.exp(-0.5 * (distances / sigma)**2)
            weights /= weights.sum()  # Normalizar
            
            # Média ponderada
            grid_z[i, j] = np.sum(weights * z_neigh)
            
            # Desvio padrão ponderado
            variance = np.sum(weights * (z_neigh - grid_z[i, j])**2)
            grid_std[i, j] = np.sqrt(variance)
            
            # Contagem
            grid_count[i, j] = len(indices)
    
    return grid_z, grid_std, grid_count


def interpolate_median(x, y, z, grid_x, grid_y, search_radius=5000):
    """
    Interpolação usando mediana dos vizinhos
    
    Equivalente a: interpmed.py do CAPTOOLKIT
    Mais robusto a outliers que média
    
    Parâmetros
    ----------
    x, y : array
        Coordenadas dos dados
    z : array
        Valores a interpolar
    grid_x, grid_y : 2D arrays
        Grade regular de saída
    search_radius : float
        Raio de busca em metros
    
    Retorna
    -------
    grid_z : 2D array
        Valores interpolados (mediana)
    grid_mad : 2D array
        MAD (Median Absolute Deviation)
    grid_count : 2D array
        Número de pontos
    """
    
    # KDTree
    tree = cKDTree(np.column_stack([x, y]))
    
    # Grades de saída
    grid_z = np.full(grid_x.shape, np.nan)
    grid_mad = np.full(grid_x.shape, np.nan)
    grid_count = np.zeros(grid_x.shape, dtype=int)
    
    # Loop sobre grade
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            
            xg = grid_x[i, j]
            yg = grid_y[i, j]
            
            # Buscar vizinhos
            indices = tree.query_ball_point([xg, yg], search_radius)
            
            if len(indices) < 3:
                continue
            
            z_neigh = z[indices]
            
            # Mediana
            grid_z[i, j] = np.median(z_neigh)
            
            # MAD (Median Absolute Deviation)
            mad = np.median(np.abs(z_neigh - grid_z[i, j]))
            grid_mad[i, j] = mad
            
            # Contagem
            grid_count[i, j] = len(indices)
    
    return grid_z, grid_mad, grid_count


def interpolate_idw(x, y, z, grid_x, grid_y, search_radius=5000, power=2):
    """
    Interpolação por Inverse Distance Weighting (IDW)
    
    Parâmetros
    ----------
    x, y : array
        Coordenadas dos dados
    z : array
        Valores a interpolar
    grid_x, grid_y : 2D arrays
        Grade de saída
    search_radius : float
        Raio de busca
    power : float
        Expoente da distância (default: 2)
    
    Retorna
    -------
    grid_z : 2D array
        Valores interpolados
    """
    
    tree = cKDTree(np.column_stack([x, y]))
    
    grid_z = np.full(grid_x.shape, np.nan)
    
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            
            xg = grid_x[i, j]
            yg = grid_y[i, j]
            
            indices = tree.query_ball_point([xg, yg], search_radius)
            
            if len(indices) == 0:
                continue
            
            x_neigh = x[indices]
            y_neigh = y[indices]
            z_neigh = z[indices]
            
            # Distâncias
            distances = np.sqrt((x_neigh - xg)**2 + (y_neigh - yg)**2)
            
            # Evitar divisão por zero
            distances[distances < 1e-10] = 1e-10
            
            # Pesos IDW
            weights = 1.0 / (distances ** power)
            weights /= weights.sum()
            
            # Interpolação
            grid_z[i, j] = np.sum(weights * z_neigh)
    
    return grid_z


def interpolate_nearest(x, y, z, grid_x, grid_y):
    """
    Interpolação por vizinho mais próximo
    Mais rápida, mas menos suave
    
    Parâmetros
    ----------
    x, y : array
        Coordenadas dos dados
    z : array
        Valores
    grid_x, grid_y : 2D arrays
        Grade de saída
    
    Retorna
    -------
    grid_z : 2D array
        Valores interpolados
    """
    
    # Usar scipy.interpolate.griddata
    points = np.column_stack([x, y])
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    grid_z_flat = griddata(points, z, grid_points, method='nearest')
    grid_z = grid_z_flat.reshape(grid_x.shape)
    
    return grid_z


def interpolate_linear(x, y, z, grid_x, grid_y):
    """
    Interpolação linear (triangulação)
    
    Parâmetros
    ----------
    x, y : array
        Coordenadas dos dados
    z : array
        Valores
    grid_x, grid_y : 2D arrays
        Grade de saída
    
    Retorna
    -------
    grid_z : 2D array
        Valores interpolados
    """
    
    points = np.column_stack([x, y])
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    grid_z_flat = griddata(points, z, grid_points, method='linear')
    grid_z = grid_z_flat.reshape(grid_x.shape)
    
    return grid_z


def interpolate_cubic(x, y, z, grid_x, grid_y):
    """
    Interpolação cúbica (mais suave)
    
    Parâmetros
    ----------
    x, y : array
        Coordenadas
    z : array
        Valores
    grid_x, grid_y : 2D arrays
        Grade de saída
    
    Retorna
    -------
    grid_z : 2D array
        Valores interpolados
    """
    
    points = np.column_stack([x, y])
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    grid_z_flat = griddata(points, z, grid_points, method='cubic')
    grid_z = grid_z_flat.reshape(grid_x.shape)
    
    return grid_z


def smooth_grid(grid, sigma=1):
    """
    Suaviza grade usando filtro Gaussiano
    
    Parâmetros
    ----------
    grid : 2D array
        Grade a suavizar
    sigma : float
        Desvio padrão do filtro Gaussiano (em células)
    
    Retorna
    -------
    smoothed : 2D array
        Grade suavizada
    """
    
    # Criar máscara de NaNs
    mask = ~np.isnan(grid)
    
    # Substituir NaNs por 0 temporariamente
    grid_filled = np.nan_to_num(grid, nan=0.0)
    
    # Aplicar filtro Gaussiano
    smoothed = gaussian_filter(grid_filled, sigma=sigma)
    
    # Normalizar pela máscara
    mask_smoothed = gaussian_filter(mask.astype(float), sigma=sigma)
    mask_smoothed[mask_smoothed < 1e-10] = 1e-10  # Evitar divisão por zero
    
    smoothed = smoothed / mask_smoothed
    
    # Restaurar NaNs
    smoothed[~mask] = np.nan
    
    return smoothed


def create_regular_grid(x_min, x_max, y_min, y_max, resolution):
    """
    Cria grade regular
    
    Parâmetros
    ----------
    x_min, x_max : float
        Limites em X (metros)
    y_min, y_max : float
        Limites em Y (metros)
    resolution : float
        Resolução da grade (metros)
    
    Retorna
    -------
    grid_x, grid_y : 2D arrays
        Coordenadas da grade
    """
    
    # Vetores 1D
    x_vec = np.arange(x_min, x_max + resolution, resolution)
    y_vec = np.arange(y_min, y_max + resolution, resolution)
    
    # Grade 2D
    grid_x, grid_y = np.meshgrid(x_vec, y_vec)
    
    return grid_x, grid_y


def bin_statistics(x, y, z, grid_x, grid_y, statistic='mean'):
    """
    Calcula estatística em bins (células da grade)
    
    Parâmetros
    ----------
    x, y : array
        Coordenadas dos dados
    z : array
        Valores
    grid_x, grid_y : 2D arrays
        Grade de saída
    statistic : str
        'mean', 'median', 'std', 'count', 'min', 'max'
    
    Retorna
    -------
    grid_stat : 2D array
        Estatística em cada célula
    """
    
    from scipy.stats import binned_statistic_2d
    
    # Extrair bins da grade
    x_edges = np.concatenate([grid_x[0, :], [grid_x[0, -1] + (grid_x[0, 1] - grid_x[0, 0])]])
    y_edges = np.concatenate([grid_y[:, 0], [grid_y[-1, 0] + (grid_y[1, 0] - grid_y[0, 0])]])
    
    # Calcular estatística
    stat, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z,
        statistic=statistic,
        bins=[x_edges, y_edges]
    )
    
    return stat.T  # Transpor para corresponder à grade


def remove_outliers_grid(grid, n_sigma=3):
    """
    Remove outliers de grade usando critério de sigma
    
    Parâmetros
    ----------
    grid : 2D array
        Grade com dados
    n_sigma : float
        Número de desvios padrão
    
    Retorna
    -------
    grid_filtered : 2D array
        Grade sem outliers (substituídos por NaN)
    """
    
    # Calcular média e std (ignorando NaNs)
    mean = np.nanmean(grid)
    std = np.nanstd(grid)
    
    # Criar máscara de outliers
    outliers = np.abs(grid - mean) > n_sigma * std
    
    # Substituir outliers por NaN
    grid_filtered = grid.copy()
    grid_filtered[outliers] = np.nan
    
    n_outliers = np.sum(outliers)
    pct = 100 * n_outliers / np.sum(~np.isnan(grid))
    
    print(f"Outliers removidos: {n_outliers} ({pct:.1f}%)")
    
    return grid_filtered


def fill_gaps_grid(grid, max_gap_size=3):
    """
    Preenche pequenos gaps na grade por interpolação
    
    Parâmetros
    ----------
    grid : 2D array
        Grade com gaps (NaNs)
    max_gap_size : int
        Tamanho máximo de gap a preencher (em células)
    
    Retorna
    -------
    grid_filled : 2D array
        Grade com gaps preenchidos
    """
    
    from scipy.ndimage import distance_transform_edt
    
    # Máscara de dados válidos
    mask = ~np.isnan(grid)
    
    # Calcular distância de cada NaN ao dado válido mais próximo
    distances = distance_transform_edt(~mask)
    
    # Preencher apenas gaps pequenos
    fill_mask = (distances <= max_gap_size) & (~mask)
    
    if not np.any(fill_mask):
        return grid.copy()
    
    # Obter coordenadas
    y_coords, x_coords = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    
    # Pontos válidos
    valid_points = np.column_stack([
        x_coords[mask].ravel(),
        y_coords[mask].ravel()
    ])
    valid_values = grid[mask].ravel()
    
    # Pontos a preencher
    fill_points = np.column_stack([
        x_coords[fill_mask].ravel(),
        y_coords[fill_mask].ravel()
    ])
    
    # Interpolar (nearest neighbor para gaps pequenos)
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(valid_points, valid_values)
    fill_values = interp(fill_points)
    
    # Criar grade preenchida
    grid_filled = grid.copy()
    grid_filled[fill_mask] = fill_values
    
    n_filled = np.sum(fill_mask)
    print(f"Células preenchidas: {n_filled}")
    
    return grid_filled


# ============================================
# EXEMPLO DE USO
# ============================================

if __name__ == "__main__":
    
    print("="*60)
    print("TESTE DE FUNÇÕES DE INTERPOLAÇÃO")
    print("="*60)
    
    # Dados de exemplo
    np.random.seed(42)
    n_points = 1000
    
    x = np.random.uniform(0, 100000, n_points)  # 0-100 km
    y = np.random.uniform(0, 100000, n_points)
    z = np.sin(x/10000) * np.cos(y/10000) + np.random.normal(0, 0.1, n_points)
    
    # Criar grade
    grid_x, grid_y = create_regular_grid(0, 100000, 0, 100000, 1000)  # 1 km
    
    print(f"\nDados de entrada:")
    print(f"  Pontos: {n_points}")
    print(f"  Grade: {grid_x.shape}")
    
    # Testar interpolações
    print("\n1. Interpolação Gaussiana...")
    grid_gauss, std_gauss, count_gauss = interpolate_gaussian(
        x, y, z, grid_x, grid_y, search_radius=5000
    )
    print(f"   Células preenchidas: {np.sum(~np.isnan(grid_gauss))}/{grid_gauss.size}")
    
    print("\n2. Interpolação por Mediana...")
    grid_median, mad_median, count_median = interpolate_median(
        x, y, z, grid_x, grid_y, search_radius=5000
    )
    print(f"   Células preenchidas: {np.sum(~np.isnan(grid_median))}/{grid_median.size}")
    
    print("\n3. Interpolação IDW...")
    grid_idw = interpolate_idw(x, y, z, grid_x, grid_y, search_radius=5000)
    print(f"   Células preenchidas: {np.sum(~np.isnan(grid_idw))}/{grid_idw.size}")
    
    print("\n4. Suavização...")
    grid_smooth = smooth_grid(grid_gauss, sigma=2)
    print(f"   ✓ Grade suavizada")
    

    print("\n✓ Testes concluídos!")
