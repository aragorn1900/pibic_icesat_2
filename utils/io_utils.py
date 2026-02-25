"""
Funções para leitura/escrita HDF5
Trata escalares, arrays e todos os casos
"""

import h5py
import numpy as np
from pathlib import Path

# ============================================
# LEITURA DE ATL06
# ============================================

def read_atl06(filepath, variables=None, ground_track='gt1l'):
    """
    Lê arquivo ATL06 e extrai variáveis
      
    Parâmetros
    ----------
    filepath : str ou Path
        Caminho para arquivo ATL06
    variables : list, opcional
        Lista de variáveis para extrair
    ground_track : str
        Ground track: 'gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'
    
    Retorna
    -------
    data : dict
        Dicionário com variáveis extraídas
    """
    
    if variables is None:
        variables = [
            'latitude',
            'longitude', 
            'h_li',
            'delta_time',
            'atl06_quality_summary',
            'h_li_sigma',
            'segment_id',
            'dh_fit_dx'
        ]
    
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        
        # Verificar se ground track existe
        if ground_track not in f.keys():
            print(f"⚠ Ground track {ground_track} não encontrado em {filepath}")
            return None
        
        # Caminhos base
        base = f'{ground_track}/land_ice_segments'
        fit_stats = f'{base}/fit_statistics'
        geophys = f'{base}/geophysical'
        
        # Extrair cada variável
        for var in variables:
            try:
                # Coordenadas e altura no base
                if var in ['latitude', 'longitude', 'delta_time', 'segment_id', 
                          'h_li', 'h_li_sigma', 'atl06_quality_summary']:
                    var_path = f'{base}/{var}'
                
                # dh_fit_dx e outras em fit_statistics
                elif var in ['dh_fit_dx', 'h_robust_sprd', 'snr_significance',
                            'signal_selection_source']:
                    var_path = f'{fit_stats}/{var}'
                
                # Correções em geophysical
                elif var in ['tide_earth', 'tide_load', 'tide_ocean', 'tide_pole', 'dac']:
                    var_path = f'{geophys}/{var}'
                
                else:
                    # Tentar no base
                    var_path = f'{base}/{var}'
                
                # Ler dados
                data[var] = f[var_path][:]
                
            except KeyError:
                print(f"⚠ Variável {var} não encontrada em {ground_track}")
                data[var] = np.array([])
        
        # Adicionar metadados
        data['ground_track'] = ground_track
        data['filename'] = Path(filepath).name
        
        # Orbit info
        try:
            data['orbit_number'] = f['orbit_info/orbit_number'][0]
            data['rgt'] = f['orbit_info/rgt'][0]
            data['cycle_number'] = f['orbit_info/cycle_number'][0]
        except:
            data['orbit_number'] = -1
            data['rgt'] = -1
            data['cycle_number'] = -1
    
    return data


def read_all_ground_tracks(filepath, variables=None):
    """
    Lê todos os ground tracks de um arquivo ATL06
    
    Retorna
    -------
    all_data : dict
        Dicionário com dados de todos os GTs
    """
    
    ground_tracks = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    all_data = {}
    
    for gt in ground_tracks:
        data = read_atl06(filepath, variables=variables, ground_track=gt)
        if data is not None and len(data.get('latitude', [])) > 0:
            all_data[gt] = data
    
    return all_data


# ============================================
# LEITURA/ESCRITA HDF5 GENÉRICA
# ============================================

def read_hdf5(filepath, variables=None):
    """
    Lê arquivo HDF5 processado
    sempre retorna arrays
    
    Parâmetros
    ----------
    filepath : str ou Path
        Caminho do arquivo
    variables : list, opcional
        Variáveis específicas a ler. Se None, lê todas.
    
    Retorna
    -------
    data : dict
        Dados lidos 
    """
    
    data = {}
    
    try:
        with h5py.File(filepath, 'r') as f:
            
            # Se não especificou variáveis, pegar todas as chaves
            if variables is None:
                variables = list(f.keys())
            
            # Processar cada variável
            for var in variables:
                
                if var not in f.keys():
                    # Não existe como dataset, verificar atributos
                    if var in f.attrs.keys():
                        data[var] = f.attrs[var]
                    continue
                
                obj = f[var]
                
                # Verificar tipo do objeto
                if isinstance(obj, h5py.Group):
                    # É um grupo (subdiretório), pular
                    continue
                
                elif isinstance(obj, h5py.Dataset):
                    # É um dataset (dados)
                    
                    shape = obj.shape
                    
                    if shape == ():
                        # ESCALAR (shape vazio)
                        # IMPORTANTE: Converter para array de 1 elemento
                        scalar_value = obj[()]
                        data[var] = np.array([scalar_value])  # ← SEMPRE ARRAY
                        
                    elif shape == (1,):
                        # Array de tamanho 1
                        # Manter como array (não extrair o valor)
                        data[var] = obj[:]  # ← ARRAY de 1 elemento
                        
                    elif len(shape) == 1 and shape[0] > 0:
                        # Array 1D normal
                        data[var] = obj[:]
                        
                    elif len(shape) > 1:
                        # Array multidimensional
                        data[var] = obj[:]
                        
                    else:
                        # Caso estranho, tentar ler de forma segura
                        try:
                            value = obj[()]
                            # Garantir que é array
                            data[var] = np.atleast_1d(value)
                        except:
                            try:
                                data[var] = obj[:]
                            except:
                                # Pular variável problemática
                                continue
            
            # Ler TODOS os atributos globais
            for attr_name in f.attrs.keys():
                if attr_name not in data:
                    data[attr_name] = f.attrs[attr_name]
    
    except Exception as e:
        print(f"⚠ Erro ao ler {Path(filepath).name}: {e}")
        return {}
    
    return data

def write_hdf5(filepath, data_dict, mode='w', compression='gzip'):
    """
    Escreve dados em arquivo HDF5
    
    Parâmetros
    ----------
    filepath : str ou Path
        Caminho de saída
    data_dict : dict
        Dicionário com arrays numpy ou valores escalares
    mode : str
        'w' = escrever novo (sobrescreve)
        'a' = append (adiciona/atualiza)
    compression : str
        'gzip', 'lzf', ou None
    """
    
    with h5py.File(filepath, mode) as f:
        
        for key, value in data_dict.items():
            
            # Deletar se já existe (modo append)
            if key in f.keys():
                del f[key]
            
            # Criar dataset ou atributo
            if isinstance(value, (np.ndarray, list)):
                # Array -> dataset
                value_array = np.asarray(value)
                
                if compression and value_array.size > 1:
                    f.create_dataset(key, data=value_array, compression=compression)
                else:
                    f.create_dataset(key, data=value_array)
            
            elif isinstance(value, (int, float, np.integer, np.floating)):
                # Número escalar -> atributo
                f.attrs[key] = value
            
            elif isinstance(value, str):
                # String -> atributo
                f.attrs[key] = value
            
            elif isinstance(value, bool):
                # Boolean -> atributo
                f.attrs[key] = value
            
            elif isinstance(value, bytes):
                # Bytes -> atributo
                f.attrs[key] = value
            
            else:
                # Tentar converter para array
                try:
                    f.create_dataset(key, data=np.array(value))
                except:
                    # Último recurso: ignorar
                    pass


# ============================================
# MESCLAGEM DE ARQUIVOS
# ============================================

def merge_hdf5_files(input_files, output_file, variables=None):
    """
    Mescla múltiplos arquivos HDF5 em um único
    
    Equivalente a: merge.py do CAPTOOLKIT
    
    Parâmetros
    ----------
    input_files : list
        Lista de caminhos de arquivos
    output_file : str ou Path
        Arquivo de saída
    variables : list, opcional
        Variáveis específicas a mesclar. Se None, todas.
    """
    
    if len(input_files) == 0:
        print("⚠ Nenhum arquivo para mesclar!")
        return
    
    print(f"Mesclando {len(input_files)} arquivos...")
    
    # Ler primeiro arquivo para obter estrutura
    first_data = read_hdf5(input_files[0])
    
    if variables is None:
        # Identificar apenas arrays (datasets), não atributos
        variables = [k for k, v in first_data.items() 
                     if isinstance(v, np.ndarray) and v.size > 1]
    
    # Inicializar listas para concatenação
    merged_data = {var: [] for var in variables}
    
    # Coletar metadados
    all_metadata = {}
    
    # Ler todos os arquivos
    for filepath in input_files:
        try:
            data = read_hdf5(filepath, variables=variables)
            
            # Concatenar arrays
            for var in variables:
                if var in data and isinstance(data[var], np.ndarray) and data[var].size > 0:
                    merged_data[var].append(data[var])
            
            # Coletar metadados únicos
            for key, value in data.items():
                if key not in variables:
                    if isinstance(value, (int, float, str)):
                        if key not in all_metadata:
                            all_metadata[key] = []
                        all_metadata[key].append(value)
        
        except Exception as e:
            print(f"⚠ Erro ao ler {filepath}: {e}")
            continue
    
    # Concatenar arrays
    for var in variables:
        if len(merged_data[var]) > 0:
            merged_data[var] = np.concatenate(merged_data[var])
        else:
            merged_data[var] = np.array([])
    
    # Adicionar metadados consolidados
    merged_data['n_files'] = len(input_files)
    for key, values in all_metadata.items():
        if len(values) > 0:
            try:
                merged_data[key] = np.unique(values)
            except:
                merged_data[key] = values[0]
    
    # Salvar
    write_hdf5(output_file, merged_data)
    
    print(f"✓ Mesclado em: {output_file}")
    if len(variables) > 0 and variables[0] in merged_data:
        print(f"  Total de pontos: {len(merged_data[variables[0]]):,}")


# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def list_hdf5_contents(filepath):
    """Lista conteúdo de arquivo HDF5"""
    
    print(f"\n{'='*60}")
    print(f"Conteúdo de: {Path(filepath).name}")
    print(f"{'='*60}")
    
    with h5py.File(filepath, 'r') as f:
        
        print("\nDatasets:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                shape = f[key].shape
                dtype = f[key].dtype
                print(f"  {key:30s} {str(shape):20s} {dtype}")
        
        print("\nAtributos:")
        for attr in f.attrs.keys():
            value = f.attrs[attr]
            print(f"  {attr:30s} = {value}")
    
    print(f"{'='*60}\n")


def count_points_in_files(file_list, var='latitude'):
    """Conta total de pontos em múltiplos arquivos"""
    
    total = 0
    
    for filepath in file_list:
        try:
            data = read_hdf5(filepath, variables=[var])
            if var in data and isinstance(data[var], np.ndarray):
                total += len(data[var])
        except:
            continue
    

    return total
