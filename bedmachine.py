import earthaccess
import os

# 1. Autenticação (abrirá uma janela para login ou usará suas credenciais salvas)
earthaccess.login()

# 2. Pesquisar pelo dataset BedMachine Antarctica (V3)
# O short_name é NSIDC-0756
results = earthaccess.search_data(
    short_name="NSIDC-0756",
    version="3"
)

# 3. Definir o diretório de saída
output_dir = r"D:\WILLIAM\PIBIC_MARCOPIBIC_MARCO\thwaites_winter\data\bedmachine"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 4. Baixar o arquivo .nc
print("Iniciando download via Earthdata Cloud...")
earthaccess.download(results, output_dir)