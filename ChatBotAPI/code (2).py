# @title Instalar .NET SDK e Ferramenta Interactive (Execute uma vez)
# @markdown Selecione a versão desejada do .NET SDK (LTS ou a do seu projeto)
dotnet_version = "8.0" #@param ["8.0", "7.0", "6.0", "latest"]

print(f"Instalando .NET SDK versão: {dotnet_version}...")
# Baixa o script de instalação
!wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh -q
# Dá permissão de execução
!chmod +x ./dotnet-install.sh
# Executa a instalação na pasta ./dotnet
!./dotnet-install.sh --version {dotnet_version} --install-dir ./dotnet
# Verifica a versão instalada
!./dotnet/dotnet --version

# Adiciona o diretório dotnet ao PATH da sessão atual
import os
os.environ['DOTNET_ROOT'] = os.path.abspath('./dotnet')
os.environ['PATH'] += ':'+os.environ['DOTNET_ROOT']
print("DOTNET_ROOT e PATH configurados.")

# Instala a ferramenta global dotnet-interactive
print("Instalando Microsoft.dotnet-interactive...")
!dotnet tool install --global Microsoft.dotnet-interactive
print(".NET Interactive instalado.")

# Opcional: Instalar o kernel Jupyter (pode ajudar na integração, mas nem sempre necessário no Colab)
# print("Instalando kernel Jupyter do .NET Interactive...")
# !dotnet interactive jupyter install
# print("Kernel Jupyter instalado.")

print("\nAmbiente .NET pronto!")