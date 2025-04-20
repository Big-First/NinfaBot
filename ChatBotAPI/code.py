# Instala o SDK do .NET (ajuste a versão se necessário - use a LTS ou a que seu projeto usa)
!wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
!chmod +x ./dotnet-install.sh
!./dotnet-install.sh --version latest --install-dir ./dotnet
!./dotnet/dotnet --version

# Adiciona o dotnet ao PATH (para esta sessão do Colab)
import os
os.environ['DOTNET_ROOT'] = os.path.abspath('./dotnet')
os.environ['PATH'] += ':'+os.environ['DOTNET_ROOT']

# Instala a ferramenta .NET Interactive (Polyglot Notebooks)
!dotnet tool install --global Microsoft.dotnet-interactive

# (Opcional, mas recomendado) Instala extensões úteis como plotagem
# !dotnet interactive jupyter install # Instala o kernel Jupyter (pode não ser necessário no Colab direto)
# !pip install plotly -q # Exemplo para gráficos