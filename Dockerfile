# Usa uma imagem oficial do Python otimizada para Linux e ARM64/AMD64
FROM python:3.11-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para compilar pacotes leves
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências
COPY requirements.txt .

# Configura o pip para instalar a versão leve de CPU direto do repositório estável
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copia o resto dos arquivos do projeto (incluindo o app.py e o modelo .pkl)
COPY . .

# Expõe a porta 5000 do Flask
EXPOSE 5000

# Variáveis de ambiente para travar o uso de memória por threads extras
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

# Comando definitivo que o Docker vai rodar para ligar a sua API
CMD ["python", "app.py"]