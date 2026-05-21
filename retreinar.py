import requests
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# 1. Credenciais do Supabase
SUPABASE_URL = "https://rmlyubbislrtgwdshmpd.supabase.co"
API_KEY = "COLE_SUA_CHAVE_SERVICE_ROLE_AQUI" 
NOME_TABELA = "fact_checks"

endpoint = f"{SUPABASE_URL}/rest/v1/{NOME_TABELA}?select=*"
headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

try:
    print("📡 Baixando dados do Supabase...")
    resposta = requests.get(endpoint, headers=headers)
    
    if resposta.status_code == 200:
        dados = resposta.json()
        print(f" {len(dados)} registros recebidos. Iniciando limpeza...")

        # 2. TRANSFORMAÇÃO 
        df = pd.DataFrame(dados)
        
        # Filtra apenas o que é 'true' ou 'false'
        df_limpo = df[df['verdict'].isin(['true', 'false'])].copy()
        
        # Converte para números
        df_limpo['alvo'] = df_limpo['verdict'].map({'true': 1, 'false': 0})

        print(f" Limpeza concluída! Ficaram {len(df_limpo)} notícias válidas para o treino.")

        # 3. TREINAMENTO 
        print(" Iniciando o retreinamento do modelo SVM...")
        vetorizador = TfidfVectorizer(ngram_range=(1, 2))
        modelo = LinearSVC(class_weight='balanced', dual=False)

        X = vetorizador.fit_transform(df_limpo['query'])
        y = df_limpo['alvo']

        modelo.fit(X, y)

        # 4. DEPLOY (Salvando os novos arquivos .pkl)
        joblib.dump(vetorizador, 'vetorizador.pkl')
        joblib.dump(modelo, 'modelo_eleicoes.pkl')

        print(" SUCESSO! A IA aprendeu com o banco de dados e os arquivos .pkl foram atualizados!")
    else:
        print(f" Erro ao puxar dados: {resposta.status_code}")
        print(resposta.text)

except Exception as e:
    print(f" Erro na pipeline: {str(e)}")