import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

# 1. Credenciais do Supabase
SUPABASE_URL = "https://rmlyubbislrtgwdshmpd.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJtbHl1YmJpc2xydGd3ZHNobXBkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODk5MTM1NSwiZXhwIjoyMDk0NTY3MzU1fQ.A4x1u2ZnxfPwiRinWf7mfUGcIOVWy2Q9U_KTRVtbZyI"
NOME_TABELA = "fact_checks"

headers = {
    "apikey": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

try:
    # 1. DOWNLOAD com paginação
    print("Baixando todos os dados do Supabase...")
    todos = []
    limit = 1000
    offset = 0

    while True:
        endpoint = f"{SUPABASE_URL}/rest/v1/{NOME_TABELA}?select=*&limit={limit}&offset={offset}"
        resposta = requests.get(endpoint, headers=headers)
        batch = resposta.json()
        if not batch:
            break
        todos.extend(batch)
        offset += limit
        print(f"  {len(todos)} registros baixados...")

    print(f"Total final: {len(todos)} registros")

    # 2. TRANSFORMAÇÃO
    df = pd.DataFrame(todos)
    df_limpo = df[df['verdict'].isin(['true', 'false', 'uncertain'])].copy()
    df_limpo['alvo'] = df_limpo['verdict'].map({'true': 1, 'false': 0, 'uncertain': 2})

    print("Balanceamento original:")
    print(df_limpo['verdict'].value_counts())

    # 3. BALANCEAMENTO true/false 50/50 (mantém todos os uncertain)
    true_df = df_limpo[df_limpo['alvo'] == 1]
    false_df = df_limpo[df_limpo['alvo'] == 0].sample(n=len(true_df), random_state=42)
    uncertain_df = df_limpo[df_limpo['alvo'] == 2]

    df_balanceado = pd.concat([true_df, false_df, uncertain_df]).sample(frac=1, random_state=42)

    print("\nBalanceamento após correção:")
    print(df_balanceado['verdict'].value_counts())
    print(f"Total para treino: {len(df_balanceado)} registros")

    # 4. TREINAMENTO
    print("\nIniciando retreinamento do modelo SVM...")
    vetorizador = TfidfVectorizer(ngram_range=(1, 2))
    svm = LinearSVC(class_weight='balanced', dual=False)
    modelo = CalibratedClassifierCV(svm, cv=3)

    x = vetorizador.fit_transform(df_balanceado['query'])
    y = df_balanceado['alvo']

    modelo.fit(x, y)

    # 5. SALVAR
    joblib.dump(vetorizador, 'vetorizador.pkl')
    joblib.dump(modelo, 'modelo_eleicoes.pkl')

    print("SUCESSO! Modelo retreinado e arquivos .pkl atualizados!")

except Exception as e:
    print(f"Erro na pipeline: {str(e)}")