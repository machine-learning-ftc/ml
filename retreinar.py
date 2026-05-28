import os
import requests
import time
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sentence_transformers import SentenceTransformer

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
API_KEY_SUPABASE = os.getenv("API_KEY_SUPABASE")

HEADERS_SUPABASE = {
    "apikey": API_KEY_SUPABASE,
    "Authorization": f"Bearer {API_KEY_SUPABASE}",
    "Content-Type": "application/json"
}

# Inicializa o tradutor de vetores local (MiniLM de 384 dimensões)
print("[ML LOCAL] Carregando modelo de embedding MiniLM local...")
encoder_local = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def puxar_dados_para_treino():
    """Puxa absolutamente todos os dados que possuem vetores no banco de dados

    (agora configurado com paginação inteligente para quebrar o teto de 1000 linhas).
    """
    print("\n[ML TREINO] Baixando a base vetorizada completa do Supabase para treino...")
    todos_os_dados = []
    offset = 0
    limite_lote = 1000
    
    while True:
        headers_paginado = {
            "apikey": API_KEY_SUPABASE,
            "Authorization": f"Bearer {API_KEY_SUPABASE}",
            "Content-Type": "application/json",
            "Range": f"{offset}-{offset + limite_lote - 1}"
        }
        
        # Puxa os dados que já foram convertidos para 384D
        endpoint = f"{SUPABASE_URL}/rest/v1/fact_checks?select=query,claim,verdict,embedding&embedding=not.is.null"
        resposta = requests.get(endpoint, headers=headers_paginado)
        
        if resposta.status_code != 200:
            raise Exception(f"Erro ao acessar o banco: {resposta.text}")
            
        dados_lote = resposta.json()
        if not dados_lote:
            break
            
        todos_os_dados.extend(dados_lote)
        if len(dados_lote) < limite_lote:
            break
            
        offset += limite_lote
        time.sleep(0.2)
        
    print(f"[ML TREINO] Base carregada! Encontrados {len(todos_os_dados)} registros vetorizados para o treino.")
    return todos_os_dados


def preparar_e_treinar():
    """Processa a matriz de dados, divide em treino/teste e calibra a Regressão Logística."""
    dados = puxar_dados_para_treino()
    
    if not dados:
        print("❌ ERRO: Nenhuma linha vetorizada disponível para treinar o modelo.")
        return

    X_lista = []
    y_lista = []
    
    print("[ML TREINO] Extraindo vetores e aplicando mapeamento de classes...")
    for reg in dados:
        verdict = str(reg.get("verdict")).lower().strip()
        embedding_raw = reg.get("embedding")
        
        # Mapeamento estrito das classes
        if verdict in ["true", "verdadeiro", "v", "1", "1.0", "correto"]:
            classe = 1
        elif verdict in ["false", "falso", "f", "0", "0.0", "fake", "fakenews"]:
            classe = 0
        else:
            continue # Ignora nulos ou 'uncertain' na construção das fronteiras de decisão
            
        # Parse do vetor caso venha como String estruturada do Postgres
        if isinstance(embedding_raw, str):
            embedding_vetor = json.loads(embedding_raw)
        else:
            embedding_vetor = embedding_raw
            
        X_lista.append(embedding_vetor)
        y_lista.append(classe)
        
    print(f"[ML TREINO] Amostras finais filtradas: {len(X_lista)} (True/False combinados).")
    
    if len(X_lista) == 0:
        print("❌ ERRO CRÍTICO: Nenhum dado sobreviveu aos filtros de mapeamento textuais!")
        return

    X = np.array(X_lista)
    y = np.array(y_lista)
    
    # Separação clássica de 20% para teste empírico (Holdout)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[ML TREINO] Ajustando coeficientes da Regressão Logística Local...")
    # class_weight='balanced' para calibrar as nuances entre True e False de forma justa
    modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
    modelo.fit(X_treino, y_treino)
    
    # Avaliação do modelo matemático
    predicoes = modelo.predict(X_teste)
    precisao = accuracy_score(y_teste, predicoes)
    
    print(f"\n=========================================")
    print(f"🎯 NOVA PRECISÃO DO MODELO LOCAL: {precisao * 100:.2f}%")
    print(f"=========================================\n")
    print(classification_report(y_teste, predicoes, target_names=["False", "True"]))
    
    # Exportação estável do artefato para o Flask (app.py) utilizar offline
    joblib.dump(modelo, "modelo_fact_checker.pkl")
    print("[ML TREINO] Arquivo 'modelo_fact_checker.pkl' atualizado com sucesso!")


if __name__ == "__main__":
    preparar_e_treinar()