# 🔥 IA para Análise Econômica Brasil + Global
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from bcb import sgs

# 📊 🔐 Configure sua chave de API do FRED
FRED_API_KEY = 'INSIRA_SUA_CHAVE_AQUI'
fred = Fred(api_key=FRED_API_KEY)

# ==========================================
# 📥 Funções de Coleta de Dados
# ==========================================

# ✅ Dados de mercado - Brasil + Global
def obter_dados_mercado(tickers, inicio, fim):
    dados = yf.download(tickers, start=inicio, end=fim)['Adj Close']
    return dados

# ✅ Dados do Banco Central do Brasil (BCB)
def obter_dados_bcb(codigos):
    dados = pd.DataFrame()
    for nome, codigo in codigos.items():
        serie = sgs.get(codigo)
        serie.columns = [nome]
        dados = pd.concat([dados, serie], axis=1)
    return dados

# ✅ Dados do Federal Reserve (FRED)
def obter_dados_fred(series):
    dados = pd.DataFrame()
    for nome, codigo in series.items():
        serie = fred.get_series(codigo)
        dados[nome] = serie
    dados.index = pd.to_datetime(dados.index)
    dados = dados.sort_index()
    return dados

# ==========================================
# 🚀 Pipeline de Modelagem e Análise
# ==========================================

def pipeline():
    inicio = '2010-01-01'
    fim = '2024-12-31'

    # 🎯 Dados de mercado
    tickers = ['^BVSP', 'USDBRL=X', '^GSPC', '^IXIC', '^STOXX50E', '^N225', 'GC=F', 'CL=F']
    mercado = obter_dados_mercado(tickers, inicio, fim)
    mercado = mercado.dropna()

    # 🔥 Dados do Banco Central do Brasil
    bcb_codigos = {
        'Selic': 4189,
        'IPCA': 433,
        'PIB_Brasil': 4380,
        'Cambio_BCB': 1
    }
    bcb = obter_dados_bcb(bcb_codigos)

    # 🌎 Dados do FRED (Global/EUA)
    fred_series = {
        'Federal_Funds_Rate': 'FEDFUNDS',
        'CPI_USA': 'CPIAUCSL',
        'PIB_USA': 'GDPC1',
        'Unemployment_USA': 'UNRATE'
    }
    fred_data = obter_dados_fred(fred_series)

    # 🔗 Unir os datasets
    df = pd.concat([mercado, bcb, fred_data], axis=1)
    df = df.fillna(method='ffill').dropna()

    # ==========================================
    # 🏗 Feature Engineering
    # ==========================================
    df['Retorno_BVSP'] = df['^BVSP'].pct_change()
    df['Retorno_SP500'] = df['^GSPC'].pct_change()
    df['Vol_BVSP'] = df['Retorno_BVSP'].rolling(21).std()
    df['Vol_SP500'] = df['Retorno_SP500'].rolling(21).std()
    df['Cambio_pct'] = df['USDBRL=X'].pct_change()

    df = df.dropna()

    # ==========================================
    # 🏷 Criação dos labels (Regimes Econômicos)
    # ==========================================
    df['Regime'] = np.where(
        (df['PIB_Brasil'].pct_change() > 0) & (df['IPCA'].pct_change() < 0.01),
        1,  # Expansão + Inflação baixa
        0   # Contração ou Inflação alta
    )

    # ==========================================
    # 🧠 Machine Learning - Random Forest
    # ==========================================
    features = [
        'Retorno_BVSP', 'Retorno_SP500', 'Vol_BVSP', 'Vol_SP500',
        'Cambio_pct', 'Selic', 'IPCA', 'PIB_Brasil', 'Federal_Funds_Rate', 'CPI_USA'
    ]
    X = df[features]
    y = df['Regime']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n🔍 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão - Regimes Econômicos")
    plt.show()

    # ==========================================
    # 🔮 Previsão do Regime Atual
    # ==========================================
    ultimo = df.iloc[-1][features].values.reshape(1, -1)
    regime_atual = modelo.predict(ultimo)[0]

    # ==========================================
    # 📊 Estratégia de Alocação
    # ==========================================
    if regime_atual == 1:
        print("\n📈 Cenário Atual: Expansão + Inflação Controlada")
        alocacao = {
            'Ações Brasil': 0.35,
            'Ações EUA': 0.25,
            'Europa/Japão': 0.15,
            'Commodities': 0.10,
            'Renda Fixa': 0.10,
            'Caixa': 0.05
        }
    else:
        print("\n📉 Cenário Atual: Contração ou Inflação Alta")
        alocacao = {
            'Renda Fixa Brasil': 0.40,
            'Renda Fixa Global': 0.20,
            'Dólar': 0.10,
            'Ouro': 0.10,
            'Ações Defensivas': 0.10,
            'Caixa': 0.10
        }

    print("\n💰 Sugestão de Alocação de Carteira:")
    for ativo, percentual in alocacao.items():
        print(f"{ativo}: {percentual * 100:.1f}%")

# ==========================================
# 🚀 Executar o pipeline
pipeline()
