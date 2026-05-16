import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

add_md('''# Previsão de Câncer de Mama usando KNN

Nesse notebook, vou usar o KNN pra classificar dados de câncer de mama (benigno ou maligno).
Vou importar os dados do próprio scikit-learn, dar uma olhada neles, preparar e treinar o modelo.''')

add_code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

sns.set_theme(style="whitegrid")''')

add_md('''## 1. Conhecendo a base de dados
Vou puxar os dados e ver o tamanho da base e quais são as features (as variáveis que vou usar pra prever).''')

add_code('''dados = load_breast_cancer()

print(f"Total de linhas e colunas: {dados.data.shape}")
print(f"Features: {dados.feature_names[:5]} ... (são 30 no total)\\n")
print(f"Classes: {dados.target_names} (onde 0 é maligno e 1 é benigno)")

df = pd.DataFrame(dados.data, columns=dados.feature_names)
df['alvo'] = dados.target
df['classe'] = df['alvo'].map({0: 'Maligno', 1: 'Benigno'})

df.head()''')

add_md('''## 2. Dando uma olhada nos dados (EDA) ''')

add_code('''plt.figure(figsize=(14, 5))

# gráfico 1: proporção das classes
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='classe', palette='pastel')
plt.title('Quantos casos de cada?')
plt.xlabel('')

# gráfico 2: olhando o tamanho médio do tumor por classe
plt.subplot(1, 2, 2)
sns.kdeplot(data=df, x='mean radius', hue='classe', fill=True, palette='pastel')
plt.title('Tamanho do Raio do Tumor')
plt.xlabel('Raio Médio')

plt.show()''')

add_md('''## 3. Preparando os dados (StandardScaler)
O KNN calcula distâncias entre os pontos. Se uma variável tem valores de 1 a 10 e outra vai de 1000 a 5000, a maior vai "engolir" a menor. O `StandardScaler` resolve isso, deixando todo mundo na mesma escala (média 0 e desvio 1).''')

add_code('''# separando o que é variável (X) do que é a nossa resposta (y)
X = df.drop(['alvo', 'classe'], axis=1)
y = df['alvo']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_treino_padrao = scaler.fit_transform(X_treino)
X_teste_padrao = scaler.transform(X_teste) 

print("Dados padronizados com sucesso!")''')

add_md('''## 4. Treinando o modelo e testando vários 'Ks' ''')

add_code('''resultados = []
valores_k = range(1, 21)

# testando cada K e salvando a acurácia
for k in valores_k:
    modelo_knn = KNeighborsClassifier(n_neighbors=k)
    modelo_knn.fit(X_treino_padrao, y_treino)
    
    previsoes = modelo_knn.predict(X_teste_padrao)
    acuracia = accuracy_score(y_teste, previsoes)
    resultados.append(acuracia)

# tabelinha
df_resultados = pd.DataFrame({'K': valores_k, 'Acurácia': resultados})
df_resultados.head()''')

add_md('''## 5. Escolhendo o melhor K
Vou plotar isso num gráfico pra visualizar melhor onde a acurácia é mais alta.''')

add_code('''plt.figure(figsize=(9, 5))
sns.lineplot(data=df_resultados, x='K', y='Acurácia', marker='o')
plt.title('Acurácia vs Número de Vizinhos (K)')
plt.xticks(valores_k)
plt.grid(True)
plt.show()

melhor_k = df_resultados.loc[df_resultados['Acurácia'].idxmax()]['K']
print(f"O melhor resultado foi com K = {int(melhor_k)}")''')

add_md('''**Por que esse K?**
Olhando o gráfico, a gente pega o ponto mais alto. Se pegar um K muito baixo (tipo 1), o modelo decora os dados de treino e erra no teste (overfitting). Se pegar muito alto, ele fica muito genérico e erra também (underfitting).''')

add_md('''## 6. Modelo Final e Relatório
Agora vou criar o modelo de vez usando o melhor K e ver o relatório completo de como ele se saiu no teste.''')

add_code('''# treinando 
modelo_final = KNeighborsClassifier(n_neighbors=int(melhor_k))
modelo_final.fit(X_treino_padrao, y_treino)

previsoes_finais = modelo_final.predict(X_teste_padrao)

# mostrando as métricas de acerto, precisão, etc
print("--- Resultado Final ---\\n")
print(classification_report(y_teste, previsoes_finais, target_names=dados.target_names))''')

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("knn_breast_cancer.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

