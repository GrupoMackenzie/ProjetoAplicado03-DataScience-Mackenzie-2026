import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Carrega os datasets iniciais e faz uma limpeza
# =========================

steam = pd.read_csv('./datasets/steamgames.csv')
req = pd.read_csv('./datasets/pc_requirements.csv')

print("="*50)
print("DADOS CARREGADOS")
print("="*50)
print("Steam:", steam.shape)
print("Requirements:", req.shape)

print("\nCOLUNAS STEAM:")
print(steam.columns.tolist())

print("\nCOLUNAS REQUIREMENTS:")
print(req.columns.tolist())

pos_col = 'Positive'
neg_col = 'Negative'

steam[pos_col] = pd.to_numeric(steam[pos_col], errors='coerce')
steam[neg_col] = pd.to_numeric(steam[neg_col], errors='coerce')

steam = steam.dropna(subset=[pos_col, neg_col])

steam['Price'] = pd.to_numeric(steam['Price'], errors='coerce')

print("\nAPÓS LIMPEZA:")
print("Steam:", steam.shape)

steam['total_ratings'] = steam[pos_col] + steam[neg_col]

# Fiz essa função para evitar divisões por zero
steam['score'] = np.where(
    steam['total_ratings'] > 0,
    steam[pos_col] / steam['total_ratings'],
    0
)

# =========================
# 5. EDA
# =========================

print("\nRESUMO ESTATÍSTICO:")
print(steam[['Price','score','total_ratings']].describe())

top_games = steam.sort_values(by='score', ascending=False).head(10)

print("\nTOP 10 JOGOS POR SCORE:")
print(top_games[['Name','score','total_ratings']])

# =========================
# 6. Visualizações para Gabriel
# =========================

plt.figure()
sns.histplot(steam['score'], bins=30)
plt.title('Distribuição de Score dos Jogos')
plt.xlabel('Score')
plt.ylabel('Frequência')
plt.show()

plt.figure()
sns.histplot(steam['Price'], bins=30)
plt.title('Distribuição de Preços')
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.show()

plt.figure()
sns.scatterplot(x='Price', y='score', data=steam)
plt.title('Preço vs Avaliação')
plt.show()

# =========================
# 7. Normaliza
# =========================

steam['Name'] = steam['Name'].astype(str).str.lower().str.strip()
req['Name'] = req['Name'].astype(str).str.lower().str.strip()

# =========================
# 8. Integra os datasets
# =========================

df = pd.merge(steam, req, on='Name', how='inner')

print("\nDADOS INTEGRADOS:")
print("Shape final:", df.shape)

print("\nAMOSTRA DO DATASET FINAL:")
print(df[['Name','score','Min_RAM','Recom_RAM']].head())

print("\nRESUMO RAM:")
print(df[['Min_RAM','Recom_RAM']].describe())

# =========================
# 10. Exporta
# =========================

df.to_csv('./datasets./dataset_final.csv', index=False)

print("\nArquivo 'dataset_final.csv' gerado com sucesso na pasta datasets!")

print("\nINSIGHT:")

media_score = steam['score'].mean()
print(f"Score médio dos jogos: {media_score:.2f}")

jogos_caro = steam[steam['Price'] > steam['Price'].median()]
jogos_barato = steam[steam['Price'] <= steam['Price'].median()]

print("Score médio jogos caros:", jogos_caro['score'].mean())
print("Score médio jogos baratos:", jogos_barato['score'].mean())