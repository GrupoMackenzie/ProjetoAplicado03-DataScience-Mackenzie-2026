import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 1. Dataset Simulado de Builds (Combinações de CPU + GPU + RAM)
# Pontuações de 0 a 100 baseadas em benchmarks sintéticos
data = {
    'Build_ID': [1, 2, 3, 4, 5],
    'CPU_Score': [40, 85, 90, 30, 75],
    'GPU_Score': [45, 80, 40, 95, 70], # Build 3 e 4 têm gargalos óbvios
    'RAM_GB': [8, 16, 32, 16, 16],
    'Price_USD': [400, 1200, 1000, 900, 1000]
}
df = pd.DataFrame(data)

# 2. Engenharia de Features: Calculando o Gargalo (Bottleneck)
# Um gargalo alto significa grande disparidade entre CPU e GPU
df['Bottleneck_Penalty'] = np.abs(df['CPU_Score'] - df['GPU_Score'])

# 3. Filtragem Baseada em Conhecimento (Regras)
# Descartamos builds com gargalo muito alto (ex: diferença de score > 15)
df_balanced = df[df['Bottleneck_Penalty'] <= 15].reset_index(drop=True)

# 4. Modelo de Recomendação (Encontrar a build mais próxima do orçamento e perfil)
# Features para o KNN: [CPU_Score, GPU_Score, Price]
features = df_balanced[['CPU_Score', 'GPU_Score', 'Price_USD']]

# Treinando o modelo para encontrar a build mais próxima
knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
knn.fit(features.values)

# 5. Testando o Modelo: O usuário quer gastar $1100 e quer alta performance (~80 pontos)
user_request = [[90, 90, 400]]
distances, indices = knn.kneighbors(user_request)

recommended_build = df_balanced.iloc[indices[0][0]]
print("Build Recomendada:\n", recommended_build)
print(distances)