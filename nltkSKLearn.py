import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer as CountSKLearn
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np


# Cargar datos
data = pd.read_csv('climateTwitterData.csv', low_memory=False)

# Instanciar TweetTokenizer
tweet_tokenizer = TweetTokenizer()

# Aplicar TweetTokenizer a los textos
data['tokenized'] = data['text'].apply(lambda x: ' '.join(tweet_tokenizer.tokenize(x)))

# Crear el Bag of Words
vectorizer = CountSKLearn()
bow_matrix = vectorizer.fit_transform(data['tokenized'])

# Mostrar el tamaño de la matriz çv

print(bow_matrix.shape)
print(list(vectorizer.vocabulary_.items())[:10])

# Instanciar y aplicar Truncated SVD
svd = TruncatedSVD(n_components=1000, random_state=42)
bow_reduced = svd.fit_transform(bow_matrix)

# Calcular y mostrar la varianza explicada acumulada
explained_variance = svd.explained_variance_ratio_.sum()
print(f"Dimensiones reducidas: {bow_reduced.shape}")
print(f"Varianza explicada acumulada: {explained_variance:.2%}")

# Graficar la varianza explicada por cada componente
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.title('Varianza explicada acumulada por Truncated SVD')
plt.grid(True)
plt.show()

# Reducir nuevamente los datos a 2 componentes para visualización
svd_2d = TruncatedSVD(n_components=2, random_state=42)
bow_reduced_2d = svd_2d.fit_transform(bow_matrix)

# Graficar los dos primeros componentes principales
plt.figure(figsize=(8, 6))
plt.scatter(bow_reduced_2d[:, 0], bow_reduced_2d[:, 1], alpha=0.5)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Visualización de Tweets sobre Clima usando Truncated SVD')
plt.show()