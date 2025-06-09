import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Função de pré-processamento
def preprocess(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('portuguese')]
    return " ".join(tokens)

# Carregar documentos de todos os CSVs
def load_csv_documents(file_path, column_name):
    data = pd.read_csv(file_path)
    return data[column_name].dropna().tolist()

# Caminhos dos arquivos CSV e nome da coluna de interesse
csv_files = ["solicitacoes_dataset1.csv", "solicitacoes_dataset2.csv", "solicitacoes_dataset3.csv"]
column_name = "solicitacoes"

# Carregar todas as solicitações de todos os CSVs
all_documents = []
for file in csv_files:
    docs = load_csv_documents(file, column_name)
    all_documents.extend(docs)

# Pré-processar todos os documentos
documents_clean = [preprocess(doc) for doc in all_documents]

# Vetorização TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents_clean)

# Exibir os vetores TF-IDF
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print("Vetores TF-IDF (Bag of Words):")
print(df_tfidf)

# Matriz de similaridade cosseno
similarity_matrix = cosine_similarity(tfidf_matrix)

# Exibir a matriz como DataFrame
labels = [f"Solicitação {i+1}" for i in range(len(all_documents))]
df_similarity = pd.DataFrame(similarity_matrix, index=labels, columns=labels)

print("\nMatriz de Similaridade (Cosseno):")
print(df_similarity)

# Identificar solicitações mais similares e mais distintas (usando os textos)
np.fill_diagonal(similarity_matrix, np.nan)
max_sim = np.nanmax(similarity_matrix)
min_sim = np.nanmin(similarity_matrix)
max_pair = np.where(similarity_matrix == max_sim)
min_pair = np.where(similarity_matrix == min_sim)

print(f"\nSolicitações mais similares:")
print(f"Texto 1: {all_documents[max_pair[0][0]]}")
print(f"Texto 2: {all_documents[max_pair[1][0]]}")
print(f"Similaridade = {max_sim:.2f}")

print(f"\nSolicitações mais distintas:")
print(f"Texto 1: {all_documents[min_pair[0][0]]}")
print(f"Texto 2: {all_documents[min_pair[1][0]]}")
print(f"Similaridade = {min_sim:.2f}")

# Visualização com heatmap mais legível
plt.figure(figsize=(max(7, len(all_documents) * 0.5), 5))
ax = sns.heatmap(
    df_similarity,
    annot=False,  # Não mostra os valores dentro das células
    cmap="YlGnBu",
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)
plt.title("Matriz de Similaridade Cosseno entre Solicitações")

# Mostra apenas alguns ticks nos eixos
step = max(1, len(all_documents) // 10)  # Mostra no máximo 10 ticks
ax.set_xticks(np.arange(0, len(all_documents), step))
ax.set_yticks(np.arange(0, len(all_documents), step))
ax.set_xticklabels([f"S{i+1}" for i in range(0, len(all_documents), step)], rotation=90)
ax.set_yticklabels([f"S{i+1}" for i in range(0, len(all_documents), step)], rotation=0)

plt.tight_layout()
plt.show()

# Exibe um índice para consulta dos textos
print("\nÍndice das Solicitações:")
for idx, texto in enumerate(all_documents):
    print(f"Solicitação {idx+1}: {texto}")