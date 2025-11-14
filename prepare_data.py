import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#
# Load your dataset
df = pd.read_csv("sample_101_IMDB_10000.csv")

# Clean column names if necessary
df.columns = [c.strip().lower() for c in df.columns]
print("Columns:", df.columns.tolist())

# Choose relevant columns available in your dataset
columns_to_use = ['title', 'genre', 'desc']
df = df[columns_to_use].dropna().reset_index(drop=True)

#  
# Combine textual features
df['combined'] = df['genre'] + " " + df['desc']

# 
# Compute TF-IDF vectors and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix)

# 
# Save processed data
pickle.dump(df, open("movie_data.pkl", "wb"))
pickle.dump(cosine_sim, open("cosine_sim.pkl", "wb"))

print("✅ Saved movie_data.pkl and cosine_sim.pkl successfully.")
