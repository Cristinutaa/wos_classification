from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import ast
from joblib import Parallel, delayed

def process_chunk(chunk_indices, text_features, labels_df, nn, n_neighbors=11):
    chunk_interactions = defaultdict(lambda: Counter())
    for idx in chunk_indices:
        current_labels = set(data.iloc[idx]['label'])
        _, neighbors = nn.kneighbors(text_features[idx:idx+1], n_neighbors=n_neighbors)

        # Exclude the current instance from the neighbors
        neighbors = [n for n in neighbors[0] if n != idx]

        # Aggregate labels of the nearest neighbors
        neighbor_labels = labels_df.iloc[neighbors].sum().to_dict()

        # Collect and return the interactions for this instance
        for label in current_labels:
            for other_label, count in neighbor_labels.items():
                chunk_interactions[label][other_label] += count
    return chunk_interactions

# Load the dataset
file_path = '../WoS_data/different_size/train100000.csv'
data = pd.read_csv(file_path)
data['label'] = data['label'].apply(ast.literal_eval)

# Prepare TF-IDF features
tfidf = TfidfVectorizer(max_df=0.6, stop_words='english', ngram_range=(1,3), max_features=40000)
text_features = tfidf.fit_transform(data['text'])

# MultiLabel binarization
mlb = MultiLabelBinarizer()
labels_encoded = mlb.fit_transform(data['label'])
labels_df = pd.DataFrame(labels_encoded, columns=mlb.classes_)

# Fit Nearest Neighbors on the entire dataset
nn = NearestNeighbors(n_neighbors=11, metric='cosine')  # Adjusted n_neighbors to 6
nn.fit(text_features)

# Initialize a structure to store interactions between labels
label_interactions = defaultdict(lambda: Counter())

# Divide the dataset indices into chunks
n_jobs = -1  # Use all available cores, adjust based on your system
chunk_size = 35  # Example chunk size, adjust based on your needs
chunks = [range(i, min(i + chunk_size, len(data))) for i in range(0, len(data), chunk_size)]

# Process chunks in parallel
results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk, text_features, labels_df, nn) for chunk in tqdm(chunks))

# Initialize a structure to store aggregated interactions between labels
label_interactions = defaultdict(lambda: Counter())

# Aggregate results from all chunks
for chunk_result in results:
    for label, interactions in chunk_result.items():
        for other_label, count in interactions.items():
            label_interactions[label][other_label] += count
            
# Calculate mean interactions for each label
label_publication_counts = labels_df.sum()
label_means = {}
for label, interactions in label_interactions.items():
    label_total_publications = label_publication_counts[label]  # Get the total publications for the label
    label_means[label] = {other_label: count / label_total_publications for other_label, count in interactions.items()}

# Rank other disciplines for each discipline based on mean interactions
ranked_disciplines = {label: sorted(means.items(), key=lambda x: x[1], reverse=True) for label, means in label_means.items()}

# Display the ranked disciplines
for label, ranks in ranked_disciplines.items():
    print(f"\nDiscipline: {label}")
    for other_label, mean_count in ranks:
        print(f"  {other_label}: {mean_count}")