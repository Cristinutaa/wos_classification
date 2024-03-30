import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
import sys
# Load the data
train_data = pd.read_csv(sys.argv[0])
test_data = pd.read_csv(sys.argv[1])

# Convert the string representation of list into actual list
train['label'] = train['label'].apply(literal_eval)
test['label'] = test['label'].apply(literal_eval)

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X_train = train['text']
Y_train = mlb.fit_transform(train['label'])

X_test = test['text']
Y_test = mlb.fit_transform(test['label'])

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the vectorizer on the training set, then transform the test set
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize the SVM classifier
svm = SVC(kernel='linear')

# One-vs-Rest strategy to handle multi-label classification
from sklearn.multiclass import OneVsRestClassifier
svm = OneVsRestClassifier(svm)

# Train the classifier
svm.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = svm.predict(X_test_tfidf)

# Evaluate the classifier
print(classification_report(y_test, y_pred, target_names=mlb.classes_))