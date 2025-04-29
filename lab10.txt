# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, Embedding, LayerNormalization, MultiHeadAttention, Input

# Load and prepare data
url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
             "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
             "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
             "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
             "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
             "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
             "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
             "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

df = pd.read_csv(url, names=col_names, compression='gzip')

# Sample 5000 rows for faster training
df = df.sample(n=5000, random_state=42)

# Encode categorical features
encoder = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    df[col] = encoder.fit_transform(df[col])

# Convert label into binary: normal = 0, attack = 1
df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal.' else 'attack')
df['label'] = encoder.fit_transform(df['label'])

# Feature engineering
df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
df['total_count'] = df['count'] + df['srv_count']
df['error_rate'] = (df['serror_rate'] + df['rerror_rate']) / 2

# Split dataset
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary to store metrics
metrics = {}

# --------------------------- Machine Learning Models ---------------------------

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
metrics['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'precision': precision_score(y_test, y_pred_dt),
    'recall': recall_score(y_test, y_pred_dt),
    'f1_score': f1_score(y_test, y_pred_dt)
}
print("Decision Tree Report:\n", classification_report(y_test, y_pred_dt))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
metrics['KNN'] = {
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'precision': precision_score(y_test, y_pred_knn),
    'recall': recall_score(y_test, y_pred_knn),
    'f1_score': f1_score(y_test, y_pred_knn)
}
print("KNN Report:\n", classification_report(y_test, y_pred_knn))

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
metrics['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1_score': f1_score(y_test, y_pred_lr)
}
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
metrics['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, y_pred_nb),
    'precision': precision_score(y_test, y_pred_nb),
    'recall': recall_score(y_test, y_pred_nb),
    'f1_score': f1_score(y_test, y_pred_nb)
}
print("Naive Bayes Report:\n", classification_report(y_test, y_pred_nb))

# --------------------------- Deep Learning Models ---------------------------

# Reshape for LSTM and CNN
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTM
lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], 1)),
    LSTM(64, activation='tanh'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=0)

y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype("int32")
metrics['LSTM'] = {
    'accuracy': accuracy_score(y_test, y_pred_lstm),
    'precision': precision_score(y_test, y_pred_lstm),
    'recall': recall_score(y_test, y_pred_lstm),
    'f1_score': f1_score(y_test, y_pred_lstm)
}
print("LSTM Report:\n", classification_report(y_test, y_pred_lstm))

# CNN
cnn_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=0)

y_pred_cnn = (cnn_model.predict(X_test_lstm) > 0.5).astype("int32")
metrics['CNN'] = {
    'accuracy': accuracy_score(y_test, y_pred_cnn),
    'precision': precision_score(y_test, y_pred_cnn),
    'recall': recall_score(y_test, y_pred_cnn),
    'f1_score': f1_score(y_test, y_pred_cnn)
}
print("CNN Report:\n", classification_report(y_test, y_pred_cnn))

# Transformer (simple self-attention)
input_layer = Input(shape=(X_train_lstm.shape[1], 1))
attention = MultiHeadAttention(num_heads=2, key_dim=1)(input_layer, input_layer)
flatten = Flatten()(attention)
output_layer = Dense(1, activation='sigmoid')(flatten)

transformer_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

transformer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
transformer_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=0)

y_pred_transformer = (transformer_model.predict(X_test_lstm) > 0.5).astype("int32")
metrics['Transformer'] = {
    'accuracy': accuracy_score(y_test, y_pred_transformer),
    'precision': precision_score(y_test, y_pred_transformer),
    'recall': recall_score(y_test, y_pred_transformer),
    'f1_score': f1_score(y_test, y_pred_transformer)
}
print("Transformer Report:\n", classification_report(y_test, y_pred_transformer))

# --------------------------- Plotting the Comparison ---------------------------

# Plot metrics
metric_df = pd.DataFrame(metrics).T

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
colors = ['#5DADE2', '#48C9B0', '#F4D03F', '#E67E22']

for idx, metric in enumerate(metric_names):
    ax = axes[idx//2, idx%2]
    metric_df[metric].plot(kind='bar', ax=ax, color=colors[idx])
    ax.set_title(metric.capitalize())
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_xticklabels(metric_df.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()
