import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data from CSV file
data = pd.read_csv('tweets_final.csv')

# Extract relevant columns
tweets_data = data['text']
data['type'] = data['type'].str.replace(r"[\[\]']", '', regex=True)
categories = data['type']

# Function to clean the text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters and punctuation
    return text

# Apply the function to the tweets
tweets_data = tweets_data.apply(clean_text)

# Preprocess the data
tokenizer = Tokenizer(num_words=5000)  # Use the top 5000 words
tokenizer.fit_on_texts(tweets_data)
X_seq = tokenizer.texts_to_sequences(tweets_data)
X_pad = pad_sequences(X_seq, maxlen=100)

# Convert labels to numerical values
label_dict = {'vulnerability': 0, 'ransomware': 1, 'ddos': 2, 'leak': 3, 'general': 4, '0day': 5, 'botnet': 6, 'all': 7}
y = np.array([label_dict[category] for category in categories])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Load the GloVe embeddings
def load_glove_embeddings(glove_file, embedding_dim):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients
    return embeddings_index

# Path to the downloaded GloVe file (e.g., glove.6B.100d.txt)
glove_file = '../glove/glove.6B.100d.txt'
embedding_dim = 100  # Dimensions of GloVe vectors

# Load GloVe embeddings
embeddings_index = load_glove_embeddings(glove_file, embedding_dim)

# Prepare embedding matrix
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector  # Words not found in embedding index will be all-zeros.

# Build the CNN model with pre-trained GloVe embeddings
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, 
                    output_dim=embedding_dim, 
                    weights=[embedding_matrix], 
                    input_length=100, 
                    trainable=False))  # Use pre-trained embeddings, do not train

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))  # Add Dropout to prevent overfitting
model.add(Dense(8, activation='softmax'))  # Output layer with 8 classes

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Generate predictions and plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()