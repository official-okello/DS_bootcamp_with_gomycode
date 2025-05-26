import re
import nltk
import contractions
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Downloading necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ----------------------------------------------------------------Importing and Exploring Data-------------------------------------------------------------

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def explore_data(data):
    print("Dataset Overview:")
    print(data.head())
    print("\nDataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())
    print("\nSentiment Distribution:")
    print(data['sentiment'].value_counts())

# ----------------------------------------------------------------Data Processing----------------------------------------------------------------

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = contractions.fix(text)  # Expand contractions
    
    # Only parse with BeautifulSoup if it looks like HTML
    if '<' in text and '>' in text:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
            text = BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    tokens = word_tokenize(text.lower())  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]  # Lemmatization

    return ' '.join(tokens)

def preprocess_data(data):
    data['cleaned_text'] = data['review'].apply(clean_text)
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0}) # Convert sentiments to 0 and 1
    return data

# Converting cleaned text to TF-IDF features
def vectorize_text(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['cleaned_text'])
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()  # Convert to NumPy array

# ----------------------------------------------------------Model Building and Compiling-----------------------------------------------------------

# Splitting the dataset into training and testing sets
def split_data(data):
    # Split text data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['sentiment'], test_size=0.2, random_state=42)

    # Convert training & testing text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)  # Reduce feature space
    tfidf_matrix_train = vectorizer.fit_transform(X_train)
    tfidf_matrix_test = vectorizer.transform(X_test)

    # Convert sparse matrix to NumPy array
    X_train = tfidf_matrix_train if isinstance(tfidf_matrix_train, np.ndarray) else tfidf_matrix_train.toarray()
    X_test = tfidf_matrix_test if isinstance(tfidf_matrix_test, np.ndarray) else tfidf_matrix_test.toarray()

    X_train = tfidf_matrix_train.astype(np.float32)
    X_test = tfidf_matrix_test.astype(np.float32)

    # Ensure y_train & y_test are NumPy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test

# Building the model
def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model
    

# Training the model
def train_model(model, X_train, y_train):
    
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    return history

# Evaluating the model
def evaluate_model(model, X_test, X_train, y_train, y_test, history):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Converting probabilities to binary predictions
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Number of parameters: {model.count_params()}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Number of epochs: {len(history.history['loss'])}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {model.evaluate(X_test, y_test)[0]:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix visualization
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Training History Plotting
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------Main Function----------------------------------------------------------------
def main():
    # Load and explore the dataset
    file_path = 'C:/Users/USER/OneDrive/Desktop/DS_GOMYCODE/ML/datasets/IMDB Dataset.csv'  # Dataset path
    data = load_data(file_path)
    if data is not None:
        explore_data(data)

        # Preprocess the data
        data = preprocess_data(data)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(data)

        # Build the model
        model = build_model(X_train.shape[1])

        # Train the model
        history = train_model(model, X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, X_train, y_train, y_test, history)

        # Plot training history
        plot_training_history(history)

if __name__ == "__main__":
    main()



# ----------------------------------------------------------------Reports----------------------------------------------------------------
# The script is designed to classify movie reviews as positive or negative using an artificial neural network.
# It includes data loading, preprocessing, model building, training, and evaluation.
# The model uses TF-IDF vectorization for text representation and a sequential neural network architecture.
# The evaluation metrics include accuracy, loss, and a classification report.
# The script also visualizes the training history and confusion matrix.
# The model is trained on a dataset of movie reviews, achieving a good accuracy on the test set.
# The script is modular, allowing for easy modifications and extensions.
# The model can be further improved by tuning hyperparameters, adding more layers, or using different architectures.



---

# ## 🔍 **Insights Gained from the Project**

''' 1. **Importance of Text Preprocessing:**

   * Text preprocessing (removing noise, tokenization, lemmatization, etc.) had a significant impact on model performance.
   * Expanding contractions and removing HTML tags improved text clarity and feature quality.

    2. **TF-IDF + Neural Networks Work Well:**

    * Using `TfidfVectorizer` provided a solid representation of text features.
    * A simple feedforward neural network achieved decent classification performance even without sequence models (like LSTMs or Transformers).


    3. **Binary Classification Is Achievable with Simpler Models:**

    * Even with relatively shallow dense layers, the model was able to classify IMDB reviews as positive or negative effectively.

    4. **Data Imbalance Can Skew Model Behavior:**

    * Observing the distribution of sentiment classes emphasized the need to check for balanced datasets, which affects generalization.

    5. **Model Evaluation Is Crucial:**
    * Using metrics like accuracy, precision, recall, and F1-score provided a comprehensive view of model performance.
    * Confusion matrix visualization helped identify specific misclassifications, guiding further improvements.

# # ## 🛠️ **Key Features Implemented**
# # | **Feature**                          | **Description**                                                                                     |
# # | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
# # | **Data Loading**                      | Loads the IMDB dataset from a CSV file and handles file not found errors gracefully.                 |
# # | **Data Exploration**                  | Displays dataset overview, info, description, and sentiment distribution.                            |
# # | **Text Preprocessing**                | Cleans text by expanding contractions, removing HTML tags, URLs, and normalizing whitespace.        |
# # | **Tokenization and Lemmatization**    | Tokenizes text, removes stopwords, and lemmatizes words to reduce dimensionality.                   |
# # | **TF-IDF Vectorization**              | Converts cleaned text into numerical features using `TfidfVectorizer`, reducing feature space.       |
# # | **Model Building**                    | Constructs a sequential neural network with dense layers, dropout for regularization, and sigmoid output. |
# # | **Model Training**                    | Trains the model on the training set with validation split to monitor overfitting.                   |
# # | **Model Evaluation**                  | Evaluates model performance using accuracy, loss, classification report, and confusion matrix.       |
# # | **Training History Visualization**    | Plots training and validation loss and accuracy over epochs to visualize model performance.         |


# # ## 📊 **Data Visualization**
# # * **Sentiment Distribution**: Visualized the distribution of positive and negative reviews to understand class balance.
# # * **Confusion Matrix**: Used a heatmap to visualize true vs predicted sentiments, helping identify misclassifications.
# # * **Training History**: Plotted training and validation loss and accuracy to monitor model performance over epochs.

# # ## 📝 **Documentation and Comments**
# # * The code is well-commented, explaining each step of the process from data loading to model evaluation.
# # * Function names are descriptive, making it easy to understand their purpose.

# # ## ⚙️ **Technologies Used**
# # * **Python Libraries**:
# #   - `pandas` for data manipulation
# #   - `numpy` for numerical operations
# #   - `nltk` for natural language processing tasks
# #   - `BeautifulSoup` for HTML parsing
# #   - `sklearn` for machine learning utilities (train-test split, vectorization, metrics)
# #   - `tensorflow` for building and training the neural network model
# # * **Modeling Framework**: TensorFlow/Keras for building and training the neural network.
# # * **Visualization**: `matplotlib` and `seaborn` for plotting training history and confusion matrix.

# # ## 📈 **Performance Metrics**
# # * **Accuracy**: The model achieved a good accuracy on the test set, indicating effective classification.
# # * **Loss**: The loss decreased over epochs, showing that the model was learning.
# # * **Classification Report**: Provided precision, recall, and F1-score for both classes, giving a detailed view of model performance.

# # ## ⚠️ **Challenges Faced & How They Were Overcome**
# # | **Challenge**                                                 | **Solution**                                                                                                                                     |
# # | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
# # | **1. TensorFlow "Cast string to float" error**                | Ensured labels (`y_train`, `y_test`) were of numeric type (`int` or `float`), not strings. Used `astype(int)` or `LabelEncoder` where necessary. |
# # | **2. Sparse matrix `len()` error**                            | Replaced `len(X_train)` with `X_train.shape[0]` to avoid ambiguity in sparse matrix dimensions.                                                  |
# # | **3. Keras warning for `input_shape` in `Dense` layer**       | Used `Input(shape=(input_dim,))` explicitly or adjusted layer setup to avoid deprecated usage.                                                   |
# # | **4. BeautifulSoup warning: "MarkupResemblesLocatorWarning"** | Ensured `text` inputs to `BeautifulSoup` were strings, not file paths. Verified that data was read properly from the CSV.                        |
# # | **5. Model overfitting on training data**                     | Added `Dropout` layers and used `validation_split` to monitor and control overfitting during training.                                           |

# # ---

# # ## 🚀 **Potential Improvements**

# # ### 🔁 **1. Preprocessing Enhancements**

# # * **Spell Correction**: Integrate tools like `TextBlob` or `SymSpell` to correct misspellings.
# # * **Named Entity Removal**: Remove proper nouns that don’t affect sentiment but add noise.
# # * **Negation Handling**: Replace `"not good"` with `"not_good"` to better capture sentiment nuances.

# # ### 📈 **2. Feature Engineering**

# # * **n-grams**: Use bigrams/trigrams in `TfidfVectorizer` to capture context (`ngram_range=(1,2)` or `(1,3)`).
# # * **Custom Stopword Lists**: Fine-tune stopword removal by retaining domain-specific terms.

# # ### 🧠 **3. Model Architecture**

# # * **Try LSTM/GRU Layers**: Use `Embedding` + `LSTM` layers for sequential data understanding (especially for longer reviews).
# # * **Hyperparameter Tuning**: Use `GridSearchCV` (with wrapper classes or `KerasTuner`) to tune units, layers, dropout, etc.

# # ### 🧪 **4. Evaluation Improvements**

# # * **Cross-validation**: Use k-fold CV to ensure robust performance instead of relying on one train/test split.
# # * **Precision/Recall Optimization**: Depending on use case (e.g. content moderation), optimize for precision or recall rather than just accuracy.

# # ### 💾 **5. Model Saving and Inference**

# # * Save model and vectorizer with `joblib` or `pickle` for real-world deployment.
# # * Create a function to accept raw text input, preprocess it, vectorize, and predict using the trained model.

# ---'''
