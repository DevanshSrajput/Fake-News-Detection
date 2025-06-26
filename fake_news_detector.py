import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class FakeNewsDetector:
    """
    A comprehensive fake news detection system using machine learning
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.pac_model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        self.svm_model = SVC(kernel='linear', random_state=42)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_dataset_from_csv(self, file_path, text_col='text', label_col='label', sample_size=None):
        """
        Load dataset from a CSV file.
        The CSV must have a text column and a label column.
        """
        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        
        print(f"Original dataset size: {len(df)}")
        
        # Sample the dataset if it's too large
        if sample_size and len(df) > sample_size:
            print(f"Sampling {sample_size} rows from {len(df)} total rows...")
            df = df.sample(n=sample_size, random_state=42)
        
        # Drop missing values
        initial_size = len(df)
        df = df.dropna(subset=[text_col, label_col])
        print(f"Dropped {initial_size - len(df)} rows with missing values")
        
        # Standardize label values if needed (e.g., 0/1 or FAKE/REAL)
        df[label_col] = df[label_col].astype(str).str.upper()
        
        # Only keep required columns
        df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
        print(f"Final dataset size: {len(df)} articles")
        print(f"Label counts:\n{df['label'].value_counts()}")
        return df

    def preprocess_text_batch(self, texts):
        """
        Optimized batch text preprocessing
        """
        processed_texts = []
        
        print("Preprocessing text data...")
        for text in tqdm(texts, desc="Processing articles", unit="article"):
            try:
                if pd.isna(text) or text == "":
                    processed_texts.append("")
                    continue
                    
                # Convert to string and lowercase
                text = str(text).lower()
                
                # Remove HTML tags
                text = re.sub(r'<.*?>', '', text)
                
                # Remove URLs
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                
                # Remove punctuation
                text = text.translate(str.maketrans('', '', string.punctuation))
                
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Skip tokenization for empty texts
                if not text:
                    processed_texts.append("")
                    continue
                
                # Tokenize (with fallback)
                try:
                    tokens = word_tokenize(text)
                except:
                    tokens = text.split()
                
                # Remove stopwords and stem
                processed_tokens = []
                for token in tokens:
                    if token not in self.stop_words and len(token) > 2:
                        try:
                            stemmed_token = self.stemmer.stem(token)
                            processed_tokens.append(stemmed_token)
                        except:
                            processed_tokens.append(token)
                
                processed_texts.append(' '.join(processed_tokens))
                
            except Exception as e:
                print(f"Error processing text: {str(e)[:100]}")
                processed_texts.append("")
        
        return processed_texts

    def preprocess_text(self, text):
        """
        Single text preprocessing (kept for compatibility)
        """
        if pd.isna(text) or text == "":
            return ""
            
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                return ""
            
            # Tokenize
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Remove stopwords and stem
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    try:
                        stemmed_token = self.stemmer.stem(token)
                        processed_tokens.append(stemmed_token)
                    except:
                        processed_tokens.append(token)
            
            return ' '.join(processed_tokens)
        except:
            return ""
    
    def prepare_data(self, df):
        """
        Prepare data for machine learning with optimized preprocessing
        """
        print(f"Starting data preparation with {len(df)} articles...")
        
        # Use batch preprocessing for better performance
        df['processed_text'] = self.preprocess_text_batch(df['text'].tolist())
        
        # Remove empty processed texts
        initial_count = len(df)
        df = df[df['processed_text'].str.len() > 0]
        print(f"Removed {initial_count - len(df)} articles with empty processed text")
        
        if len(df) == 0:
            raise ValueError("No valid articles remaining after preprocessing!")
        
        # Convert labels to binary
        unique_labels = set(df['label'].unique())
        print(f"Unique labels found: {unique_labels}")
        
        # Handle different label formats
        label_map = {}
        for label in unique_labels:
            label_str = str(label).upper()
            if label_str in ['REAL', 'TRUE', '0']:
                label_map[label] = 0
            elif label_str in ['FAKE', 'FALSE', '1']:
                label_map[label] = 1
            else:
                print(f"Warning: Unknown label '{label}', mapping to FAKE")
                label_map[label] = 1
        
        print(f"Label mapping: {label_map}")
        df['label_binary'] = df['label'].map(label_map)
        
        # Drop rows with undefined labels
        df = df.dropna(subset=['label_binary'])
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after label mapping!")
        
        print(f"Final dataset size: {len(df)}")
        print(f"Label distribution: {df['label_binary'].value_counts().to_dict()}")
        
        # Split data
        X = df['processed_text']
        y = df['label_binary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print("Vectorizing text data...")
        # Vectorize text using TF-IDF
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        print(f"Training set size: {X_train_tfidf.shape}")
        print(f"Test set size: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train both Passive Aggressive and SVM models
        """
        print("Training Passive Aggressive Classifier...")
        self.pac_model.fit(X_train, y_train)
        
        print("Training SVM Classifier...")
        self.svm_model.fit(X_train, y_train)
        
        print("Models trained successfully!")
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        print(f"\n=== {model_name} Evaluation ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return accuracy, f1, cm, y_pred
    
    def plot_confusion_matrix(self, cm, model_name):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['REAL', 'FAKE'], 
                   yticklabels=['REAL', 'FAKE'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, pac_metrics, svm_metrics):
        """
        Compare model performance
        """
        print("\n=== Model Comparison ===")
        
        comparison_df = pd.DataFrame({
            'Model': ['Passive Aggressive', 'SVM'],
            'Accuracy': [pac_metrics[0], svm_metrics[0]],
            'F1 Score': [pac_metrics[1], svm_metrics[1]]
        })
        
        print(comparison_df)
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        ax1.bar(comparison_df['Model'], comparison_df['Accuracy'], color=['skyblue', 'lightcoral'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # F1 Score comparison
        ax2.bar(comparison_df['Model'], comparison_df['F1 Score'], color=['lightgreen', 'orange'])
        ax2.set_title('Model F1 Score Comparison')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def predict_article(self, article_text):
        """
        Predict if a single article is real or fake
        """
        # Preprocess the article
        processed_article = self.preprocess_text(article_text)
        
        if not processed_article:
            return {'error': 'Text could not be processed or is empty'}
        
        # Vectorize
        article_tfidf = self.tfidf_vectorizer.transform([processed_article])
        
        # Make predictions with both models
        pac_prediction = self.pac_model.predict(article_tfidf)[0]
        svm_prediction = self.svm_model.predict(article_tfidf)[0]
        
        # Get prediction probabilities (for PAC, we'll use decision function)
        pac_confidence = abs(self.pac_model.decision_function(article_tfidf)[0])
        
        results = {
            'pac_prediction': 'FAKE' if pac_prediction == 1 else 'REAL',
            'svm_prediction': 'FAKE' if svm_prediction == 1 else 'REAL',
            'pac_confidence': pac_confidence,
            'consensus': 'FAKE' if (pac_prediction + svm_prediction) >= 1 else 'REAL'
        }
        
        return results

def main():
    """
    Main execution function
    """
    print("=== Fake News Detection System ===")
    print("Initializing detector...")
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Configuration
    DATA_PATH = "WELFake_Dataset.csv"
    TEXT_COLUMN = "text"
    LABEL_COLUMN = "label"
    SAMPLE_SIZE = 10000  # Limit dataset size for faster processing - set to None for full dataset

    try:
        # Load dataset
        df = detector.load_dataset_from_csv(
            DATA_PATH, 
            text_col=TEXT_COLUMN, 
            label_col=LABEL_COLUMN,
            sample_size=SAMPLE_SIZE
        )
        
        print("\nSample of the dataset:")
        print(df.head())
        
        # Prepare data
        X_train, X_test, y_train, y_test = detector.prepare_data(df)
        
        # Train models
        detector.train_models(X_train, y_train)
        
        # Evaluate Passive Aggressive Classifier
        pac_accuracy, pac_f1, pac_cm, pac_pred = detector.evaluate_model(
            detector.pac_model, X_test, y_test, "Passive Aggressive Classifier"
        )
        
        # Evaluate SVM
        svm_accuracy, svm_f1, svm_cm, svm_pred = detector.evaluate_model(
            detector.svm_model, X_test, y_test, "Support Vector Machine"
        )
        
        # Plot confusion matrices
        detector.plot_confusion_matrix(pac_cm, "Passive Aggressive Classifier")
        detector.plot_confusion_matrix(svm_cm, "Support Vector Machine")
        
        # Compare models
        detector.compare_models(
            (pac_accuracy, pac_f1), (svm_accuracy, svm_f1)
        )
        
        # Test with sample articles
        print("\n=== Testing with Sample Articles ===")
        
        test_articles = [
            "The stock market experienced significant volatility today as investors reacted to new economic data.",
            "SHOCKING: Scientists discover that eating pizza cures all diseases instantly! Doctors hate this one trick!"
        ]
        
        for i, article in enumerate(test_articles, 1):
            print(f"\nTest Article {i}: {article[:60]}...")
            results = detector.predict_article(article)
            if 'error' not in results:
                print(f"PAC Prediction: {results['pac_prediction']}")
                print(f"SVM Prediction: {results['svm_prediction']}")
                print(f"Consensus: {results['consensus']}")
                print(f"PAC Confidence: {results['pac_confidence']:.4f}")
            else:
                print(f"Error: {results['error']}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()