import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import json
import pickle
from typing import Dict, List, Tuple, Any

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CustomerSupportTicketClassifier:
    """
    A comprehensive machine learning pipeline for customer support ticket classification
    """
    
    def __init__(self):
        self.issue_type_model = None
        self.urgency_model = None
        self.tfidf_vectorizer = None
        self.label_encoders = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common product names for entity extraction
        self.product_list = [
            'laptop', 'computer', 'phone', 'tablet', 'monitor', 'keyboard', 'mouse',
            'printer', 'scanner', 'router', 'modem', 'speaker', 'headphones',
            'camera', 'software', 'application', 'app', 'website', 'portal',
            'system', 'server', 'database', 'network', 'wifi', 'internet'
        ]
        
        # Complaint keywords
        self.complaint_keywords = [
            'broken', 'damaged', 'not working', 'error', 'bug', 'issue', 'problem',
            'slow', 'delayed', 'late', 'crashed', 'frozen', 'stuck', 'failed',
            'missing', 'lost', 'corrupted', 'unavailable', 'down', 'offline'
        ]
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and apply lemmatization
        """
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from ticket text
        """
        df_features = df.copy()
        
        # Text length features
        df_features['text_length'] = df_features['ticket_text'].str.len()
        df_features['word_count'] = df_features['ticket_text'].str.split().str.len()
        
        # Sentiment analysis
        df_features['sentiment_polarity'] = df_features['ticket_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity if x else 0
        )
        df_features['sentiment_subjectivity'] = df_features['ticket_text'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity if x else 0
        )
        
        # Count of complaint keywords
        df_features['complaint_keyword_count'] = df_features['ticket_text'].apply(
            self.count_complaint_keywords
        )
        
        # Count of uppercase words (might indicate urgency)
        df_features['uppercase_count'] = df_features['ticket_text'].apply(
            lambda x: len(re.findall(r'\b[A-Z]{2,}\b', x)) if x else 0
        )
        
        return df_features
    
    def count_complaint_keywords(self, text: str) -> int:
        """
        Count complaint keywords in text
        """
        if not text:
            return 0
        
        text_lower = text.lower()
        count = 0
        for keyword in self.complaint_keywords:
            if keyword in text_lower:
                count += 1
        return count
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from ticket text using rule-based approach
        """
        entities = {
            'products': [],
            'dates': [],
            'complaint_keywords': []
        }
        
        if not text:
            return entities
        
        text_lower = text.lower()
        
        # Extract product names
        for product in self.product_list:
            if product in text_lower:
                entities['products'].append(product)
        
        # Extract dates using regex
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2}\b',    # MM/DD/YY or MM-DD-YY
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend(matches)
        
        # Extract complaint keywords
        for keyword in self.complaint_keywords:
            if keyword in text_lower:
                entities['complaint_keywords'].append(keyword)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and preprocess the dataset
        """
        print("Starting data preparation...")
        
        # Handle missing values
        df = df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level'])
        
        # Clean text
        df['cleaned_text'] = df['ticket_text'].apply(self.clean_text)
        
        # Extract additional features
        df = self.extract_features(df)
        
        # Tokenize and lemmatize
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        print(f"Data preparation complete. Shape: {df.shape}")
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train classification models for issue type and urgency level
        """
        print("Starting model training...")
        
        # Prepare features
        X_text = df['processed_text']
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
        
        # Additional features
        additional_features = df[['text_length', 'word_count', 'sentiment_polarity', 
                                'sentiment_subjectivity', 'complaint_keyword_count', 
                                'uppercase_count']].values
        
        # Combine TF-IDF with additional features
        X_combined = np.hstack([X_tfidf.toarray(), additional_features])
        
        # Encode labels
        self.label_encoders['issue_type'] = LabelEncoder()
        self.label_encoders['urgency_level'] = LabelEncoder()
        
        y_issue = self.label_encoders['issue_type'].fit_transform(df['issue_type'])
        y_urgency = self.label_encoders['urgency_level'].fit_transform(df['urgency_level'])
        
        # Split data
        X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
            X_combined, y_issue, y_urgency, test_size=0.2, random_state=42, stratify=y_issue
        )
        
        # Train Issue Type Classifier
        print("Training Issue Type Classifier...")
        self.issue_type_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        self.issue_type_model.fit(X_train, y_issue_train)
        
        # Train Urgency Level Classifier
        print("Training Urgency Level Classifier...")
        self.urgency_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        self.urgency_model.fit(X_train, y_urgency_train)
        
        # Evaluate models
        issue_pred = self.issue_type_model.predict(X_test)
        urgency_pred = self.urgency_model.predict(X_test)
        
        # Calculate metrics
        issue_accuracy = accuracy_score(y_issue_test, issue_pred)
        urgency_accuracy = accuracy_score(y_urgency_test, urgency_pred)
        
        print(f"Issue Type Classifier Accuracy: {issue_accuracy:.4f}")
        print(f"Urgency Level Classifier Accuracy: {urgency_accuracy:.4f}")
        
        # Detailed classification reports
        issue_report = classification_report(
            y_issue_test, issue_pred, 
            target_names=self.label_encoders['issue_type'].classes_,
            output_dict=True
        )
        
        urgency_report = classification_report(
            y_urgency_test, urgency_pred,
            target_names=self.label_encoders['urgency_level'].classes_,
            output_dict=True
        )
        
        results = {
            'issue_accuracy': issue_accuracy,
            'urgency_accuracy': urgency_accuracy,
            'issue_report': issue_report,
            'urgency_report': urgency_report,
            'test_data': {
                'X_test': X_test,
                'y_issue_test': y_issue_test,
                'y_urgency_test': y_urgency_test,
                'issue_pred': issue_pred,
                'urgency_pred': urgency_pred
            }
        }
        
        print("Model training complete!")
        return results
    
    def predict_ticket(self, ticket_text: str) -> Dict[str, Any]:
        """
        Predict issue type, urgency level, and extract entities for a single ticket
        """
        # Clean and preprocess text
        cleaned_text = self.clean_text(ticket_text)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        processed_text = ' '.join(tokens)
        
        # Extract additional features
        text_length = len(ticket_text)
        word_count = len(ticket_text.split())
        sentiment = TextBlob(ticket_text)
        sentiment_polarity = sentiment.sentiment.polarity
        sentiment_subjectivity = sentiment.sentiment.subjectivity
        complaint_keyword_count = self.count_complaint_keywords(ticket_text)
        uppercase_count = len(re.findall(r'\b[A-Z]{2,}\b', ticket_text))
        
        # Vectorize text
        X_tfidf = self.tfidf_vectorizer.transform([processed_text])
        
        # Combine features
        additional_features = np.array([[text_length, word_count, sentiment_polarity,
                                       sentiment_subjectivity, complaint_keyword_count,
                                       uppercase_count]])
        
        X_combined = np.hstack([X_tfidf.toarray(), additional_features])
        
        # Make predictions
        issue_pred = self.issue_type_model.predict(X_combined)[0]
        urgency_pred = self.urgency_model.predict(X_combined)[0]
        
        # Get prediction probabilities
        issue_proba = self.issue_type_model.predict_proba(X_combined)[0]
        urgency_proba = self.urgency_model.predict_proba(X_combined)[0]
        
        # Decode predictions
        predicted_issue = self.label_encoders['issue_type'].inverse_transform([issue_pred])[0]
        predicted_urgency = self.label_encoders['urgency_level'].inverse_transform([urgency_pred])[0]
        
        # Extract entities
        entities = self.extract_entities(ticket_text)
        
        result = {
            'predicted_issue_type': predicted_issue,
            'predicted_urgency_level': predicted_urgency,
            'issue_confidence': float(max(issue_proba)),
            'urgency_confidence': float(max(urgency_proba)),
            'extracted_entities': entities,
            'text_features': {
                'text_length': text_length,
                'word_count': word_count,
                'sentiment_polarity': sentiment_polarity,
                'sentiment_subjectivity': sentiment_subjectivity,
                'complaint_keyword_count': complaint_keyword_count,
                'uppercase_count': uppercase_count
            }
        }
        
        return result
    
    def save_models(self, filepath: str):
        """
        Save trained models and vectorizers
        """
        model_data = {
            'issue_type_model': self.issue_type_model,
            'urgency_model': self.urgency_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoders': self.label_encoders,
            'product_list': self.product_list,
            'complaint_keywords': self.complaint_keywords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """
        Load trained models and vectorizers
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.issue_type_model = model_data['issue_type_model']
        self.urgency_model = model_data['urgency_model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.label_encoders = model_data['label_encoders']
        self.product_list = model_data['product_list']
        self.complaint_keywords = model_data['complaint_keywords']
        
        print(f"Models loaded from {filepath}")
    
    def visualize_results(self, results: Dict[str, Any]):
        """
        Create visualizations for model performance
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Issue Type Confusion Matrix
        issue_cm = confusion_matrix(
            results['test_data']['y_issue_test'], 
            results['test_data']['issue_pred']
        )
        sns.heatmap(issue_cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Issue Type Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Urgency Level Confusion Matrix
        urgency_cm = confusion_matrix(
            results['test_data']['y_urgency_test'], 
            results['test_data']['urgency_pred']
        )
        sns.heatmap(urgency_cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Reds')
        axes[0, 1].set_title('Urgency Level Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # Feature Importance for Issue Type
        if hasattr(self.issue_type_model, 'feature_importances_'):
            feature_names = (list(self.tfidf_vectorizer.get_feature_names_out()) + 
                           ['text_length', 'word_count', 'sentiment_polarity', 
                            'sentiment_subjectivity', 'complaint_keyword_count', 
                            'uppercase_count'])
            
            # Get top 10 features
            top_indices = np.argsort(self.issue_type_model.feature_importances_)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = self.issue_type_model.feature_importances_[top_indices]
            
            axes[1, 0].barh(range(len(top_features)), top_importances)
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features)
            axes[1, 0].set_title('Top 10 Features - Issue Type')
            axes[1, 0].set_xlabel('Importance')
        
        # Feature Importance for Urgency Level
        if hasattr(self.urgency_model, 'feature_importances_'):
            top_indices = np.argsort(self.urgency_model.feature_importances_)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = self.urgency_model.feature_importances_[top_indices]
            
            axes[1, 1].barh(range(len(top_features)), top_importances)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features)
            axes[1, 1].set_title('Top 10 Features - Urgency Level')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_performance_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to demonstrate the pipeline
    """
    # Initialize classifier
    classifier = CustomerSupportTicketClassifier()
    
    # Load data (replace with actual file path)
    # df = pd.read_excel('ai_dev_assignment_tickets_complex_1000.xlsx')
    
    # For demonstration, create sample data
    sample_data = {
        'ticket_id': [1, 2, 3, 4, 5],
        'ticket_text': [
            "My laptop screen is broken and not displaying anything. Need urgent repair.",
            "The application keeps crashing when I try to save my work. This is frustrating.",
            "I forgot my password and cannot access my account. Please help.",
            "The printer in the office is not working. Paper jam issue.",
            "Website is loading very slowly. Users are complaining about performance."
        ],
        'issue_type': ['Hardware', 'Software', 'Access', 'Hardware', 'Performance'],
        'urgency_level': ['High', 'Medium', 'Low', 'Medium', 'High'],
        'product': ['laptop', 'application', 'account', 'printer', 'website']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Prepare data
    df_prepared = classifier.prepare_data(df)
    
    # Train models
    results = classifier.train_models(df_prepared)
    
    # Test prediction on new ticket
    test_ticket = "My computer is frozen and I cannot access any applications. This is very urgent!"
    prediction = classifier.predict_ticket(test_ticket)
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Ticket: {test_ticket}")
    print(f"Predicted Issue Type: {prediction['predicted_issue_type']}")
    print(f"Predicted Urgency Level: {prediction['predicted_urgency_level']}")
    print(f"Issue Confidence: {prediction['issue_confidence']:.4f}")
    print(f"Urgency Confidence: {prediction['urgency_confidence']:.4f}")
    print(f"Extracted Entities: {json.dumps(prediction['extracted_entities'], indent=2)}")
    
    # Save models
    classifier.save_models('ticket_classification_models.pkl')
    
    # Visualize results
    classifier.visualize_results(results)
    
    return classifier, results


if __name__ == "__main__":
    classifier, results = main()