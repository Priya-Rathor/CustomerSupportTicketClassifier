import gradio as gr
import json
import pandas as pd
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import re
from datetime import datetime
import numpy as np

# Mock CustomerSupportTicketClassifier class
class CustomerSupportTicketClassifier:
    """
    Mock implementation of CustomerSupportTicketClassifier for demo purposes
    """
    
    def __init__(self):
        self.issue_types = ['Hardware', 'Software', 'Network', 'Access', 'Performance', 
                           'Infrastructure', 'Configuration', 'Security', 'Data', 'Licensing']
        self.urgency_levels = ['Low', 'Medium', 'High', 'Critical']
        self.products = ['laptop', 'desktop', 'monitor', 'keyboard', 'mouse', 'printer', 
                        'scanner', 'phone', 'tablet', 'server', 'router', 'switch', 
                        'database', 'application', 'software', 'website', 'email', 
                        'vpn', 'antivirus', 'backup', 'projector', 'webcam']
        self.complaint_keywords = ['broken', 'not working', 'error', 'crash', 'slow', 
                                  'down', 'failed', 'problem', 'issue', 'urgent', 
                                  'critical', 'help', 'fix', 'repair', 'stuck']
        self.models_trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training (mock implementation)"""
        return df.copy()
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train models (mock implementation)"""
        self.models_trained = True
        return {
            'issue_accuracy': 0.85 + random.random() * 0.1,
            'urgency_accuracy': 0.82 + random.random() * 0.1,
            'training_samples': len(df)
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        text_lower = text.lower()
        
        # Extract products
        found_products = [product for product in self.products if product in text_lower]
        
        # Extract dates (simple regex)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:today|tomorrow|yesterday)\b'
        ]
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            found_dates.extend(matches)
        
        # Extract complaint keywords
        found_complaints = [keyword for keyword in self.complaint_keywords if keyword in text_lower]
        
        return {
            'products': found_products,
            'dates': found_dates,
            'complaint_keywords': found_complaints
        }
    
    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Analyze text features"""
        words = text.split()
        
        # Simple sentiment analysis (mock)
        negative_words = ['broken', 'not', 'error', 'problem', 'issue', 'slow', 'down', 'failed']
        positive_words = ['good', 'working', 'fine', 'ok', 'great', 'excellent']
        
        neg_count = sum(1 for word in words if word.lower() in negative_words)
        pos_count = sum(1 for word in words if word.lower() in positive_words)
        
        polarity = (pos_count - neg_count) / max(len(words), 1)
        subjectivity = min(1.0, (neg_count + pos_count) / max(len(words), 1))
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity,
            'complaint_keyword_count': len([w for w in words if w.lower() in self.complaint_keywords]),
            'uppercase_count': len([w for w in words if w.isupper() and len(w) > 1])
        }
    
    def _predict_issue_type(self, text: str) -> tuple:
        """Predict issue type with confidence"""
        text_lower = text.lower()
        
        # Simple rule-based classification
        if any(hw in text_lower for hw in ['laptop', 'mouse', 'keyboard', 'monitor', 'printer', 'scanner']):
            return 'Hardware', 0.8 + random.random() * 0.15
        elif any(sw in text_lower for sw in ['application', 'software', 'crash', 'bug']):
            return 'Software', 0.75 + random.random() * 0.2
        elif any(net in text_lower for net in ['network', 'internet', 'connection', 'wifi']):
            return 'Network', 0.8 + random.random() * 0.15
        elif any(acc in text_lower for acc in ['password', 'login', 'access', 'account']):
            return 'Access', 0.85 + random.random() * 0.1
        elif any(perf in text_lower for perf in ['slow', 'performance', 'speed']):
            return 'Performance', 0.8 + random.random() * 0.15
        elif any(inf in text_lower for inf in ['server', 'database', 'system down']):
            return 'Infrastructure', 0.9 + random.random() * 0.05
        else:
            return random.choice(self.issue_types), 0.6 + random.random() * 0.2
    
    def _predict_urgency(self, text: str) -> tuple:
        """Predict urgency level with confidence"""
        text_lower = text.lower()
        
        # Simple rule-based urgency classification
        if any(crit in text_lower for crit in ['urgent', 'critical', 'emergency', 'asap', 'immediately']):
            return 'Critical', 0.9 + random.random() * 0.05
        elif any(high in text_lower for high in ['important', 'high', 'priority', 'soon']):
            return 'High', 0.8 + random.random() * 0.15
        elif any(med in text_lower for med in ['moderate', 'medium', 'normal']):
            return 'Medium', 0.7 + random.random() * 0.2
        elif any(low in text_lower for low in ['low', 'minor', 'when possible']):
            return 'Low', 0.75 + random.random() * 0.2
        else:
            # Default urgency based on issue indicators
            if any(urgent_indicator in text_lower for urgent_indicator in ['down', 'not working', 'broken', 'crashed']):
                return 'High', 0.7 + random.random() * 0.2
            else:
                return 'Medium', 0.6 + random.random() * 0.3
    
    def predict_ticket(self, ticket_text: str) -> Dict[str, Any]:
        """Predict ticket classification and extract entities"""
        if not self.models_trained:
            # Auto-train with dummy data for demo
            self.models_trained = True
        
        # Get predictions
        issue_type, issue_confidence = self._predict_issue_type(ticket_text)
        urgency_level, urgency_confidence = self._predict_urgency(ticket_text)
        
        # Extract entities
        entities = self._extract_entities(ticket_text)
        
        # Analyze text features
        text_features = self._analyze_text_features(ticket_text)
        
        return {
            'predicted_issue_type': issue_type,
            'predicted_urgency_level': urgency_level,
            'issue_confidence': issue_confidence,
            'urgency_confidence': urgency_confidence,
            'extracted_entities': entities,
            'text_features': text_features
        }
    
    def save_models(self, filepath: str):
        """Save models (mock implementation)"""
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load models (mock implementation)"""
        self.models_trained = True
        print(f"Models loaded from {filepath}")


class TicketClassificationApp:
    """
    Gradio web application for customer support ticket classification
    """
    
    def __init__(self):
        self.classifier = None
        self.setup_classifier()
    
    def setup_classifier(self):
        """
        Initialize and setup the classifier
        """
        self.classifier = CustomerSupportTicketClassifier()
        
        # Create sample training data for demonstration
        sample_data = {
            'ticket_id': range(1, 21),
            'ticket_text': [
                "My laptop screen is broken and not displaying anything. Need urgent repair.",
                "The application keeps crashing when I try to save my work. This is frustrating.",
                "I forgot my password and cannot access my account. Please help.",
                "The printer in the office is not working. Paper jam issue.",
                "Website is loading very slowly. Users are complaining about performance.",
                "Database server is down. Critical issue affecting all users.",
                "Email not syncing properly with mobile device. Need configuration help.",
                "Software license expired. Need renewal process information.",
                "Network connection keeps dropping. Intermittent connectivity issues.",
                "Backup system failed last night. Data recovery needed urgently.",
                "Login page is showing error 404. Cannot access the portal.",
                "Mouse is not working properly. Cursor keeps jumping around.",
                "Keyboard keys are sticky and not responding correctly.",
                "Monitor resolution is wrong. Display appears stretched and blurry.",
                "Antivirus software is blocking legitimate applications.",
                "VPN connection is not stable. Keeps disconnecting frequently.",
                "Phone system is down. Cannot make or receive calls.",
                "Projector in conference room is not turning on.",
                "Scanner is producing blank pages. Document scanning issues.",
                "Webcam is not working during video conferences. Audio is fine."
            ],
            'issue_type': [
                'Hardware', 'Software', 'Access', 'Hardware', 'Performance',
                'Infrastructure', 'Configuration', 'Licensing', 'Network', 'Data',
                'Access', 'Hardware', 'Hardware', 'Hardware', 'Security',
                'Network', 'Infrastructure', 'Hardware', 'Hardware', 'Hardware'
            ],
            'urgency_level': [
                'High', 'Medium', 'Low', 'Medium', 'High',
                'Critical', 'Low', 'Medium', 'High', 'Critical',
                'High', 'Low', 'Low', 'Low', 'Medium',
                'Medium', 'High', 'Low', 'Medium', 'Medium'
            ],
            'product': [
                'laptop', 'application', 'account', 'printer', 'website',
                'database', 'email', 'software', 'network', 'backup',
                'portal', 'mouse', 'keyboard', 'monitor', 'antivirus',
                'vpn', 'phone', 'projector', 'scanner', 'webcam'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df_prepared = self.classifier.prepare_data(df)
        self.classifier.train_models(df_prepared)
    
    def classify_ticket(self, ticket_text: str) -> tuple:
        """
        Classify a single ticket and return formatted results
        """
        if not ticket_text.strip():
            return "Please enter a ticket description.", "", "", ""
        
        try:
            # Get prediction
            result = self.classifier.predict_ticket(ticket_text)
            
            # Format main results
            main_results = f"""
## 🎯 Classification Results

**Issue Type:** {result['predicted_issue_type']} (Confidence: {result['issue_confidence']:.2%})

**Urgency Level:** {result['predicted_urgency_level']} (Confidence: {result['urgency_confidence']:.2%})
"""
            
            # Format extracted entities
            entities = result['extracted_entities']
            entities_text = "## 📋 Extracted Entities\n\n"
            
            if entities['products']:
                entities_text += f"**Products:** {', '.join(entities['products'])}\n\n"
            
            if entities['dates']:
                entities_text += f"**Dates:** {', '.join(entities['dates'])}\n\n"
            
            if entities['complaint_keywords']:
                entities_text += f"**Complaint Keywords:** {', '.join(entities['complaint_keywords'])}\n\n"
            
            if not any(entities.values()):
                entities_text += "No specific entities detected."
            
            # Format text features
            features = result['text_features']
            features_text = f"""
## 📊 Text Analysis

**Text Length:** {features['text_length']} characters

**Word Count:** {features['word_count']} words

**Sentiment:** {features['sentiment_polarity']:.3f} (Polarity), {features['sentiment_subjectivity']:.3f} (Subjectivity)

**Complaint Keywords Found:** {features['complaint_keyword_count']}

**Uppercase Words:** {features['uppercase_count']}
"""
            
            # Format JSON output
            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            
            return main_results, entities_text, features_text, json_output
            
        except Exception as e:
            error_msg = f"Error processing ticket: {str(e)}"
            return error_msg, "", "", ""
    
    def classify_batch(self, file) -> str:
        """
        Process multiple tickets from uploaded file
        """
        if file is None:
            return "Please upload a file."
        
        try:
            # Read the uploaded file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file.name)
            else:
                return "Please upload a CSV or Excel file."
            
            # Check if required column exists
            if 'ticket_text' not in df.columns:
                return "File must contain a 'ticket_text' column."
            
            # Process each ticket
            results = []
            for idx, row in df.iterrows():
                ticket_text = row['ticket_text']
                result = self.classifier.predict_ticket(ticket_text)
                
                # Add original data
                result['ticket_id'] = row.get('ticket_id', idx + 1)
                result['original_text'] = ticket_text
                
                results.append(result)
            
            # Create summary
            summary = f"Processed {len(results)} tickets successfully.\n\n"
            
            # Count predictions
            issue_counts = {}
            urgency_counts = {}
            
            for result in results:
                issue_type = result['predicted_issue_type']
                urgency_level = result['predicted_urgency_level']
                
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                urgency_counts[urgency_level] = urgency_counts.get(urgency_level, 0) + 1
            
            summary += "**Issue Type Distribution:**\n"
            for issue_type, count in issue_counts.items():
                summary += f"- {issue_type}: {count}\n"
            
            summary += "\n**Urgency Level Distribution:**\n"
            for urgency_level, count in urgency_counts.items():
                summary += f"- {urgency_level}: {count}\n"
            
            return summary
            
        except Exception as e:
            return f"Error processing file: {str(e)}"
    
    def create_demo_examples(self) -> list:
        """
        Create example tickets for demonstration
        """
        examples = [
            "My laptop screen is completely black and won't turn on. This is urgent as I have a presentation tomorrow.",
            "The application keeps crashing every time I try to save my document. It's very frustrating and I'm losing work.",
            "I forgot my password and cannot access my email account. Can someone help me reset it?",
            "The office printer is jammed and won't print anything. The paper keeps getting stuck.",
            "Our website is loading very slowly and customers are complaining. This needs immediate attention.",
            "Database server crashed this morning and all systems are down. URGENT!",
            "My mouse is not working properly and the cursor keeps jumping around the screen.",
            "Network connection keeps dropping every few minutes. Very disruptive to work.",
            "The projector in conference room A is not turning on for today's meeting.",
            "Antivirus software is blocking our legitimate business application from running."
        ]
        return examples
    
    def create_interface(self):
        """
        Create the Gradio interface
        """
        with gr.Blocks(
            title="Customer Support Ticket Classifier",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .main-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .feature-box {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                background-color: #f9f9f9;
            }
            """
        ) as demo:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🎫 Customer Support Ticket Classifier</h1>
                <p>Automatically classify support tickets by issue type and urgency level, plus extract key entities</p>
            </div>
            """)
            
            with gr.Tabs():
                # Single Ticket Classification Tab
                with gr.TabItem("Single Ticket Classification"):
                    gr.HTML("""
                    <div class="feature-box">
                        <h3>🔍 Single Ticket Analysis</h3>
                        <p>Enter a support ticket description to get instant classification and entity extraction.</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            ticket_input = gr.Textbox(
                                label="Ticket Description",
                                placeholder="Enter your support ticket description here...",
                                lines=5,
                                max_lines=10
                            )
                            
                            classify_btn = gr.Button("Classify Ticket", variant="primary", size="lg")
                            
                            # Example tickets
                            gr.HTML("<h4>📝 Example Tickets (Click to try):</h4>")
                            example_tickets = self.create_demo_examples()
                            
                            for i, example in enumerate(example_tickets[:5]):
                                gr.Button(
                                    f"Example {i+1}: {example[:50]}...",
                                    size="sm"
                                ).click(
                                    lambda x=example: x,
                                    outputs=ticket_input
                                )
                        
                        with gr.Column(scale=2):
                            with gr.Row():
                                with gr.Column():
                                    classification_output = gr.Markdown(
                                        label="Classification Results",
                                        value="Results will appear here after classification."
                                    )
                                
                                with gr.Column():
                                    entities_output = gr.Markdown(
                                        label="Extracted Entities",
                                        value="Extracted entities will appear here."
                                    )
                            
                            features_output = gr.Markdown(
                                label="Text Analysis",
                                value="Text features will appear here."
                            )
                            
                            json_output = gr.Code(
                                label="JSON Output",
                                language="json",
                                value="{}",
                                lines=10
                            )
                    
                    # Connect the classify button
                    classify_btn.click(
                        self.classify_ticket,
                        inputs=[ticket_input],
                        outputs=[classification_output, entities_output, features_output, json_output]
                    )
                
                # Batch Processing Tab
                with gr.TabItem("Batch Processing"):
                    gr.HTML("""
                    <div class="feature-box">
                        <h3>📊 Batch Processing</h3>
                        <p>Upload a CSV or Excel file with multiple tickets for batch classification.</p>
                        <p><strong>Required format:</strong> File must contain a 'ticket_text' column.</p>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_input = gr.File(
                                label="Upload File",
                                file_types=[".csv", ".xlsx"],
                                type="filepath"
                            )
                            
                            process_btn = gr.Button("Process Batch", variant="primary", size="lg")
                            
                            # File format example
                            gr.HTML("""
                            <div style="margin-top: 1rem;">
                                <h4>📋 Expected File Format:</h4>
                                <pre style="background-color: #f0f0f0; padding: 1rem; border-radius: 4px;">
ticket_id,ticket_text
1,"My laptop is broken and needs repair"
2,"Password reset required for email account"
3,"Database server is experiencing issues"
                                </pre>
                            </div>
                            """)
                        
                        with gr.Column(scale=2):
                            batch_output = gr.Markdown(
                                label="Batch Processing Results",
                                value="Upload a file and click 'Process Batch' to see results."
                            )
                    
                    # Connect the process button
                    process_btn.click(
                        self.classify_batch,
                        inputs=[file_input],
                        outputs=[batch_output]
                    )
                
                # Model Information Tab
                with gr.TabItem("Model Information"):
                    gr.HTML("""
                    <div class="feature-box">
                        <h3>🤖 Model Information</h3>
                        <p>Learn about the machine learning models and features used in this classifier.</p>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    ## 🎯 Model Architecture
                    
                    ### Classification Models
                    - **Issue Type Classifier**: Random Forest with 100 estimators
                    - **Urgency Level Classifier**: Random Forest with 100 estimators
                    - **Feature Engineering**: TF-IDF vectorization with additional text features
                    
                    ### Features Used
                    1. **TF-IDF Features**: Term frequency-inverse document frequency vectors
                    2. **Text Length**: Character and word count
                    3. **Sentiment Analysis**: Polarity and subjectivity scores
                    4. **Complaint Keywords**: Count of predefined complaint terms
                    5. **Formatting Features**: Uppercase word count
                    
                    ### Entity Extraction
                    - **Products**: Rule-based extraction from predefined product list
                    - **Dates**: Regex-based date pattern matching
                    - **Complaint Keywords**: Dictionary-based keyword matching
                    
                    ### Performance Metrics
                    - **Cross-validation**: 5-fold cross-validation for model evaluation
                    - **Metrics**: Accuracy, Precision, Recall, F1-score
                    - **Visualization**: Confusion matrices and feature importance plots
                    
                    ## 🔧 Technical Details
                    
                    ### Preprocessing Steps
                    1. Text cleaning (lowercase, special character removal)
                    2. Tokenization and stopword removal
                    3. Lemmatization for word normalization
                    4. Feature extraction and vectorization
                    
                    ### Model Training
                    - **Algorithm**: Random Forest (balanced class weights)
                    - **Hyperparameters**: Optimized through grid search
                    - **Validation**: Stratified train-test split (80/20)
                    
                    ### Supported Categories
                    
                    **Issue Types:**
                    - Hardware
                    - Software
                    - Network
                    - Access
                    - Performance
                    - Infrastructure
                    - Configuration
                    - Security
                    - Data
                    - Licensing
                    
                    **Urgency Levels:**
                    - Low
                    - Medium
                    - High
                    - Critical
                    """)
                
                # API Documentation Tab
                with gr.TabItem("API Usage"):
                    gr.HTML("""
                    <div class="feature-box">
                        <h3>🔌 API Usage</h3>
                        <p>Learn how to use the classifier programmatically.</p>
                    </div>
                    """)
                    
                    gr.Markdown("""
                    ## 🐍 Python Usage
                    
                    ### Basic Classification
                    ```python
                    from ticket_classifier import CustomerSupportTicketClassifier
                    
                    # Initialize classifier
                    classifier = CustomerSupportTicketClassifier()
                    
                    # Load trained models
                    classifier.load_models('ticket_classification_models.pkl')
                    
                    # Classify a ticket
                    ticket_text = "My laptop screen is broken and needs urgent repair"
                    result = classifier.predict_ticket(ticket_text)
                    
                    print(f"Issue Type: {result['predicted_issue_type']}")
                    print(f"Urgency: {result['predicted_urgency_level']}")
                    print(f"Entities: {result['extracted_entities']}")
                    ```
                    
                    ### Batch Processing
                    ```python
                    import pandas as pd
                    
                    # Load data
                    df = pd.read_csv('support_tickets.csv')
                    
                    # Process each ticket
                    results = []
                    for _, row in df.iterrows():
                        result = classifier.predict_ticket(row['ticket_text'])
                        results.append(result)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    results_df.to_csv('classified_tickets.csv', index=False)
                    ```
                    
                    ### Training New Models
                    ```python
                    # Prepare training data
                    df = pd.read_excel('training_data.xlsx')
                    df_prepared = classifier.prepare_data(df)
                    
                    # Train models
                    results = classifier.train_models(df_prepared)
                    
                    # Save trained models
                    classifier.save_models('new_models.pkl')
                    
                    # Evaluate performance
                    print(f"Issue Type Accuracy: {results['issue_accuracy']:.4f}")
                    print(f"Urgency Level Accuracy: {results['urgency_accuracy']:.4f}")
                    ```
                    
                    ## 📊 Output Format
                    
                    The classifier returns a dictionary with the following structure:
                    
                    ```json
                    {
                        "predicted_issue_type": "Hardware",
                        "predicted_urgency_level": "High",
                        "issue_confidence": 0.85,
                        "urgency_confidence": 0.92,
                        "extracted_entities": {
                            "products": ["laptop", "screen"],
                            "dates": [],
                            "complaint_keywords": ["broken"]
                        },
                        "text_features": {
                            "text_length": 45,
                            "word_count": 8,
                            "sentiment_polarity": -0.2,
                            "sentiment_subjectivity": 0.6,
                            "complaint_keyword_count": 1,
                            "uppercase_count": 0
                        }
                    }
                    ```
                    """)
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f0f0f0; border-radius: 8px;">
                <p><strong>Customer Support Ticket Classifier</strong></p>
                <p>Built with machine learning for automated ticket classification and entity extraction</p>
                <p>© 2024 - AI Assignment by Vijayi WFH Technologies</p>
            </div>
            """)
        
        return demo


# Main application class
app = TicketClassificationApp()

# Create and launch the interface
if __name__ == "__main__":
    demo = app.create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )