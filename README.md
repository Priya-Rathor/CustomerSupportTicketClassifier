# 🎫 Customer Support Ticket Classifier

A powerful machine learning-powered web application that automatically classifies customer support tickets by issue type and urgency level, while extracting key entities from ticket descriptions.

## 🌟 Features

- **Automatic Classification**: Classifies tickets into categories like Hardware, Software, Network, Access, etc.
- **Urgency Detection**: Determines ticket priority levels (Low, Medium, High, Critical)
- **Entity Extraction**: Identifies products, dates, and complaint keywords
- **Text Analysis**: Provides sentiment analysis and text feature insights
- **Interactive Web Interface**: User-friendly Gradio-based web app
- **Batch Processing**: Upload CSV/Excel files to process multiple tickets at once
- **Real-time Results**: Instant classification with confidence scores

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Installation

1. **Clone or download the project**
   ```bash
   git clone <your-repo-url>
   cd customer-support-classifier
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install gradio pandas plotly numpy
   ```

### Running the Application

1. **Start the web application**
   ```bash
   python gradio_app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:7860`
   - If it doesn't open automatically, visit the URL shown in your terminal

3. **Start classifying tickets!**
   - Enter a ticket description in the text box
   - Click "Classify Ticket" to see results
   - Try the example tickets to see how it works

## 📱 How to Use

### Single Ticket Classification

1. **Navigate to the "Single Ticket Classification" tab**
2. **Enter a ticket description** like:
   ```
   My laptop screen is broken and not displaying anything. 
   Need urgent repair for tomorrow's presentation.
   ```
3. **Click "Classify Ticket"**
4. **View the results**:
   - Issue Type and Urgency Level with confidence scores
   - Extracted entities (products, dates, keywords)
   - Text analysis features
   - Complete JSON output

### Batch Processing

1. **Navigate to the "Batch Processing" tab**
2. **Prepare your CSV/Excel file** with this format:
   ```csv
   ticket_id,ticket_text
   1,"My laptop is broken and needs repair"
   2,"Password reset required for email account"
   3,"Database server is experiencing issues"
   ```
3. **Upload your file** and click "Process Batch"
4. **View the summary** with distribution statistics

### Example Tickets to Try

Click on any of these examples in the app:

- **Hardware Issue**: "My laptop screen is completely black and won't turn on"
- **Software Problem**: "The application keeps crashing when I save documents"
- **Access Issue**: "I forgot my password and cannot access my email account"
- **Network Problem**: "VPN connection keeps disconnecting every few minutes"
- **Critical Issue**: "Database server crashed and all systems are down. URGENT!"

## 🎯 Classification Categories

### Issue Types
- **Hardware**: Laptops, monitors, printers, mice, keyboards
- **Software**: Applications, crashes, bugs, software issues
- **Network**: Internet, WiFi, VPN, connection problems
- **Access**: Passwords, logins, account access issues
- **Performance**: Slow systems, speed issues
- **Infrastructure**: Servers, databases, system-wide issues
- **Configuration**: Setup, settings, email configuration
- **Security**: Antivirus, security software issues
- **Data**: Backups, data recovery, file issues
- **Licensing**: Software licenses, renewals

### Urgency Levels
- **Critical**: System down, urgent, emergency situations
- **High**: Important issues, high priority, needs quick attention
- **Medium**: Normal priority, standard issues
- **Low**: Minor issues, can wait, low priority

## 🔧 Technical Details

### Architecture
- **Frontend**: Gradio web interface with modern design
- **Backend**: Python with mock machine learning classifier
- **Classification**: Rule-based system (demo version)
- **Entity Extraction**: Regex and keyword-based extraction
- **Text Analysis**: Basic sentiment analysis and text features

### File Structure
```
customer-support-classifier/
├── gradio_app.py          # Main application file
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── sample_data/           # Sample CSV files for testing
    ├── sample_tickets.csv
    └── batch_example.xlsx
```

### Key Components

1. **CustomerSupportTicketClassifier**: Main classifier class
2. **TicketClassificationApp**: Gradio web interface
3. **Entity Extraction**: Products, dates, keywords detection
4. **Text Analysis**: Length, sentiment, feature extraction

## 📊 Sample Data Format

### For Single Tickets
Just enter any text describing a support issue:
```
"The office printer is jammed and won't print. Paper keeps getting stuck."
```

### For Batch Processing
CSV/Excel file with `ticket_text` column:
```csv
ticket_id,ticket_text
1,"My laptop screen is broken and needs repair"
2,"Password reset required for email account"
3,"Database server is experiencing issues"
```

## 🛠️ Customization

### Adding New Issue Types
Edit the `issue_types` list in `CustomerSupportTicketClassifier`:
```python
self.issue_types = ['Hardware', 'Software', 'Network', 'YourNewType']
```

### Adding New Products
Add to the `products` list for better entity extraction:
```python
self.products = ['laptop', 'printer', 'your_product']
```

### Modifying Classification Rules
Update the `_predict_issue_type()` method to add new rules:
```python
if any(keyword in text_lower for keyword in ['new_keyword']):
    return 'NewIssueType', confidence_score
```

## 🔍 Troubleshooting

### Common Issues

1. **"NameError: name 'CustomerSupportTicketClassifier' is not defined"**
   - Make sure you're using the complete fixed version of the code
   - Check that all imports are present at the top of the file

2. **"Module not found" errors**
   - Install missing packages: `pip install gradio pandas plotly numpy`
   - Make sure you're in the correct virtual environment

3. **File upload not working**
   - Ensure your CSV/Excel file has a 'ticket_text' column
   - Check file format (CSV or XLSX only)

4. **App not opening in browser**
   - Manually visit `http://localhost:7860`
   - Check if the port is already in use

### Performance Tips

- **For better performance**: Use shorter ticket descriptions
- **For batch processing**: Keep files under 1000 tickets for demo version
- **For real deployment**: Replace mock classifier with trained ML models

## 🎨 Features Overview

### Web Interface Tabs

1. **Single Ticket Classification**: Process one ticket at a time
2. **Batch Processing**: Upload and process multiple tickets
3. **Model Information**: Learn about the classifier architecture
4. **API Usage**: Documentation for programmatic use

### Output Information

- **Classification Results**: Issue type and urgency with confidence
- **Extracted Entities**: Products, dates, complaint keywords found
- **Text Analysis**: Length, word count, sentiment analysis
- **JSON Output**: Complete structured results for integration

## 🚀 Future Enhancements

- Replace mock classifier with real machine learning models
- Add more sophisticated NLP features
- Implement user feedback and model retraining
- Add more entity types and extraction rules
- Create REST API endpoints
- Add visualization dashboards
- Implement user authentication

## 🤝 Contributing

Feel free to fork this project and submit improvements!

---

**Built with ❤️ using Python, Gradio, and Machine Learning**


