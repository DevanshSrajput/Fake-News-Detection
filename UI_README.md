# 📰 Fake News Detection System - Modern UI

A beautiful, modern web interface for the Fake News Detection System built with Streamlit.

## 🚀 Quick Start

### Option 1: Use the Batch File (Windows)

Simply double-click `run_app.bat` to start the application.

### Option 2: Command Line

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🎯 Features

### 🏠 Home Dashboard

- System overview and status
- Quick navigation to main features
- Real-time system status indicators

### 📊 Model Training

- Load existing datasets or upload new ones
- Configure training parameters
- Real-time training progress
- Model performance metrics and visualizations
- Save/load trained models

### 🔍 Fake News Detection

- Multiple input methods:
  - Type/paste text directly
  - Upload text files
  - Sample articles for testing
- Real-time analysis with multiple ML models
- Confidence scores and consensus predictions
- Article statistics and insights

### 📈 Analytics Dashboard

- Comprehensive model performance analysis
- Interactive charts and visualizations
- Confusion matrices
- Dataset insights and statistics

### ℹ️ About & Documentation

- System information and technical details
- Usage tips and best practices

## 🎨 UI Features

- **Modern Design**: Clean, professional interface with gradient backgrounds
- **Responsive Layout**: Works on desktop and mobile devices
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Updates**: Live progress tracking and status updates
- **Intuitive Navigation**: Easy-to-use sidebar navigation
- **Visual Feedback**: Color-coded predictions and status indicators

## 📊 Visualizations

- **Performance Comparison Charts**: Compare model accuracy and F1 scores
- **Confusion Matrices**: Interactive heatmaps showing model performance
- **Confidence Bars**: Visual representation of prediction confidence
- **Dataset Distribution**: Pie charts showing data balance
- **Real-time Progress**: Progress bars for training and analysis

## 🔧 Technical Details

### Architecture

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Your existing FakeNewsDetector class
- **Visualizations**: Plotly for interactive charts
- **State Management**: Streamlit session state for data persistence

### Key Components

- **Model Training Pipeline**: Automated training with progress tracking
- **Prediction Engine**: Real-time article analysis
- **Visualization Engine**: Interactive charts and graphs
- **Data Management**: CSV loading, preprocessing, and caching

## 📝 Usage Tips

1. **First Time Setup**: Start by training models on the "Train Models" page
2. **Loading Models**: Use saved models for faster startup
3. **Testing**: Try the sample articles to understand system behavior
4. **Performance**: Monitor model metrics in the Analytics section
5. **Data Quality**: Ensure your dataset has proper 'text' and 'label' columns

## 🎯 Supported Input Formats

### Dataset Requirements

- CSV format with 'text' and 'label' columns
- Labels should be: REAL/FAKE, TRUE/FALSE, or 0/1
- UTF-8 encoding recommended

### Article Input

- Plain text (any length)
- Text files (.txt format)
- Pasted content from web/documents

## 🔍 Model Information

The system uses two machine learning models:

1. **Passive Aggressive Classifier**: Fast, online learning algorithm
2. **Support Vector Machine (SVM)**: Robust classification with linear kernel

Both models use TF-IDF vectorization for text feature extraction.

## 🎨 Customization

The UI includes custom CSS for:

- Modern gradient backgrounds
- Color-coded prediction boxes
- Professional metric cards
- Responsive design elements

## 📈 Performance Metrics

The system displays:

- **Accuracy**: Overall correct predictions
- **F1 Score**: Balanced precision and recall
- **Confusion Matrix**: Detailed classification results
- **Confidence Scores**: Prediction certainty levels

## 🛠️ Troubleshooting

### Common Issues

1. **Model not trained**: Train or load a model first
2. **Dataset format**: Ensure proper CSV format with required columns
3. **Empty predictions**: Check if text preprocessing is working
4. **Memory issues**: Reduce sample size for large datasets

### Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## 📱 Mobile Support

The interface is responsive and works on:

- Desktop computers
- Tablets
- Mobile phones (portrait and landscape)

---

**Built with ❤️ using Python, Streamlit, and Plotly**
