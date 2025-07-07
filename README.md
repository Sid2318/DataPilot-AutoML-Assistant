# 🚀 AutoML Assistant

A comprehensive automated machine learning (AutoML) web application built with Streamlit that streamlines the entire machine learning pipeline from data cleaning to model deployment. This tool is designed to make machine learning accessible to both beginners and experts by automating complex processes while providing detailed insights and explanations.

## 🌟 Features

### 📊 Complete ML Pipeline
- **Data Upload & Validation**: Support for CSV files with automatic validation
- **Intelligent Data Cleaning**: Automated detection and removal of constant columns, ID-like columns, and data type optimization
- **Exploratory Data Analysis**: Interactive data profiling with comprehensive statistical analysis
- **Smart Preprocessing**: Automatic handling of missing values, encoding, scaling, and class imbalance
- **Hyperparameter Tuning**: Advanced optimization using Optuna for multiple algorithms
- **Ensemble Learning**: Automated ensemble model creation with voting and stacking strategies
- **AI-Powered Insights**: Intelligent explanations of results using Google's Gemini AI

### 🤖 Supported Machine Learning Algorithms
- **Classification**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, K-Nearest Neighbors
- **Regression**: Linear Regression, Random Forest, XGBoost, LightGBM, SVR, KNN Regressor
- **Ensemble Methods**: Voting Classifier/Regressor, Stacking Classifier/Regressor

### 🔧 Advanced Features
- **Automatic Problem Type Detection**: Intelligently determines whether your problem is classification or regression
- **Class Imbalance Handling**: SMOTE, Random Under-sampling, SMOTEENN techniques
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Model Performance Visualization**: Interactive plots and detailed performance metrics
- **Comprehensive Reporting**: Detailed AutoML reports with AI-powered explanations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd automl-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Key (Optional)**
   For AI-powered explanations, set your Google API key:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```
   Or edit the API key directly in `explain_report.py`

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
automl-assistant/
├── app.py                 # Main Streamlit application
├── cleaner.py            # Data cleaning utilities
├── eda_helper.py         # Exploratory data analysis functions
├── preprocessing.py      # Data preprocessing pipeline
├── tuner.py             # Model hyperparameter tuning
├── explain_report.py    # AI-powered report explanations
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🛠️ Component Overview

### 📝 `app.py`
The main Streamlit application that orchestrates the entire ML pipeline:
- User interface for file upload and parameter selection
- Step-by-step workflow management
- Results visualization and ensemble building
- Report generation and AI explanations

### 🧹 `cleaner.py`
Data cleaning utilities that automatically:
- Remove constant columns (no variance)
- Drop ID-like columns (monotonic sequences)
- Optimize data types
- Handle missing values intelligently

### 📊 `eda_helper.py`
Exploratory Data Analysis functions:
- Generate comprehensive data profiles
- Statistical summaries and visualizations
- Missing value analysis
- Data distribution insights

### ⚙️ `preprocessing.py`
Comprehensive preprocessing pipeline:
- Task type detection (classification/regression)
- Class imbalance detection and handling
- Feature encoding and scaling
- Train-test split preparation

### 🎯 `tuner.py`
Advanced hyperparameter optimization:
- Multi-algorithm support
- Bayesian optimization with Optuna
- Cross-validation scoring
- Ensemble model building

### 🧠 `explain_report.py`
AI-powered insights using Google's Gemini:
- Intelligent report analysis
- Model performance explanations
- Recommendations for improvement

## 💡 Usage Guide

### Step 1: Upload Your Dataset
- Click "Upload your dataset" and select a CSV file
- The application will automatically validate and preview your data

### Step 2: Clean Your Data
- Click "🚀 Clean My Data" to automatically:
  - Remove problematic columns
  - Optimize data types
  - Handle basic data quality issues

### Step 3: Explore Your Data
- Click "🚀 Exploratory Data Analysis" to generate:
  - Interactive data profiling report
  - Statistical summaries
  - Missing value analysis
  - Data distribution visualizations

### Step 4: Preprocess Your Data
- Select your target column
- Choose sampling strategy (if needed for class imbalance)
- Click "🚀 Data Preprocessing" to prepare data for modeling

### Step 5: Model Tuning
- Set number of trials for hyperparameter optimization
- Click "Run AutoML Tuning" to:
  - Automatically detect problem type
  - Tune multiple algorithms
  - Generate performance comparisons

### Step 6: Build Ensemble
- Review sorted model results
- Select ensemble type (Voting or Stacking)
- Choose number of top models to combine
- Click "Build Ensemble and Evaluate"

### Step 7: Get AI Insights
- View comprehensive AutoML report
- Read AI-powered explanations and recommendations
- Download detailed results

## 🔧 Configuration Options

### Hyperparameter Tuning
- **Number of trials**: 10-200 (default: 30)
- **Timeout**: 600 seconds per model
- **Cross-validation**: 5-fold stratified (classification) or regular (regression)

### Sampling Strategies
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Random Under-sampling**: Reduce majority class samples
- **SMOTEENN**: Combination of SMOTE and Edited Nearest Neighbors

### Ensemble Methods
- **Voting**: Simple or weighted voting of predictions
- **Stacking**: Meta-learning approach with final estimator

## 📊 Performance Metrics

### Classification
- **Accuracy**: Overall correctness of predictions
- **Cross-validation scores**: Robust performance estimation
- **Class distribution analysis**: Imbalance detection

### Regression
- **Mean Squared Error (MSE)**: Prediction error measurement
- **Cross-validation scores**: Model stability assessment
- **Feature importance**: Understanding key predictors

## 🚨 Troubleshooting

### Common Issues

1. **Large datasets taking too long**
   - Reduce number of trials
   - Use sampling strategies
   - Consider feature selection

2. **Memory errors**
   - Reduce dataset size
   - Use simpler models
   - Close other applications

3. **API key issues**
   - Verify Google API key is set correctly
   - Check API quotas and limits
   - Ensure proper permissions

### Performance Tips
- Use appropriate number of trials (30-50 for most cases)
- Consider data size when selecting ensemble complexity
- Monitor system resources during tuning

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Optuna**: For efficient hyperparameter optimization
- **Scikit-learn**: For comprehensive ML algorithms
- **Google Gemini**: For AI-powered explanations
- **XGBoost & LightGBM**: For gradient boosting implementations

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation for common solutions

---

**Made with ❤️ for the ML community**

Transform your data into insights with just a few clicks! 🚀