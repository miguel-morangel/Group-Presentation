# Cat Marketing Analytics Platform

A data-driven marketing analytics platform for optimizing cat-related content and campaigns. This platform provides predictive analytics, data visualization, and business recommendations for cat-related marketing campaigns.

Access the Streamlit Application [here](https://catmarketing.streamlit.app/).

## Features

- 📊 **Predictive Analytics**: Engagement rate prediction and campaign performance forecasting
- 📈 **Data Visualization**: Interactive dashboards and real-time analytics
- 🎯 **Campaign Optimization**: Data-driven recommendations and insights
- ⏰ **Temporal Analysis**: Optimal posting time analysis
- 🔍 **Content Analysis**: NLP-based caption analysis and performance metrics
- 📱 **Multi-Platform Support**: Analytics for Instagram, Facebook, and TikTok

## Prerequisites

- Python 3.9 or higher (required)
- pip (Python package installer)
- Virtual environment tool (venv)

### System-Specific Requirements

#### Windows

- Visual Studio Build Tools with "Desktop development with C++"
- Git Bash (recommended) or Command Prompt

#### macOS

- Xcode Command Line Tools
- Homebrew (recommended)

#### Linux

- Python development headers
- Build essentials package

## Installation

0. Make sure you have ssh keys properly set up with Github.

Look up instructions to set up ssh keys. 

1. Clone the repository:

```bash
git clone git@github.com:miguel-morangel/Group-Presentation.git
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

4. Download NLTK data:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Project Structure

``` bash
catmarketing/
├── app.py                    # Main Streamlit application
├── data_analysis.py         # Data processing and analysis utilities
├── model_trainer.py         # Core model training framework
├── train_model.py          # Model training script
├── model_creation.py       # Model architecture and configuration
├── model_prediction.py     # Prediction pipeline
├── engineer_features.py    # Feature engineering pipeline
├── data_utils.py           # Data processing utilities
├── clustering.py           # User segmentation analysis
├── time_series_analysis.py # Temporal pattern analysis
├── nlp_processing.py       # Natural language processing utilities
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

### Key Components

- **app.py**: Main Streamlit application providing the user interface and integrating all components
- **model_trainer.py**: Core class `CatMarketingPredictor` managing the ML pipeline
- **train_model.py**: Script for training and validating the predictive model
- **engineer_features.py**: Feature engineering pipeline with the `FeatureEngineer` class
- **data_analysis.py**: Data cleaning, processing, and analysis utilities
- **model_prediction.py**: Prediction pipeline for real-time inference
- **time_series_analysis.py**: Temporal pattern analysis and optimal timing predictions
- **nlp_processing.py**: Text analysis and content optimization utilities

## Usage

1. Prepare your data:
   - Place your data files in `data/raw/`:
     - catfluencer_campaigns.csv
     - catfluencer_engagement.csv
     - catfluencer_posts.csv

2. Train the model:

```bash
python train_model.py
```

3. Launch the application:

```bash
streamlit run app.py
```

4. Access the dashboard at `http://localhost:8501`

## Data Requirements

The platform expects three main data files:

1. **catfluencer_campaigns.csv**:
   - Campaign metadata
   - Budget information
   - Platform details

2. **catfluencer_engagement.csv**:
   - Engagement metrics
   - Temporal data
   - User interactions

3. **catfluencer_posts.csv**:
   - Content information
   - Performance metrics
   - Posting details


## Troubleshooting

### Common Issues

1. **Model Not Found Error**:
   - Ensure you've run `train_model.py` before launching the app
   - Check if the `models` directory exists and contains model files

2. **Import Errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure you're in the virtual environment

3. **Data Loading Errors**:
   - Verify data files exist in `data/raw/`
   - Check file permissions
   - Validate CSV file formats
