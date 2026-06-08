# LGBTQ Tweets Sentiment Analysis - Optimized Pipeline v2.0

## 📊 Overview

Advanced sentiment analysis pipeline for LGBTQ+ tweets achieving **87.68% accuracy** through ensemble voting and advanced feature engineering.

### Key Results
- **Accuracy**: 87.68% (↑ 20.98% from baseline)
- **Precision**: 87.70%
- **Recall**: 87.68%
- **F1-Score**: 87.41%
- **Dataset**: 32,456 tweets

## 🎯 Project Goals

✅ **Phase 1: EDA** - Comprehensive data exploration  
✅ **Phase 2: Preprocessing** - Advanced feature engineering with ensemble sentiment labeling  
✅ **Phase 3: Model Training** - Ensemble voting (Naive Bayes 40% + Random Forest 60%)  
✅ **Phase 4: Evaluation** - Detailed metrics and confusion matrices  

## 🔧 Optimizations Implemented

### 1. Ensemble Voting
- Naive Bayes: 73.44% (conservative, reliable)
- Random Forest: 93.61% (powerful, prone to overfitting)
- **Ensemble (40% NB + 60% RF)**: 87.68% (balanced)

### 2. Advanced TF-IDF Features
```python
TF-IDF Parameters:
- max_features: 5,000
- max_df: 0.85 (filter common words)
- min_df: 2
- ngram_range: (1, 2)
- sublinear_tf: True
- norm: 'l2'
```

### 3. Sentiment Score Features (4)
- TextBlob polarity
- VADER compound score
- Confidence score
- Polarity × VADER interaction term

### 4. Class Weight Balancing
```
Negative:  weight = 0.895
Neutral:   weight = 1.748 (3-class imbalance handling)
Positive:  weight = 0.763
```

### 5. Advanced Preprocessing
- Dual sentiment labeling (TextBlob + VADER)
- Ensemble voting for high confidence labels
- Stratified train-test split (80-20)
- Hyperparameter optimization for both models

## 📈 Performance Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | 66.70% | 87.68% | **+20.98%** |
| Precision | 67.91% | 87.70% | **+19.79%** |
| Recall | 66.70% | 87.68% | **+20.98%** |
| F1-Score | 64.83% | 87.41% | **+22.58%** |

## 📊 Per-Class Performance

### Negative Class (2,418 samples)
```
Precision: 88.86%
Recall:    91.07%
F1-Score:  89.95%
```

### Neutral Class (1,238 samples)
```
Precision: 87.82%
Recall:    68.74%
F1-Score:  77.12%
```

### Positive Class (2,836 samples)
```
Precision: 86.67%
Recall:    93.05%
F1-Score:  89.75%
```

## 📁 Project Structure

```
sentimal-analisis-lgbtq/
├── sentiment_optimized_v2.py      # Main optimized pipeline
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── results/
│   ├── 00_OPTIMIZATION_SUMMARY.txt # Optimization details
│   ├── 01_Model_Comparison.png     # NB vs RF vs Ensemble
│   ├── 02_Optimized_Confusion_Matrix.png
│   ├── 03_Optimized_Per_Class.png  # Per-class metrics
│   ├── 02_Optimized_Metrics.csv    # Model metrics
│   ├── 03_Optimized_Per_Class.csv  # Per-class metrics
│   └── 04_Confusion_Matrix.csv     # Confusion matrix data
└── data/
    └── LGBT_Tweets_processed.csv   # Dataset (32,456 tweets)
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Pipeline
```bash
python sentiment_optimized_v2.py
```

### Output
- Visualizations: PNG files in `hasil/`
- Metrics: CSV files in `hasil/`
- Summary: `00_OPTIMIZATION_SUMMARY.txt`

## 📊 Sentiment Distribution

```
Positive:  14,180 tweets (43.7%)
Negative:  12,087 tweets (37.2%)
Neutral:    6,189 tweets (19.1%)
```

## 🎓 Technical Stack

- **Language**: Python 3.x
- **ML Framework**: scikit-learn
- **Feature Engineering**: TF-IDF vectorization
- **Sentiment Analysis**: TextBlob + VADER
- **Visualization**: Matplotlib + Seaborn
- **Data Processing**: Pandas + NumPy

## ✨ Key Improvements Over Baseline

1. **Ensemble Voting** - Combines strengths of NB (reliable) and RF (powerful)
2. **Advanced Features** - TF-IDF (5,000) + Sentiment features (4) = 5,004 total
3. **Class Balancing** - Weighted approach for imbalanced data
4. **Hyperparameter Tuning** - Optimized parameters for both models
5. **Better Neutral Handling** - Improved from 27.5% to 68.7% recall

## 📊 Confusion Matrix (Optimized)

```
           Predicted
        NEG  NEU  POS
Actual
NEG     2202  59  157
NEU      138  851  249
POS      138   59  2639
```

## 💡 Usage Examples

### Making Predictions
```python
from sentiment_optimized_v2 import model, vectorizer

# New tweet
tweet = "I love the LGBTQ community!"

# Vectorize and predict
X = vectorizer.transform([tweet])
prediction = ensemble_model.predict(X)
label = ['Negative', 'Neutral', 'Positive'][prediction[0]]

print(f"Sentiment: {label}")
```

## 🎯 Next Steps

1. **Fine-tune thresholds** for each class
2. **Add more features** (emoji, hashtags, user features)
3. **Implement real-time inference** API
4. **Deploy to production** with monitoring
5. **Collect more training data** for minority classes

## ⚠️ Limitations

1. **Neutral Class**: Still challenging, 68.7% recall (vs 91% for Negative)
2. **Text-only**: No emoji, hashtags, or metadata
3. **English tweets**: Not tested on other languages
4. **Fixed vocabulary**: Limited to 5,000 most common words

## 📚 References

- **TextBlob**: Simple NLP tasks and sentiment analysis
- **VADER**: Lexicon and rule-based sentiment tool
- **Scikit-learn**: ML pipeline and metrics
- **Ensemble Methods**: Better generalization through voting

## 📄 License

MIT License - Feel free to use for research and commercial projects

## 📧 Contact

For questions or improvements, please open an issue or submit a pull request.

---

**Last Updated**: 2024  
**Status**: ✅ Production Ready  
**Accuracy**: 87.68%  
**Dataset Size**: 32,456 tweets
