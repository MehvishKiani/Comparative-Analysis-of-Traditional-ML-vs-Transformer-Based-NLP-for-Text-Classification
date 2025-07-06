# Comparative-Analysis-of-Traditional-ML-vs-Transformer-Based-NLP-for-Text-Classification
To compare the performance, efficiency, and practicality of traditional machine learning (TF-IDF + Logistic Regression) and modern transformer-based models (BERT) in binary text classification using IMDb movie reviews.
 📂 Dataset Used
- **Name:** IMDb Movie Reviews
- **Source:** Hugging Face Datasets (`datasets.load_dataset("imdb")`)
- **Classes:** Binary (Positive / Negative)
- **Samples:** 50,000 (25,000 train / 25,000 test)

---

## 🛠️ Libraries and Tools
| Library           | Purpose                            |
|------------------|------------------------------------|
| scikit-learn      | ML models, preprocessing           |
| pandas / numpy    | Data handling                      |
| transformers      | BERT model + tokenizer             |
| datasets          | Load IMDb dataset                  |
| torch             | Deep learning backend              |
| matplotlib/seaborn| Visualizations                     |
| Google Colab      | GPU-based training                 |

---

## 🔄 Preprocessing
### Traditional ML
- Lowercasing
- Optional: stopword/punctuation removal
- `TfidfVectorizer(max_features=5000)`

### Transformer-Based
- Tokenization using `BertTokenizer`
- Truncation + Padding
- Conversion to PyTorch dataset for Hugging Face `Trainer`

---

## 📊 Models and Algorithms
| Model Type   | Configuration                        |
|--------------|--------------------------------------|
| Traditional  | TF-IDF + Logistic Regression         |
| Modern       | BERT (bert-base-uncased)             |
| Optimizer    | AdamW (BERT), SGD (LR)               |
| Framework    | `Trainer` API (transformers)         |

---

## ⚖️ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📊 Results and Improvements
| Metric         | TF-IDF + LR       | BERT Transformer    |
|----------------|-------------------|----------------------|
| Accuracy       | **90%**           | 82%                  |
| F1-Score       | 90%               | 82%                  |
| Training Time  | < 1 min (CPU)     | ~10 min (GPU)        |
| Model Size     | Lightweight       | Heavy (110M params)  |
| Interpretability| High             | Low                  |

---

##  Optimization Comparison
| Feature             | Traditional (TF-IDF + LR) | Modern (BERT)               |
|---------------------|---------------------------|------------------------------|
| Accuracy            |  Higher (90%)            |  Lower (82%)             |
| Speed               |  Faster                |  Slower                 |
| Resources           |  CPU only              |  Needs GPU              |
| Deployment          |  Lightweight           |  Heavy                  |
| Interpretability    |  Transparent            |  Black-box              |
| Fine-Tuning         |  Not required          |  Essential              |

---

##  Challenges
- BERT needed a GPU and large memory
- Fine-tuning BERT with limited data caused underfitting
- Token truncation impacted BERT accuracy
