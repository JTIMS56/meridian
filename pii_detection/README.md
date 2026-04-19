# MERIDIAN — PII Detection Module

## Overview
This module extends the MERIDIAN geopolitical intelligence platform with a **Personally Identifiable Information (PII) Detection and Removal** pipeline. It implements four models of increasing complexity to identify and redact PII entities in educational text data, aligned with the [Kaggle PII Data Detection competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data).

## PII Entity Types
| Label | Description | Example |
|-------|-------------|---------|
| `NAME_STUDENT` | Full name of a student | "John Smith" |
| `EMAIL` | Email address | "jsmith@university.edu" |
| `USERNAME` | Online username | "jsmith2024" |
| `ID_NUM` | ID or account number | "A12345678" |
| `PHONE_NUM` | Phone number | "(555) 123-4567" |
| `URL_PERSONAL` | Personal URL/website | "github.com/jsmith" |
| `STREET_ADDRESS` | Physical address | "123 Main St, Apt 4" |

## Models
1. **Logistic Regression + TF-IDF** — interpretable baseline
2. **Random Forest** — engineered features with regex & context windows
3. **BiLSTM-CRF** — sequence labeling with word embeddings
4. **DeBERTa Transformer** — fine-tuned token classification (SOTA)

## Directory Structure
```
pii_detection/
├── config.py                # Paths, hyperparameters, label mappings
├── run_pii_pipeline.py      # End-to-end orchestrator
├── train.py                 # Unified training entrypoint
├── evaluate.py              # Unified evaluation & visualization
├── preprocessing/
│   ├── tokenizer.py         # Tokenization & alignment utilities
│   └── feature_eng.py       # Feature extraction (regex, shape, context)
├── models/
│   ├── baseline_logreg.py   # Model 1: Logistic Regression
│   ├── random_forest.py     # Model 2: Random Forest
│   ├── bilstm_crf.py        # Model 3: BiLSTM-CRF
│   └── transformer_ner.py   # Model 4: DeBERTa token classifier
├── evaluation/
│   ├── metrics.py           # Precision, recall, F1, AUC-ROC
│   └── visualizations.py    # Confusion matrices, ROC curves, error analysis
├── utils/
│   ├── pii_patterns.py      # Regex patterns for PII types
│   └── data_loader.py       # Kaggle dataset loading & BIO conversion
├── data/                    # Kaggle dataset (not committed)
│   └── .gitkeep
└── notebooks/
    └── pii_exploration.ipynb  # EDA & model comparison notebook
```

## Quick Start
```bash
# 1. Download Kaggle data into pii_detection/data/
#    - train.json, test.json

# 2. Run full pipeline
cd C:\Users\dbely\Documents\meridian
py -3.11 pii_detection/run_pii_pipeline.py --mode train --model all

# 3. Evaluate all models
py -3.11 pii_detection/evaluate.py --output results/
```

## Integration with MERIDIAN
This module shares MERIDIAN's NLP preprocessing patterns (tokenization, entity
recognition) and follows the same environment-aware deployment conventions
(`RENDER=true` for cloud vs. local execution).
