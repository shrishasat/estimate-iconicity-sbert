
# Iconicity Prediction using SBERT + Ridge Regression

This project builds a machine learning pipeline to predict iconicity ratings of English words using SBERT embeddings and Ridge regression. It enables estimation of iconicity for words not covered in existing human-rated datasets. The predicted ratings are integrated with MEG-aligned audiovisual speech data for neuro-linguistic analyses.

##  Overview

We start with a dataset of ~14,000 human-rated iconicity values, extract SBERT embeddings for words, and train a Ridge regression model. For all remaining words (without human ratings), we estimate iconicity scores using this model.

The final output is an Excel file with corrected MEG onset timings and newly predicted iconicity ratings for each word.

##  Methods

- Model: `all-mpnet-base-v2` from [SentenceTransformers](https://www.sbert.net/)
- Regressor: Ridge Regression (`sklearn`)
- Input: Human-rated iconicity data + MEG-aligned text data
- Output: Excel file with predicted iconicity values


## Authors

**Shrisha Sathishkumar**  
Research assistant 
Centre for Human Brain Health, University of Birmingham  


**Dr. Hyojin Park**  
Principal Investigator, NEURECA Lab  
Centre for Human Brain Health, University of Birmingham  

---

## Setup

```bash
pip install pandas numpy scikit-learn sentence-transformers openpyxl
