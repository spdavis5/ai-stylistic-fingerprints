# Analyzing Structural and Stylistic Fingerprints in AI Text Classification

This repository contains the code and research paper for my LING 581 (Natural Language Processing) project at Brigham Young University. The study investigates whether AI-generated text can be reliably distinguished from human-written text using only structural and stylistic features, independent of semantic content.

## Project Overview

As Large Language Models become more advanced, detecting AI-generated text is a critical challenge. Most detection models rely on lexical features (specific words or n-grams), which can cause them to fail when applied to different topics or domains.

This project shifts the focus from content to form. We test the hypothesis that AI models leave a consistent stylistic fingerprint (characterized by metrics such as sentence length uniformity and vocabulary repetition) that can be detected regardless of the text's subject matter.

## Technologies Used

* **Python 3.11**
* **Scikit-Learn:** Logistic Regression modeling, data scaling.
* **Pandas / NumPy:** Data manipulation and feature engineering.
* **NLTK:** Part-of-Speech tagging, VADER sentiment analysis, tokenization.
* **TextStat:** Readability scoring (Flesch-Kincaid, Gunning Fog).
* **Google Colab:** Cloud-based execution environment.

## Methodology

To isolate the stylistic fingerprint, we engineered 16 numeric features divided into three categories, ignoring all semantic word vectors:

1.  **Structure:** Sentence length variance, character counts, and average word length.
2.  **Readability:** Gunning Fog Index, Flesch Reading Ease, Simpson's Index, and Herdan's C.
3.  **Morphology & Tone:** Density of specific parts of speech (nouns, adjectives, auxiliary verbs), punctuation ratios, and VADER sentiment scores.

These features were used to train a Logistic Regression model, selected for its interpretability in identifying feature importance weights.

## Findings

The study produced strong evidence for a stylistic AI fingerprint. The model achieved 88.3% accuracy on document-level text (Dataset 1), proving that style alone is a strong predictor of authorship.

**Key Indicators:**
* **AI Text:** Predicted by a high Gunning Fog Index (indicating mechanical complexity) and a high Simpson's Index (indicating low vocabulary diversity/high repetition).
* **Human Text:** Predicted by a high Sentence Length Standard Deviation, confirming that human writing exhibits significantly more structural variance ("burstiness") than AI writing.

**Limitations:**
Accuracy dropped to 70% on the sentence-level dataset (Dataset 2). This indicates that structural features require a sufficient text length to generate a reliable signal; they are less effective on short-form content.

## How to Run the Code

The project is designed to run in Google Colab. The workflow consists of two notebooks that must be executed in order.

### Step 1: Data Processing & EDA
**File:** `process_and_eda.ipynb`

1.  Open the notebook in Google Colab.
2.  The notebook uses the `kagglehub` library to automatically download the datasets.
3.  **Select your dataset:** Modify the `dataset_number` variable (1, 2, or 3) to choose the target dataset:
    * **Dataset 1:** Large-scale baseline (~487k essays)
    * **Dataset 2:** Sentence-level stress test (~19k sentences)
    * **Dataset 3:** Secondary validation set (~27k essays)
4.  The notebook cleans the data, performs feature extraction, and saves a processed CSV file (e.g., `processed_dataset1.csv`) to your connected Google Drive.

### Step 2: Model Training & Classification
**File:** `classification_modelsV4.ipynb`

1.  Open the notebook in Google Colab after completing Step 1.
2.  Ensure your Google Drive is mounted.
3.  The notebook reads the processed CSV file. Ensure the file path matches the location where Step 1 saved the data.
4.  It normalizes the features (StandardScaler), trains the Logistic Regression model, and outputs classification reports and feature importance visualizations.

## Paper

The final research paper is available in the `/report` directory:
* Davis_Spencer_Analyzing_Structural_Stylistic_AI_Fingerprints.pdf

## Contact

**Spencer Davis**
* LinkedIn: [https://www.linkedin.com/in/davisspencer1/]
* Email: dspencem@byu.edu
