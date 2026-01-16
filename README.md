# 🗞️ News Article Clustering with TF-IDF & KMeans

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)


---

A machine learning solution to group **news articles** based on their content using **unsupervised learning**. This enables automated **topic discovery**, **tone-based classification**, and supports use cases like **news personalization** and **media monitoring**.

---

##  Business Objective

In the age of information overload, clustering news articles allows media platforms and businesses to:

* Deliver **personalized content feeds**
* Track emerging **topics or narratives**
* Monitor **brand mentions** across news sources
* Automate **news categorization** at scale

This project uses **TF-IDF** for text representation and **KMeans clustering** for topic grouping.

---

##  Dataset Overview

The dataset contains a simulated set of news articles with the following format:

| Column    | Description                             |
| --------- | --------------------------------------- |
| `Article` | Raw textual content of the news article |

*  Plain text, single-column dataset
*  Used for training unsupervised clustering models

---

##  Features Used

| Category   | Features       | Method             |
| ---------- | -------------- | ------------------ |
| **Text**   | `Article`      | `TfidfVectorizer`  |
| **Output** | Cluster labels | KMeans predictions |

---

##  Models Implemented

The following models and steps are implemented:

* **TF-IDF Vectorization** to convert text into sparse numerical format
* **KMeans Clustering** to assign topic-based groups
* **PCA** for dimensionality reduction and visualization
* **Silhouette Score** to evaluate cluster quality

---

## Project Structure

```
news_article_clustering_project/
├── data/
│   └── simulated_news.csv             # Input dataset
├── model/
│   ├── kmeans_model.pkl               # Trained KMeans model
│   └── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── model_training.py                  # Model training script
├── app.py                             # Streamlit app for clustering articles
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
└── .github/
    └── ISSUE_TEMPLATE/                # GitHub issue templates
```

---

## Pipeline Steps

### 1. **Text Vectorization**

* Load CSV with news content
* Convert text into numerical vectors using **TF-IDF**
* Stopword removal included

### 2. **Clustering**

* Fit **KMeans** with `n_clusters=5`
* Assign cluster labels to articles
* Save model and vectorizer using `pickle`

### 3. **Evaluation**

* Use **silhouette score** to measure cohesion and separation between clusters

### 4. **Visualization & UI**

* Load new data in a Streamlit app
* Predict clusters using trained model
* Reduce dimensions using **PCA**
* Show interactive cluster plot and downloadable results

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/amitkharche/news_article_clustering_project.git
cd news_article_clustering_project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python model_training.py
```

This will generate:

* `model/kmeans_model.pkl`
* `model/tfidf_vectorizer.pkl`

### Step 4: Run Streamlit App

```bash
streamlit run app.py
```

Upload a CSV with a column named `Article`, and get cluster predictions with PCA visualization.

---

## Requirements

```text
pandas
scikit-learn
matplotlib
seaborn
streamlit
```

*(Full list available in `requirements.txt`)*

---

## Future Enhancements

* Integrate **SHAP** or **LIME** for cluster explainability
* Add **topic labeling** via keyword extraction
* Compare with **DBSCAN**, **LDA**, or **GMM**
* Enable multilingual clustering using spaCy or transformers

---

## Contact

If you have questions or want to collaborate, feel free to connect:

* [LinkedIn – Amit Kharche](https://www.linkedin.com/in/amitkharche)
* [Medium – @amitkharche14](https://medium.com/@amitkharche)
* [GitHub – @amitkharche](https://github.com/amitkharche)

---
