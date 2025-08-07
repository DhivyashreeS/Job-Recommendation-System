# Personalized Job Recommendation System

A data-driven Flask web app that delivers the top 5 LinkedIn job openings tailored to a user’s skill profile by combining NLP, clustering, and a supervised SVM classifier.

---

## Overview  
Job seekers face an overwhelming volume of listings with inconsistent relevance, while recruiters spend hours sifting through unqualified candidates. This system bridges that gap by:  
1. **Extracting & Normalizing Skills** from user input  
2. **Vectorizing** job titles & descriptions with TF-IDF  
3. **Clustering** jobs via Truncated SVD + K-Means to reveal macro-segments  
4. **Training** an SVM classifier (76% accuracy, AUC 0.83) to rank the best matches  
5. **Serving** real-time recommendations through a lightweight Flask interface

---

## Data & Feature Engineering  
**Download Dataset**: https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024
- **LinkedIn Postings** (~527K rows): title, company, location, description, URL  
- **Skill Mappings** (~1.3M rows): standardized tags per posting  
- **Cleaning**: removed duplicates & non-US locations; parsed “city, state”; standardized text  
- **TF-IDF**: combined title + description (max_features=10 000; ngram_range=(1,2))  
- **Dimensionality Reduction**: Truncated SVD (k=100) for clustering

---

## Modeling Pipeline  
1. **Train/Test Split**: 80/20 hold-out  
2. **Baseline Classifiers**: Logistic Regression, Decision Tree, KNN, Naive Bayes  
3. **Advanced Models**: Random Forest, XGBoost, Support Vector Machine  
4. **Evaluation**: Accuracy, Precision, Recall, F1, ROC/AUC → **SVM chosen**  

---

## Web Application  
- **Backend**: Flask loads `svm_model.pkl`, `vectorizer.pkl`, `job_data.pkl`  
- **Frontend**: simple form to enter comma-separated skills, click “Search,” and view a table of Top 5 jobs (title, company, location, link)  
- **Cluster Insights** (optional): shows which of the 7 job clusters your query falls into (e.g., Data Science, Software Engineering, Sales)

---

## Usage  
1. Run the Flask app (e.g., `python app.py`)  
2. Navigate to `http://localhost:5000`  
3. Paste skills (e.g., `Python, SQL, Machine Learning`)  
4. Click **Search** and explore the Top 5 recommended LinkedIn jobs

---

## Future Enhancements  
- Integrate deep embeddings (BERT) for richer semantic matching  
- Add user-feedback loop to refine recommendations dynamically  
- Deploy as a scalable REST API on AWS/GCP  
- Expand beyond US jobs to global markets and multi-language support  

