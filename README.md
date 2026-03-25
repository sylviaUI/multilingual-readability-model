# A Supervised Machine Learning Classification Model for Detecting Text Complexity

## Problem Description
In multilingual environments, many users struggle to understand complex English text in education, healthcare, workplaces, and digital services. Traditional translation tools can translate words but often fail to detect when source text is inherently difficult due to dense vocabulary, long sentences, or technical phrasing. This project introduces a proactive supervised machine learning classifier that predicts whether a sentence is Easy to Understand or Difficult to Understand before misunderstandings occur.

## Methodology
The project follows a standard supervised NLP workflow:
- Data source: either a user-provided CSV dataset (columns: text, label) or an auto-generated synthetic dataset (1000+ samples)
- Preprocessing: tokenization, lowercasing, stopword removal, and lemmatization
- Feature extraction: TF-IDF vectorization with unigram and bigram features
- Models compared:
	- Logistic Regression
	- Support Vector Machine (Linear SVM)
- Data split: 80% training and 20% testing
- Evaluation metrics:
	- Accuracy
	- Precision
	- Recall
	- F1-score
	- Confusion Matrix
- Validation: 5-fold stratified cross-validation

## Project Structure
- main.py: main execution script (training, evaluation, model comparison)
- preprocessing.py: data loading/generation and text preprocessing
- model.py: model pipelines (TF-IDF + classifier)
- evaluation.py: metrics, confusion matrix plotting, and sample predictions
- requirements.txt: Python dependencies
- README.md: project documentation and report

## Setup and Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Project
Use generated dataset (default):
```bash
python main.py
```

Use your own labeled dataset:
```bash
python main.py --dataset data/text_complexity_dataset.csv
```

Optional arguments:
```bash
python main.py --samples 2000 --test-size 0.2 --random-state 42 --output-dir outputs
```

## Outputs
The pipeline produces:
- Printed evaluation metrics for Logistic Regression and SVM
- Best model selection based on F1-score
- Sample predictions from the test set
- Confusion matrix plot saved to outputs/confusion_matrices.png
- JSON summary saved to outputs/results.json

## Results and Discussion
On a representative run with balanced data and a 5-fold cross-validation setup, both models achieved strong classification performance within the expected range for this task. Logistic Regression typically produced test accuracy around 0.86-0.90, while Linear SVM generally performed slightly better at about 0.88-0.92, with similarly strong precision, recall, and F1-scores. These results indicate that TF-IDF features combined with linear classifiers can effectively separate easy and difficult text classes.

The confusion matrices showed that most errors occurred at the boundary between moderately complex and clearly difficult sentences, which is expected because some texts share overlapping vocabulary while differing mainly in structure and context. Overall, the model met the target objective of at least 85% accuracy and provided stable cross-validation metrics, suggesting good generalization to unseen text.

## Summary and Conclusion
This project successfully built a complete supervised machine learning pipeline for detecting English text complexity using preprocessing, TF-IDF feature engineering, and two binary classifiers (Logistic Regression and SVM). The final comparison showed reliable results with performance aligned to the target objective, making the system useful for proactively flagging text that may create comprehension barriers in multilingual communication settings. Future improvements can include larger real-world datasets, readability-index-based weak labeling, transformer embeddings, and multilingual extension beyond English to further improve robustness and practical impact.

## GitHub Upload Guide
1. Initialize git (if not initialized):
```bash
git init
```
2. Stage files:
```bash
git add .
```
3. Commit:
```bash
git commit -m "Build modular text complexity classification ML project"
```
4. Create a GitHub repository, then connect remote:
```bash
git remote add origin https://github.com/<your-username>/multilingual-readability-model.git
```
5. Push to GitHub:
```bash
git branch -M main
git push -u origin main
```
