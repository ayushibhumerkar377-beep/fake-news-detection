FAKE NEWS DETECTION 

OVERVIEW 
A machine learning project to classify news articles as Real or Fake using NLP and text classification techniques.

Technologies 
Python 
Pandas & Numpy 
Scikit-learn 
Matplotlib & Seaborn 

Methodology 

1. Text processing and cleaning 
2. Title + articl text combination
3. TF-IDF feature extraction
4. Train-test split 
5. Model training and evaluation

Models used 
-KNN
-Logistic Regression
-Random Forest 
-Neural Network 

Evaluation 
Models are compared using :

-Accuracy 
-precision 
-Recall 
-F1-score
-Confusion Matrix 

Dataset 
the project uses a news article dataset containing Title, full text, category, and lable.

The dataset contains 1200 recordes, including 600 Real and 600 Fake news articles

Results
All four models achieved 1.00 Accuracy, Precision, Recall, and F1-score on the test set for this dataset.

The unusually high performance may be related to the characteristics and separability of the provided dataset.

Project output 
The project generates graph for:
-Class distribution 
-Model accuracy comparison 
-All metrics comparison
-Confusion matrices 
-Random Foresst feature importance 
-model comparison CSV

How to run 
bash 
pip install -r requirements.txt
python fake_news.py 


