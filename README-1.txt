Naive Bayes Sentiment Classifier

Author: 김수윤
Student ID: 2024311065

Files needed for this to work:

nbc_sentiment.py   -> Main implementation (Task 1, 2)
train.csv          -> Provided training dataset
test.csv           -> Provided testing dataset

How to run the program:

1. Make sure Python 3.x is installed on your system.
2. Place the files `train.csv`, `test.csv`, and `nbc_sentiment.py` in the same folder.
3. Open a terminal or command prompt and run python nbc_sentiment.py. If you have a Python IDE installed such as PyCharm, just hit run while the files are both in the game folder.

What will be included in the output of the file:

- Top 30 Unigrams and Bigrams will be printed.
- Model will be trained and evaluated.
- Final Precision, Recall, and F1 Score will be shown.
- 10 wrong predictions will be displayed for analysis.

Requirements needed for the Python code:

- No external packages required.
- Only built-in Python libraries used:
    - csv
    - re
    - math
    - collections

Notes:

- "_NEG" suffix is retained in features following "not"/"never".
- 700 unigrams + 300 bigrams = 1,000 features used as required.