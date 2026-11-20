import csv
import re
import math
from collections import Counter, defaultdict

# This is the list of common words
stop_words = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both
but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself
yourselves
""".split())

# This cleans up and tokenizes the text. Also handles negations.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9_ ]", "", text)
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word in stop_words:
            i += 1
            continue
            # Checks for negation words such as "not" or "never"
        if word in ('not', 'never') and i+1 < len(tokens):
            next_word = tokens[i+1]
            if next_word not in stop_words:
                result.append(next_word + '_NEG')
                i += 2
                continue
        result.append(word)
        i += 1
    return result

# Here we count the unigrams and bigrams across all reviews
def count_ngrams(reviews):
    uni = Counter()
    bi = Counter()
    for tokens in reviews:
        for i in range(len(tokens)):
            uni[tokens[i]] += 1
            if i < len(tokens) - 1:
                bigram = tokens[i] + ' ' + tokens[i+1]
                bi[bigram] += 1
    return uni, bi

# We load and preprocesses all reviews from a CSV file
def load_reviews(filename):
    reviews = []
    with open(filename, encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            reviews.append(preprocess_text(row['text']))
    return reviews

# Here we extract only the relevant features that matter from a review
def extract_features(text, feats):
    tokens = preprocess_text(text)
    comb = set(tokens + [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)])
    return [f for f in comb if f in feats]

# We then load labeled data and extracts their features
def load_labeled_data(filename, feats):
    dataset = []
    with open(filename, encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            label = 1 if row['stars'].strip() == '5' else 0
            feats_used = extract_features(row['text'], feats)
            dataset.append((feats_used, label))
    return dataset

# Here we train the NB model and compute prior probabilities and likelihoods by using laplace smoothing
def train_nb(data, feats):
    cls_count = defaultdict(int)
    feat_count = {0: defaultdict(int), 1: defaultdict(int)}
    for featlist, lbl in data:
        cls_count[lbl] += 1
        for f in featlist:
            feat_count[lbl][f] += 1

    probs = {c: cls_count[c]/len(data) for c in cls_count}
    likes = {0: {}, 1: {}}
    total_feats = len(feats)
    for c in [0,1]:
        total = sum(feat_count[c].values())
        for f in feats:
            likes[c][f] = (feat_count[c][f]+1) / (total + total_feats)
    return probs, likes

# This predicts the label for a single document
def predict(probs, likes, feats, doc):
    scores = {}
    for c in [0,1]:
        logprob = math.log(probs[c])
        for f in doc:
            if f in feats:
                logprob += math.log(likes[c].get(f, 1 / (sum(likes[c].values()) + len(feats))))
        scores[c] = logprob
    return max(scores, key=scores.get)

# We evaluate the model performance from the outputs precision, recall, and F1 score
def evaluate(preds, actuals):
    tp = sum((p == 1 and a == 1) for p, a in zip(preds, actuals))
    fp = sum((p == 1 and a == 0) for p, a in zip(preds, actuals))
    fn = sum((p == 0 and a == 1) for p, a in zip(preds, actuals))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1

# Execution
def main():
    print("Loading training data...")
    train_tokens = load_reviews("train.csv")
    uni, bi = count_ngrams(train_tokens)

    print("\nTop 30 Unigrams:")
    for u, c in uni.most_common(30):
        print(f"{u}: {c}")

    print("\nTop 30 Bigrams:")
    for b, c in bi.most_common(30):
        print(f"{b}: {c}")
    # The top unigrams and bigrams are selected as features
    top_uni = [x[0] for x in uni.most_common(700)]
    top_bi = [x[0] for x in bi.most_common(300)]
    top_feats = set(top_uni + top_bi)

    print("\nTraining Naive Bayes...")
    train = load_labeled_data("train.csv", top_feats)
    test = load_labeled_data("test.csv", top_feats)
    probs, likes = train_nb(train, top_feats)

    print("\nTesting...")
    preds, actual, wrong = [], [], []
    for feats_list, label in test:
        prediction = predict(probs, likes, top_feats, feats_list)
        preds.append(prediction)
        actual.append(label)
        if prediction != label and len(wrong) < 10:
            wrong.append((feats_list, label, prediction))

    prec, rec, f1 = evaluate(preds, actual)
    print("\nResults:")
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))

    print("\nWrong predictions:")
    for f, t, p in wrong:
        print(f"True: {t}, Pred: {p}, Features: {' | '.join(f)}")

if __name__ == '__main__':
    main()
