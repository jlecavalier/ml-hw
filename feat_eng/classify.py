__author__ = "Jay LeCavalier"

import argparse
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTROPE_FIELD = 'trope'


class Featurizer:
    def __init__(self,mingram,maxgram,mindf):
        self.vectorizer = CountVectorizer(analyzer="word", ngram_range=(mingram,maxgram),
        	                              stop_words=['trope','sentence'], min_df=mindf)

    def train_feature(self, examples):
    	return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s:\n%s" % (category, "\n".join(feature_names[top10])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min', help='minimum n-gram size', type=int, default=1, required=False)
    parser.add_argument('--max', help='maximum n-gram size', type=int, default=1, required=False)
    parser.add_argument('--mindf', help='minimum document frequency', type=int, default=1, required=False)
    args = parser.parse_args()

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer(args.min,args.max,args.mindf)

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    for x in train:
    	x[kTEXT_FIELD] = " ".join([x[kTEXT_FIELD], x[kTROPE_FIELD]])

    for x in test:
    	x[kTEXT_FIELD] = " ".join([x[kTEXT_FIELD], x[kTROPE_FIELD]])

    print("Label set: %s" % str(labels))

    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    # Score how well the training fits the data using cross_val_score module
    scores = cross_val_score(lr, x_train, y_train, cv=2, scoring='accuracy')
    print(scores)
    print(scores.mean())
    print(scores.std())

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)