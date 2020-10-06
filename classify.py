import string, sys
import pandas as pd
import numpy as np
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

""" Zurnal24ur NEWS ARTICLE CLASSIFICATION
    Please, read README.md for usage.
"""

def clean(text):
    """ Perform a cleaning process
    """
    # Remove special characters and punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).replace('˝','').replace('“','').lower()

    # Remove stop words and lemmatize
    words = text.split(' ')
    cleaned = []
    for word in words:
        if (word in stopwords):
            continue
        else:
            word_lemmatized = lemmatizer.lemmatize(word) 
            # Sometimes lemmatized word becomes a stopword, do not include those
            if (word_lemmatized not in stopwords):
                cleaned.append(word_lemmatized + ' ')        
    return ''.join(cleaned)            


def multinomial_naive_bayes():
    """ Naive Bayes classifier for multinomial models
    """
    mnb = MultinomialNB()

    # Fit Naive Bayes classifier according to training vectors and target values
    mnb.fit(features_train, labels_train)

    # Perform classification on test vectors
    mnb_pred = mnb.predict(features_test)

    return mnb_pred


def multinomial_log_reg():
    """ Logistic Regression for multinomial models
    """
    # Create the Logistic Regression model with the best parameters from the Random Search
    lrc = randomized_search()

    lrc.fit(features_train, labels_train)
    lrc_pred = lrc.predict(features_test)

    return lrc_pred
    

def randomized_search():
    """ Perform randomized search on hyper parameters
    """
    C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)]
    multi_class = ['multinomial']
    solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
    class_weight = ['balanced', None]
    penalty = ['l2']

    random_grid = {'C': C,
                'multi_class': multi_class,
                'solver': solver,
                'class_weight': class_weight,
                'penalty': penalty}
    
    lrc = LogisticRegression(random_state = 8)

    random_search = RandomizedSearchCV(estimator = lrc,
                                   param_distributions = random_grid,
                                   n_iter = 50,
                                   scoring = "accuracy",
                                   cv = 3, 
                                   verbose = 1, 
                                   random_state = 8)

    random_search.fit(features_train, labels_train)

    #print("The best hyperparameters from Random Search are:")
    #print(random_search.best_params_)

    return random_search.best_estimator_
  

if __name__ == '__main__':
    filename_labeled = sys.argv[1]
    filename_unlabeled = sys.argv[2]
    
    df_training = pd.read_csv(filename_labeled, sep='\t', names=["URL", "Category"])
    df_test = pd.read_csv(filename_unlabeled, names=["URL"])

    # Load Slovenian stopwords
    f = open("helpers/stopwords-sl.txt", 'r', encoding="utf-8")
    stopwords = [line.rstrip('\n') for line in f]
    f.close()

    # Initialize Slovenian lemmatizer
    lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)

    # Load training articles and save their cleaned version to DataFrame
    print("Loading training articles...")
    for i in range (0, len(df_training)):
        f = open("training_articles/{}.txt".format(i+1), 'r', encoding="utf-8")
        article_text = f.readlines()[1]
        df_training.loc[i, "Text"] = clean(article_text)
        f.close()

    # Load test articles and save their cleaned version to DataFrame
    print("Loading test articles...")
    for i in range (0, len(df_test)):
        f = open("test_articles/{}.txt".format(i+1), 'r', encoding="utf-8")
        article_text = f.readlines()[1]
        df_test.loc[i, "Text"] = clean(article_text)
        f.close()

    # Category name to numerical (category code) mapping
    cat_to_code = {
        "avto" : 0,
        "magazin" : 1,
        "slovenija" : 2,
        "sport" : 3,
        "svet" : 4
    }

    # Inverse mapping: code to name
    code_to_cat = {v: k for k, v in cat_to_code.items()}

    # Adding category code column
    df_training["Category_Code"] = df_training["Category"]
    df_training = df_training.replace({"Category_Code" : cat_to_code})

    X_train = df_training["Text"]
    y_train = df_training["Category_Code"]
    
    X_test = df_test["Text"]

    # Text representation: TF-IDF vectors
    tfidf = TfidfVectorizer(encoding="utf-8",
                            ngram_range = (1, 1),
                            stop_words = None,
                            lowercase = False,
                            max_df = 1.,
                            min_df = 10,
                            max_features = 300,
                            norm = 'l2',
                            sublinear_tf = True)

    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
    
    features_test = tfidf.transform(X_test).toarray()
    
    # Multinomial Naive Bayes
    num_labels_test_mnb = multinomial_naive_bayes()
    text_labels_test_mnb = [code_to_cat[x] for x in num_labels_test_mnb]
    df_test["Predicted_mnb"] = text_labels_test_mnb
    
    # Multinonomial Logistic Regression
    num_labels_test_mlr = multinomial_log_reg()
    text_labels_test_mlr = [code_to_cat[x] for x in num_labels_test_mlr]
    df_test["Predicted_mlr"] = text_labels_test_mlr

    # Save the predictions to files
    df_test.to_csv("labeled_mnb.tsv", columns=["URL", "Predicted_mnb"], index=False, header=False, sep='\t')
    df_test.to_csv("labeled_mlr.tsv", columns=["URL", "Predicted_mlr"], index=False, header=False, sep='\t')