import string, sys
import pandas as pd
import numpy as np
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

""" Zurnal24ur NEWS ARTICLE CLASSIFICATION
    Please, read README.md for usage.
"""

def explore(df):
    """ Gain insight into data
    """
    # Number of articles (all and per category)
    all = len(df["Category"])
    art_per_cat = df["Category"].value_counts()
    print("Number of articles: {}\nPer category:\n{}".format(all, art_per_cat))

    # Average length of article in each category
    print("\nAvg. article length:")
    print(df.groupby("Category")["Text"].apply(lambda x: np.mean(x.str.len())))
    print("-------------------------")


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

    print("Multinomial Naive Bayes")
    print("The training accuracy is: {:.4f}".format(accuracy_score(labels_train, mnb.predict(features_train))))
    print("The test accuracy is: {:.4f}".format(accuracy_score(labels_test, mnb_pred)))
    print("-------------------------")
    draw_confusion_matrix(mnb_pred, "Multinomial Naive Bayes")


def multinomial_log_reg():
    """ Logistic Regression for multinomial models
    """
    # Create the Logistic Regression model with the best parameters from the Random Search
    lrc = randomized_search()

    lrc.fit(features_train, labels_train)
    lrc_pred = lrc.predict(features_test)

    print("Multinomial Logistic Regression")
    print("The training accuracy is: {:.4f}".format(accuracy_score(labels_train, lrc.predict(features_train))))
    print("The test accuracy is: {:.4f}".format(accuracy_score(labels_test, lrc_pred)))

    draw_confusion_matrix(lrc_pred, "Multinomial Logistic Regression")
    

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
  

def draw_confusion_matrix(pred_labels, model):
    """ Draw a confusion matrix based on predicted class labels
    """
    aux_df = df[["Category", "Category_Code"]].drop_duplicates().sort_values("Category_Code")
    conf_matrix = confusion_matrix(labels_test, pred_labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, 
                annot=True,
                xticklabels=aux_df["Category"].values, 
                yticklabels=aux_df["Category"].values,
                cmap="Blues",
                fmt='g')
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.title("Confusion matrix - {}".format(model))
    plt.show()


if __name__ == '__main__':
    filename_labeled = sys.argv[1]
    df = pd.read_csv(filename_labeled, sep='\t', names=["URL", "Category"])

    # Load Slovenian stopwords
    f = open("helpers/stopwords-sl.txt", 'r', encoding="utf-8")
    stopwords = [line.rstrip('\n') for line in f]
    f.close()

    # Initialize Slovenian lemmatizer
    lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)

    # Load article texts
    print("Loading articles...")
    for i in range (0, len(df)):
        f = open("training_articles/{}.txt".format(i+1), 'r', encoding="utf-8")
        article_text = f.readlines()[1]
        df.loc[i, "Text"] = clean(article_text)
        f.close()

    explore(df)

    # Category name to numerical ID mapping
    category_codes = {
        "avto" : 0,
        "magazin" : 1,
        "slovenija" : 2,
        "sport" : 3,
        "svet" : 4
    }

    # Adding category code column
    df["Category_Code"] = df["Category"]
    df = df.replace({"Category_Code" : category_codes})

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Category_Code"],
                                                        test_size = 0.15,
                                                        random_state = 6)

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
    labels_test = y_test

    # Baseline accuracy: majority class
    baseline_acc_training = y_train.value_counts()[3] / len(y_train)
    baseline_acc_test = y_test.value_counts()[3] / len(y_test)

    print("Baseline training accuracy: {:.3f}".format(baseline_acc_training))
    print("Baseline test accuracy: {:.3f}".format(baseline_acc_test))
    print("-------------------------")

    # Multinomial Naive Bayes
    multinomial_naive_bayes()

    # Multinonomial Logistic Regression
    multinomial_log_reg()