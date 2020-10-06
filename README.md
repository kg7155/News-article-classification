# News article classification

Please, read the report in PDF for the description of the solution.

0. Prerequisites:
pip install lemmagen


1. First download articles using the download.py script by providing the path to the data file (URLs as rows) and information whether the articles will be used for training or test set (needed for later steps). Note that training data needs a labeled class (tab seperated) next to the URL.

python download.py data/labeled_urls.tsv training
python download.py data/unlabeled_urls.txt test


2. The evaluate.py script can be used to train models and evaluate them by splitting the data into train and test set. The script outputs the accuracy values for all models and plots the confusion matrices.

python evaluate.py data/labeled_urls.tsv


3. The classify.py script can be used to train models and classify new articles. Two parameter need to be passed: the file used for training (labeled) and the file containing URLs we want to classify. After running the script, the results will be generated in labeled_mnb.tsv for Multinomial Naive Bayes and labeled_mlr.tsv for Multinomial Logistic Regression.

python classify.py data/labeled_urls.tsv data/unlabeled_urls.txt