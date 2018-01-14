# coding=<ascii>
import glob
import email
import pandas as pd
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# function to extract the e-mail body from an e-mail document
def get_email_body(email_file):
    b = email.message_from_string(email_file)
    if b.is_multipart():
        for payload in b.get_payload():
            body = payload.get_payload()
            return body
    else:
        body = b.get_payload()
        return body


# function to read files as string and append them to our training data
def get_data_from_file(path_to_data, classification):
    data_folder = glob.glob(path_to_data)
    training_dataset = []
    for file in data_folder:
        f = open(file, 'r')
        string_content = f.read() #read file as string
        string_content = get_email_body(string_content)
        string_content = unicode(string_content, errors='ignore')
        training_dataset.append({'data': string_content, 'label': classification})
        f.close()
    data = pd.DataFrame(training_dataset)
    return data


# set paths for data with labels
data_sources = [('./training-data/spam/*', 'spam'),
                ('./training-data/ham/*', 'ham')]

# the data is stored into a Data Frame, provided by the pandas library
# this data structure enables us to store the data in a table-like format with labelled columns
my_data = pd.DataFrame({'data': [], 'label': []})

# populate the DataFrame
for path, dclass in data_sources:
    my_data = my_data.append(get_data_from_file(path, dclass))

# split training and testing data, reserving 20% of the data for testing
train_data, test_data = train_test_split(my_data, test_size=0.2)

# Classifying using Cross Validation on SVM

# using a CountVectorizer, transform each message into a list of tokens
# this will count how many times a word occurs in each message
# this automatically gives us the bag of words for each message, labelled as ham or spam

# Build a Pipeline to perform feature extraction and training on the data
# first extract features using a CountVectorizer
# then train and predict using an SVM, SVC model.
svm_pipeline = Pipeline([
    ('bow_transformer', CountVectorizer(analyzer='word', stop_words='english', min_df=2)),
    ('classifier', SVC())
])

# parameters for pipeline so we could tune automatically
# the parameters for the classifier are set for C = 1, 10
# experimenting with a linear and a radial kernel
param_svm = [
  {'classifier__C': [1, 10], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

param_test1 = {'classifier__C': [10], 'classifier__gamma': [0.0001], 'classifier__kernel': ['rbf']}

grid_svm_skf = GridSearchCV(
    svm_pipeline,  # pipeline from above
    param_grid=param_test1,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 uses "all cores"
    scoring='accuracy',
    cv=StratifiedKFold(train_data['label'], n_folds=5),  # using StratifiedKFold CV with 5 folds
)

svm_skf = grid_svm_skf.fit(train_data['data'], train_data['label'])
predictions_svm_skf = svm_skf.predict(test_data['data'])
print "TESTING USING SVM WITH STRATIFIED K-FOLD CROSS VALIDATION"
print "Grid Scores: ", svm_skf.grid_scores_
print "Best Params: ", svm_skf.best_params_
print "Expected Output: ", test_data['label']
print "Actual Output: ", predictions_svm_skf
print "Classification Report: "
print classification_report(test_data['label'], predictions_svm_skf)


print "\n"

grid_svm_cv_5 = GridSearchCV(
    svm_pipeline,
    param_grid=param_test1,
    refit=True,
    n_jobs=-1,
    scoring='accuracy',
    cv=5,  # cross validation equal 5 will compute the scores 5 times with different splits
)

svm_cv_5 = grid_svm_cv_5.fit(train_data['data'], train_data['label'])
predictions_cv_5 = svm_cv_5.predict(test_data['data'])
print "TESTING USING SVM WITH CV = 5"
print "Grid Scores: ", svm_cv_5.grid_scores_
print "Best Params: ", svm_cv_5.best_params_
print "Expected Output: ", test_data['label']
print "Actual Output: ", predictions_cv_5
print "Classification Report: "
print classification_report(test_data['label'], predictions_cv_5)