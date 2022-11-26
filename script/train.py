from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Conv1D
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier  # Ridge regression
from sklearn.linear_model import LogisticRegression  # Logistic regression
from sklearn.linear_model import SGDClassifier  # Stochastic gradient descent
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)  # Linear discriminant analysis
from sklearn.neighbors import KNeighborsClassifier  # K-nearest neighbors
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.tree import DecisionTreeClassifier  # Decision tree
from sklearn.ensemble import RandomForestClassifier  # Random forest
from sklearn.ensemble import GradientBoostingClassifier  # Gradient boosting
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.svm import SVC  # Support vector machine
from sklearn.neural_network import MLPClassifier  # Multi-layer perceptron
from sklearn.gaussian_process import GaussianProcessClassifier  # Gaussian process
from sklearn.model_selection import cross_val_score
import eli5
from eli5.sklearn import PermutationImportance

train_pkl = 'C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\data_merged(pid1to6).pkl'
test_pkl = 'C:\\Users\\uvrlab\\Downloads\\Data for test\\Test\\data_merged_test.pkl'

models = [
    RidgeClassifier(alpha=1.0, solver="auto", random_state=42),
    LogisticRegression(C=1.0, solver="lbfgs", multi_class="auto", random_state=42),
    SGDClassifier(loss="hinge", penalty="l2", alpha=0.001, random_state=42),
    LinearDiscriminantAnalysis(solver="svd", tol=0.0001),
    KNeighborsClassifier(n_neighbors=5, weights="uniform", leaf_size=30),
    GaussianNB(priors=None, var_smoothing=1e-09),
    DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        max_leaf_nodes=None,
    ),
    RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        max_leaf_nodes=None,
    ),
    GradientBoostingClassifier(
        loss="deviance",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=3,
        random_state=42,
        max_leaf_nodes=None,
    ),
    AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=42,
    ),
    SVC(C=1.0, kernel="rbf", random_state=42),
    # MLPClassifier(
    #     hidden_layer_sizes=(100,),
    #     activation="relu",
    #     solver="adam",
    #     alpha=0.0001,
    #     max_iter=200,
    #     shuffle=True,
    #     random_state=42,
    # ),
    GaussianProcessClassifier(random_state=42),
]

def dataload(fun='nn', mode='train', filepath = 'C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\data_merged(pid1to6).pkl'):
    file = filepath
    df = pd.read_pickle(file)
    y_ = df['condition'].to_numpy(dtype=np.int32)
    df.drop(["pid", "condition", "acc.x", "acc.y", "acc.z"], axis=1, inplace=True)
    col = df.columns.tolist()
    X = df.to_numpy(dtype=np.float32)    
    X = StandardScaler().fit_transform(X)
    y_ -= 1
    n_classes = 6
    if fun == 'nn':
        y = np_utils.to_categorical(y_, num_classes=n_classes)
    else:
        y = y_
    if mode == 'train':
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=77, test_size=0.2)

        return X_train, X_valid, y_train, y_valid, col
    if mode == 'test':
        return X, y

def build_1dnn(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def train():
    X_train, X_valid, y_train, y_valid, col = dataload(mode='train', filepath=train_pkl)
    model = build_1dnn(X_train.shape[1], y_train.shape[1])
    # model = load_model('train_model.h5')
    history = model.fit(X_train, y_train, verbose=1, epochs=10)
    yhat = model.predict(X_valid)
    yhat = yhat.round()
    acc = accuracy_score(y_valid, yhat)
    # model.save('train_model(dropout)_nodevice.h5')
    print('>%3f'%acc)

    perm = PermutationImportance(model, random_state=1).fit(X_train,y_train)
    eli5.show_weights(perm, feature_names = col)

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("loss.png")
    plt.close()

    labels = ["c1", "c2", "c3", "c4", "c5", "c6"]
    cm = confusion_matrix(y_valid.argmax(axis=1), yhat.argmax(axis=1))
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig('confusion_matrix(dropout)_nodevice.png')

def test():
    X_test, y_test = dataload(mode='test', filepath=test_pkl)
    model = build_1dnn(X_test.shape[1], y_test.shape[1])
    model = load_model('train_model(dropout)_nodevice.h5')   
    results = model.evaluate(X_test, y_test, verbose=1)
    yhat = model.predict(X_test)
    yhat = yhat.round()
    acc = accuracy_score(y_test, yhat)
    print('loss> %3f'%results)
    print('acc>%3f'%acc)
    # print('>loss : %3f, >acc'%(results[0], results[1])) 

    labels = ["c1", "c2", "c3", "c4", "c5", "c6"]
    cm = confusion_matrix(y_test.argmax(axis=1), yhat.argmax(axis=1))
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig('test_confusion_matrix(dropout).png')

def train_and_test(model):
    print(model.__class__.__name__)
    file = train_pkl
    df = pd.read_pickle(file)
    X = df 
    X = StandardScaler().fit_transform(X)
    y = df['condition']
    df.drop(["pid", "condition", "acc.x", "acc.y", "acc.z"], axis=1)
    # train the model
    X_train, X_valid, y_train, y_valid, col = dataload(fun='svm', mode='train', filepath=train_pkl)
    model.fit(X_train, y_train)
    # get importance
    if model.__class__ in [
        DecisionTreeClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
    ]:
        importances = model.feature_importances_
        importance = model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print("  Feature: %0d, Score: %.5f" % (i, v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
    # make predictions
    y_pred = model.predict(X_valid)
    # evaluate predictions
    print("Train  Accuracy: ", accuracy_score(y_valid, y_pred))

    X_test, y_test = dataload(fun='svm', mode='test', filepath=test_pkl)
    # results = model.evaluate(X_test, y_test)
    yhat = model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    # print('loss> %3f'%results)
    print('TEST acc>%3f'%acc)
    # evaluate predictions
    cm = confusion_matrix(y_test, yhat)
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig('test_confusion_matrix_%s.png'%model.__class__.__name__)
    # print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    # print("Classification report:\n", classification_report(y_test, y_pred))

def test_ml(model):
    print(model.__class__.__name__)
    # train the model
    X_test, y_test = dataload(fun='svm', mode='test', filepath=test_pkl)
    model = load_model('train_%s.h5'%model.__class__.__name__)
    results = model.evaluate(X_test, y_test)
    yhat = model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    print('loss> %3f'%results)
    print('acc>%3f'%acc)
    y_pred = model.predict(X_test)
    # evaluate predictions
    print("  Accuracy: ", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, yhat)
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig('test_confusion_matrix_%s.png'%model.__class__.__name_)
    # print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    # print("Classification report:\n", classification_report(y_test, y_pred))



if __name__ == '__main__':
    for model in models:
        train_and_test(model)
