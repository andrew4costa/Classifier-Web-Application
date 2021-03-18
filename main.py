import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

st.title("Andrew's First Example")

st.write("""
# Explore Different Classifier
Which one is the best?
""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine'))

classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest', 'MLP'))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of Dataset:', X.shape)
st.write('Number of Classes:', len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('k', 1, 15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'MLP':
        h1 = st.sidebar.slider('Hidden Layer 1', 1, 100)
        h2 = st.sidebar.slider('Hidden Layer 2', 1, 100)
        h3 = st.sidebar.slider('Hidden Layer 3', 1, 100)
        af = st.sidebar.selectbox('Select Activation Function', ('identity', 'logistic', 'tanh', 'relu'))
        batch_size = st.sidebar.slider('Batch Size', 0, 200)
        max_iter = st.sidebar.slider('Max Iterations', 1, 15)
        solver = st.sidebar.selectbox('Select Solver', ('lbfgs', 'sgd', 'adam'))
        params['H1'] = h1
        params['H2'] = h2
        params['H3'] = h3
        params['AF'] = af
        params['Batch Size'] = batch_size
        params['Max Iterations'] = max_iter
        params['Solver'] = solver
    else:
        max_depth = st.sidebar.slider('Max Depth', 2, 15)
        n_estimators = st.sidebar.slider('No. of Estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors= params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C= params['C'])
    elif clf_name == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(params['H1'], params['H2'], params['H3']),
                            activation=params['AF'], 
                            batch_size=params['Batch Size'], 
                            max_iter=params['Max Iterations'],
                            random_state=21,
                            solver=params['Solver'], 
                            tol=0.000000001, 
                            verbose=10)
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'],
                                     random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

acc = accuracy_score(y_test, y_predict)

st.write(f'Classifier: {classifier_name}')
st.write(f'Accuracy: {acc}')

# PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X)


x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component One')
plt.ylabel('Principal Component Two')
plt.colorbar()
# show plot
st.set_option('deprecation.showPyplotGlobalUse', False) # gets rid of streamlit error message
st.pyplot()
