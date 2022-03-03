import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from pyexplainer.pyexplainer_pyexplainer import *
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from collections import Counter
from aix360.algorithms.rbm import FeatureBinarizer,BooleanRuleCG,LogisticRuleRegression
# for neural networks
# import tensorflow as tf
import keras.models as km
from keras.layers import Dense
# from aix360.algorithms.contrastive import CEMExplainer, KerasClassifier 
# from aix360.algorithms.protodash import ProtodashExplainer
# from keras.models import Sequential, Model, load_model, model_from_json
# tf.compat.v1.disable_eager_execution()

class Project:
    
    def __init__(self, project_name):
        self.name = project_name
        self.models = {}
        self.explainer_dir = os.path.join('./explainer/',project_name)
        self.model_dir = os.path.join('./models/',project_name)
        self.random_state = 0

    def get_datasets(self):
        """ return train_data, test_data for this project"""
        path = "datasets"
        if self.name == 'ANT':
            ds_train,ds_test = "ant-1.6.csv","ant-1.7.csv"
        elif self.name == 'CAMEL':
            ds_train,ds_test = "camel-1.4.csv","camel-1.6.csv"
        elif self.name == 'JEDIT':
            ds_train,ds_test = "jedit-4.2.csv","jedit-4.3.csv"
        elif self.name == 'POI':
            ds_train,ds_test = "poi-2.5.csv","poi-3.0.csv"

        train_data = pd.read_csv(os.path.join(path,self.name,ds_train))
        test_data = pd.read_csv(os.path.join(path,self.name,ds_test))

        train_data = train_data.set_index('name')
        test_data = test_data.set_index('name')
        train_data.rename({"bug":"defect","lcom3":"lcomt"},axis=1,inplace=True)
        test_data.rename({'bug':'defect',"lcom3":"lcomt"},axis=1,inplace=True)

        return train_data, test_data

    #set and get X_train, X_test, y_train, y_test
    def set_train_test(self):
        train_data,test_data = self.get_datasets()
        if self.name =="ANT" or self.name=="CAMEL" or self.name=="JEDIT" or self.name =="POI":
            label = train_data.columns[-1] #bugs
            train_cols = train_data.columns[:-1].values
            self.X_train = train_data[train_cols]
            self.y_train = train_data[label]>0
            self.X_test = test_data[train_cols]
            self.y_test = test_data[label]>0
        else:
            print("project not available for testing")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def prepare_data(self):
        """ 1. Oversample minority class (defective class) for training data
            2. Binarize data (feature cols) for rule based models (BRCG, LogRR)
            3. Standardise data (feature cols)
            4. Normalise data 
        """
        #oversample minority class (defective class)
        smt = SMOTE(k_neighbors=5, random_state=42, n_jobs=24)
        self.X_train_rs, self.y_train_rs = smt.fit_resample(self.X_train, self.y_train)

        # Binarize data and also return standardized ordinal features
        fb = FeatureBinarizer(negations=True, returnOrd=True)
        self.X_train_bin, self.X_trainStd = fb.fit_transform(self.X_train_rs)
        self.X_test_bin, self.X_testStd = fb.transform(self.X_test)
        self.featureBinarizer = fb
        self.X_trainNorm, self.X_testNorm = normalise_data(self.X_train_rs,self.X_test)

    def train_global_model(self, model_name):
        #initialise and train model
        if model_name =='SVM':
            global_model = svm.SVC(kernel='rbf', probability=True, random_state=self.random_state)
            # global_model.fit(self.X_trainNorm, self.y_train_rs)
            global_model.fit(self.X_train_rs, self.y_train_rs)
        elif model_name == 'LR':
            global_model = LogisticRegression(random_state=self.random_state, n_jobs=24)
            global_model.fit(self.X_train_rs, self.y_train_rs)
            # global_model.fit(self.X_trainStd, self.y_train_rs)
        elif model_name == 'BRCG':
            # Instantiate BRCG with small complexity penalty and large beam search width
            global_model = BooleanRuleCG(lambda0=1e-3, lambda1=1e-3, CNF=True)
            global_model.fit(self.X_train_bin, self.y_train_rs)
        elif model_name == 'LogRR':
            global_model = LogisticRuleRegression(lambda0=1e-3, lambda1=1e-3, useOrd=True)
            global_model.fit(self.X_train_bin, self.y_train_rs, self.X_trainStd)
        # elif model_name == 'NN':
        #     global_model = nn_small()
        #     global_model.compile(loss=fn, optimizer='adam', metrics=['accuracy'])
        #     global_model.summary()
        #     global_model.fit(self.X_trainNorm, self.y_train_rs,batch_size=128, epochs=500, verbose=1, shuffle=False)
        #     global_model.save_weights(self.name+'_'+ '_nnsmall.h5') # to load later: global_model.load_weights(filename)
        # add model to list of models in this project
        self.models[model_name] = global_model

        # Save model to a file - eg. models/ANT/SVM.pkl
        # pickle.dump(model_object,open(self.model_dir + '/' + model_name + '.pkl','wb'))

        return global_model

# def nn_small():
#     # Set random seeds for repeatability
#     np.random.seed(1) 
#     tf.random.set_seed(1)
#     model = km.Sequential()
#     model.add(Dense(10, input_dim=20, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(2, kernel_initializer='normal'))    
#     return model 

# loss function
# def fn(correct, predicted):
#     return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

def normalise_data(X_train, X_test):
    Z = np.vstack((X_train, X_test))
    Zmax = np.max(Z, axis=0)
    Zmin = np.min(Z, axis=0)

    N = normalize(Z,Zmin, Zmax)
    X_train_norm = N[0:X_train.shape[0], :]
    X_test_norm = N[X_train.shape[0]:, :]

    return X_train_norm,X_test_norm

#normalize an array of samples to range [-0.5, 0.5]
def normalize(V,Zmin,Zmax):
    VN = (V - Zmin)/(Zmax - Zmin)
    VN = VN - 0.5
    return(VN)