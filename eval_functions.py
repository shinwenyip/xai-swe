import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from pyexplainer.pyexplainer_pyexplainer import *
from lime.lime.lime_tabular import LimeTabularExplainer
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
# from aix360.algorithms.rbm import FeatureBinarizer,BooleanRuleCG,LogisticRuleRegression

# not used
def get_datasets(project_name="ANT"):
    path = "datasets"
    if project_name == 'ANT':
        ds_train,ds_test = "ant-1.6.csv","ant-1.7.csv"
    elif project_name == 'CAMEL':
        ds_train,ds_test = "camel-1.4.csv","ant-1.6.csv"
    elif project_name == 'JEDIT':
        ds_train,ds_test = "jedit-4.2.csv","ant-4.3.csv"
    elif project_name == 'POI':
        ds_train,ds_test = "poi-2.5.csv","poi-3.0.csv"

    train_data = pd.read_csv(os.path.join(path,project_name,ds_train))
    test_data = pd.read_csv(os.path.join(path,project_name,ds_test))

    train_data = train_data.set_index('name')
    test_data = test_data.set_index('name')
    train_data.rename({"bug":"defect","lcom3":"lcomt"},axis=1,inplace=True)
    test_data.rename({'bug':'defect',"lcom3":"lcomt"},axis=1,inplace=True)

    # print(train_data.shape)
    # print(train_data.columns)
    # print(test_data.shape)


    return train_data, test_data
# not used
def get_train_test(project_name="ANT" ):
    train_data,test_data = get_datasets(project_name)
    if project_name=="ANT" or project_name=="CAMEL" or project_name=="JEDIT" or project_name=="POI":
        label = train_data.columns[-1] #bugs
        train_cols = train_data.columns[:-1].values
        X_train = train_data[train_cols]
        y_train = train_data[label]>0
        X_test = test_data[train_cols]
        y_test = test_data[label]>0

    return X_train, X_test, y_train, y_test
# not used
# def train_global_model(project_name, model_name, X_train,y_train):
#     """ Returns model object """
#     X_train, X_test, y_train, y_test= get_train_test(project_name)
#     #oversample minority class (defective class)
#     smt = SMOTE(k_neighbors=5, random_state=42, n_jobs=24)
#     X_train_rs,y_train_rs = smt.fit_resample(X_train, y_train)

#     # Binarize data and also return standardized ordinal features
#     fb = FeatureBinarizer(negations=True, returnOrd=True)
#     X_train_bin, X_trainStd = fb.fit_transform(X_train_rs)
#     X_test_bin, X_testStd = fb.transform(X_test)

#     #initialise and train model
#     if model_name =='SVM':
#         global_model = svm.SVC(kernel='rbf', probability=True, random_state=0)
#         model = Model(model_name, global_model, X_trainStd, X_testStd, y_train_rs, y_test)
#         # model = Model(model_name, global_model, X_train_rs, X_test, y_train_rs, y_test)
#     elif model_name == 'LR':
#         global_model = LogisticRegression(random_state=0, n_jobs=24)
#         model = Model(model_name, global_model, X_trainStd, X_testStd, y_train_rs, y_test)
#         # model = Model(model_name, global_model, X_train_rs, X_test, y_train_rs, y_test)
#     elif model_name == 'BRCG':
#         # Instantiate BRCG with small complexity penalty and large beam search width
#         global_model = BooleanRuleCG(lambda0=1e-3, lambda1=1e-3, CNF=True)
#         model = Model(model_name, model, X_train, X_test, y_train, y_test, X_trainStd, X_testStd)
#     elif model_name == 'BRCG' or model_name == 'LogRR':
#         model = aix.get_model(model_name,X_train_rs, X_test, y_train_rs, y_test )

    #fit and predict
    # global_model.fit(X_train,y_train)    
    # model.fit()

    # return model
    
def rq1_results():
    ant_svm = pd.read_csv('./eval_results/'+'rq1_ANT_SVM.csv')
    
    ax= sns.boxplot(data=ant_svm, x='project',y='euc_dist_med', hue='method')
    ax.set(ylim=(0, 5000))
    plt.show()

def rq2_result():
    ant_svm = pd.read_csv('./eval_results/'+'rq2_ANT_SVM_global_vs_local_synt_pred.csv')
    ant_svm['global_model'] = 'SVM'

    fig, axs = plt.subplots(1,2,figsize=(10,6))

    axs[0].set_title('AUC')
    axs[1].set_title('F1')
    axs[0].set(ylim=(0, 1))
    axs[1].set(ylim=(0, 1))

    sns.boxplot(data=ant_svm,x='global_model',y='AUC',hue='method',ax=axs[0])
    sns.boxplot(data=ant_svm,x='global_model',y='F1',hue='method',ax=axs[1])

    plt.show()

def rq2_probability_distribution():
    d = {True: 'DEFECT', False: 'CLEAN'}
    ant_svm = pd.read_csv('./eval_results/'+'rq2_ANT_SVM_probability_distribution.csv')

    mask = ant_svm.applymap(type) != bool
    ant_svm = ant_svm.where(mask, ant_svm.replace(d))
    ant_svm['global_model'] = 'SVM'
    pyexp_ant_svm = ant_svm[ant_svm['technique']=='pyExplainer']
    lime_ant_svm = ant_svm[ant_svm['technique']=='LIME']
    
    fig, axs = plt.subplots(1,2, figsize=(10,6))
    axs[0].set_title('pyExplainer')
    axs[1].set_title('LIME')
    axs[0].set(ylim=(0, 1))
    axs[1].set(ylim=(0, 1))

    sns.boxplot(data=pyexp_ant_svm,x='global_model',y='prob',hue='label',ax=axs[0])
    sns.boxplot(data=lime_ant_svm,x='global_model',y='prob',hue='label',ax=axs[1])
    plt.setp(axs.flat, xlabel='global_model', ylabel='Probability')
    plt.show()

def rq3_results():
    ant_svm = pd.read_csv('./eval_results/'+'rq3_ANT_SVM_eval_rulesplit_recall.csv')

    ant_svm['global_model'] = 'SVM'
    ant_svm['recall'] = ant_svm['recall']*100

    sns.boxplot(data=ant_svm,x='global_model', y='recall', 
                hue='method',palette=['darkorange','royalblue']).set(xlabel='', ylabel='Consistency Percentage (%)',ylim=(0, 100))
    plt.show()

    ant_svm_pyexp = ant_svm[ant_svm['method']=='pyExplainer']
    ant_svm_lime = ant_svm[ant_svm['method']=='LIME']
    
    print('pyExplainer')
    get_percent_unique_explanation('ANT','SVM',list(ant_svm_pyexp['explanation']))
    print('LIME')
    get_percent_unique_explanation('ANT','SVM',list(ant_svm_lime['explanation']))

def get_percent_unique_explanation(proj_name, global_model_name, explanation_list):
    total_exp = len(explanation_list)
    total_unique_exp = len(set(explanation_list))
    percent_unique = (total_unique_exp/total_exp)*100

    count_exp = Counter(explanation_list)
    max_exp_count = max(list(count_exp.values()))
    percent_dup_explanation = (max_exp_count/total_exp)*100

    # print(total_exp)
    # print(count_exp)
    print('% unique explanation is',round(percent_unique,2))
    print('% duplicate explanation is', round(percent_dup_explanation))

def evaluate_model_performance(project,model_name):
    y_test = project.y_test
    model = project.models[model_name]

    print("Dataset project: "+project.name)
    if model_name == 'BRCG':
        pred = model.predict(project.X_test_bin)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train_rs, model.predict(project.X_train_bin))*100)
        # print(model_name + " Test accuracy(in %):", metrics.accuracy_score(y_test, pred)*100)
  
    elif model_name == 'LogRR':
        pred = model.predict(project.X_test_bin, project.X_testStd)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train_rs, model.predict(project.X_train_bin,project.X_trainStd))*100)
        # print(model_name + " Test accuracy(in %):", metrics.accuracy_score(project.y_test, pred)*100)

    elif model_name == 'NN':
        print(model_name + " Training accuracy(in %):", model.evaluate(project.X_trainNorm, project.y_train_rs, verbose=0)[1] *100)
        print(model_name + " Test accuracy(in %):", model.evaluate(project.X_testNorm, y_test, verbose=0)[1] *100)
        return
    
    elif  model_name == 'LR':
        pred = model.predict(project.X_test)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train_rs, model.predict(project.X_train_rs))*100)
    elif  model_name == 'SVM':
        pred = model.predict(project.X_test)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train_rs, model.predict(project.X_train_rs))*100)
    print(model_name + " Test accuracy(in %):", metrics.accuracy_score(y_test, pred)*100)
    print(model_name +" precision(in %):", metrics.precision_score(y_test, pred)*100)
    print(model_name +" recall(in %):", metrics.recall_score(y_test, pred)*100)
    print(model_name +" f1(in %):", metrics.f1_score(y_test, pred)*100)


# metrics.accuracy_score(y_test, pred)

# def method_accuracy():
    # lime local model
    # randomly select 100 instances in y_test 
    # test_indexes = np.randint(0,2,100).astype(bool)
    # test_instances = y_test[test_indexes]
    # for each instance, generate explanation
    #metrics.accuracy_score(y_test,)

# def get_explanation(explainer_name,project_name, global_model,row_index,X_train, y_train, X_test, y_test,df_indices):
#     features = X_test.columns
#     dep = 'defect'
#     class_label = ['clean','defect']
#     if explainer_name == 'pyExplainer':
#         explainer = PyExplainer(X_train, y_train, features, dep, global_model, class_label)
#     elif explainer_name == 'lime':
#         explainer = LimeTabularExplainer(X_train.values, 
#                                       feature_names=features, class_names=class_label, 
#                                       random_state=0)
#     #df_indices is indices for correctly predicted defects

#     for i in range(0,len(df_indices)):
#take input explainer, xtest, ytest and generate an explanation for each instance in xtest



# X_test_100 = 
# lime explanations = generate_explanation(limeExp, X_test_100, y_test_100)
# returns the explanation 
def generate_explanations(explainer,X_test,y_test, global_model):
    for i in range(0,len(X_test)):
        X_explain = X_test.iloc[[i]]
        y_explain = y_test.iloc[[i]]
        row_index = str(X_explain.index[0]) #name of instance
        explanations = []
        # check which explainer
        if isinstance(explainer,LimeTabularExplainer):
            exp, synt_inst, synt_inst_for_local_model, selected_feature_indices, local_model = explainer.explain_instance(X_test.iloc[i], global_model.predict_proba, num_samples=2110)
            explanation = {}
            explanation['rule'] = exp
            explanation['synthetic_instance_for_global_model'] = synt_inst
            explanation['synthetic_instance_for_local_model'] = synt_inst_for_local_model #10 most important features binary rep
            explanation['local_model'] = local_model
            explanation['selected_feature_indices'] = selected_feature_indices #10 most important feature indexes
            explanation['name'] = row_index
            explanations.append(explanation)
        elif isinstance(explainer, PyExplainer):
            pyExplanation = explainer.explain(X_explain,
                                   y_explain,
                                   search_function = 'CrossoverInterpolation')
            pyExplanation['name'] = row_index
            pyExplanation['local_model'] = pyExplanation['local_rulefit_model']
            del pyExplanation['local_rulefit_model']
        
        return explanations


