from sklearn import metrics, tree
from aix360.metrics import *
from shap import KernelExplainer
from lime.lime_tabular import LimeTabularExplainer
from pyexplainer.pyexplainer_pyexplainer import *
import re
from operator import itemgetter
from sklearn.metrics.pairwise import  euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def generate_explanations(explainer,X_test,y_test, global_model):
    explanations = []
    if isinstance(explainer, KernelExplainer):
        shap_values = explainer.shap_values(X_test)

    for i in range(0,len(X_test)):
        X_explain = X_test.iloc[[i]]
        y_explain = y_test.iloc[[i]]
        row_index = str(X_explain.index[0]) #name of instance
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

        elif isinstance(explainer, PyExplainer):
            explanation = explainer.explain(X_explain, y_explain, search_function ='CrossoverInterpolation',)
            explanation['name'] = row_index
            explanation['local_model'] = explanation['local_rulefit_model']
            del explanation['local_rulefit_model']
            # explanation['local_prediction'] =  explanation['local_model'].predict_proba( test_data_x.iloc[[i]].values)[0][0]
            print(i)
        
        elif isinstance(explainer, KernelExplainer):
            explanation = {}
            explanation['name'] = row_index
            explanation['shap_values'] = shap_values[i]
            # row_values, row_expected_values, row_mask_shapes, main_effects = explainer.explain_row()

        explanations.append(explanation)
    return explanations 

def get_explanations(project,explain_method,model_name,X_train,y_train,global_model,random_state=1,class_label =['clean','defect'] ):
    """
    lime explanation
        name
        rule: (lime Explanation object)
        synthetic_instance_for_global_model: original generated synthetic instances - according to gaussian sampling around mean of feature data; shape=(num_samples,original num of features)
        synthetic_instance_for_local_model: synthetic instances with selected discretised features (thus binary feature values) for training local model; shape=(num_samples,num_features selected)
        local_model: local surrogate model (Ridge Regressor)
        selected_feature_indices: index of selected features by "feature selection" method
    
    py explanation
        name
        synthetic_data: generated synthetic instances using crossover and interpolation
        synthetic_predictions: corresponding predictions by global task model
        X_explain, y_explain: Instance to be explained 
        indep: independent variables(features)
        dep: target variable 
        top_k_positive_rules
        top_k_negative_rules
        local_model: local surrogate model (RuleFit model)

    shap explanation
        name
        shap_values
    """
    # explain_method - str: 'lime','pyExp', 'shap'
    filepath = 'explanations/' + project.name + '_'+ model_name +'_'+explain_method+'_explanations' +'.pkl'

    if explain_method == 'lime':
        explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, 
                                        class_names=class_label, random_state=random_state)
    elif explain_method == 'pyExp':
        explainer = PyExplainer(X_train, y_train, X_train.columns, 'defect', global_model, class_label)
    elif explain_method == 'shap':
        f = lambda x: global_model.predict_proba(x)[:,1]
        explainer = KernelExplainer(f, X_train)
    else: 
        print('explain_method must be either lime, pyExp or shap')

    if os.path.exists(filepath):
        with open(filepath,'rb') as f:
            explanations = pickle.load(f)
    else:
        print("file at",filepath)
        test_data_x,test_data_y,_= project.get_sampled_data(model_name)
        explanations = generate_explanations(explainer,test_data_x,test_data_y,global_model)
        try: 
            if not os.path.exists("explanations/"):
                os.makedirs("explanations/")
            with open(filepath,'wb') as f:
                pickle.dump(explanations,f)
        except:
            print("error storing in file")

    return explainer,explanations

def evaluate_model_performance(project,model_name):
    """
    print performance scores for original task model - train,test accuracy, precision, recall, f1
    """
    y_test = project.y_test.values
    model = project.models[model_name]

    print("Dataset project: "+project.name)
    if model_name == 'BRCG':
        pred = model.predict(project.X_test_bin)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train_rs, model.predict(project.X_train_bin))*100)
  
    elif model_name == 'LogRR':
        pred = model.predict(project.X_test_bin, project.X_testStd)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train_rs, model.predict(project.X_train_bin,project.X_trainStd))*100)

    elif model_name == 'NN':
        print(model_name + " Training accuracy(in %):", model.evaluate(project.X_trainNorm, project.y_train_rs, verbose=0)[1] *100)
        print(model_name + " Test accuracy(in %):", model.evaluate(project.X_testNorm, y_test, verbose=0)[1] *100)
        return
    
    elif  model_name == 'LR':
        pred = model.predict(project.X_test.values)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train.values, model.predict(project.X_train.values))*100)
    elif  model_name == 'SVM':
        pred = model.predict(project.X_test.values)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train.values, model.predict(project.X_train.values))*100)
    elif model_name == 'KNN':
        pred = model.predict(project.X_test.values)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train.values, model.predict(project.X_train.values))*100)
    elif model_name == 'NB':
        pred = model.predict(project.X_test.values)
        print(model_name + " Training accuracy(in %):", metrics.accuracy_score(project.y_train.values, model.predict(project.X_train.values))*100)
        
    print(model_name + " Test accuracy(in %):", metrics.accuracy_score(y_test, pred)*100)
    print(model_name +" precision(in %):", metrics.precision_score(y_test, pred)*100)
    print(model_name +" recall(in %):", metrics.recall_score(y_test, pred)*100)
    print(model_name +" f1(in %):", metrics.f1_score(y_test, pred)*100)


# Util functions for evaluations

def get_features_from_rules(rules):
    """
    return set of features extracted from rules string 
    (RuleFit model output in PyExplainer explanation)
    """
    features = []
    for rule in rules['rule']:
        rule_feats = re.findall('[a-zA-Z]+',rule)
        features = features + rule_feats
    # print(features)
    return set(features)

def feature_boundary_values(rule):
    """
    input : rfc > 36.65500068664551 & loc > -1015.6700439453125 & lcom > -815.7749938964844
    return : list of tuples [(feature, value)] where value of features are within stated range
    """
    boundaries = []
    props = rule.split(" & ")

    for prop in props:
        sep = prop.split(" ")
        if "=" in sep[1]:
            value = float(sep[2])
        elif sep[1] == '>':
            value = np.ceil(float(sep[2]))
        elif sep[1] == '<':
            value = np.floor(float(sep[2]))
        boundaries.append((sep[0],value))

    return boundaries

def feature_outside_boundary_values(rule):
    """
    input : rfc > 36.65500068664551 & loc > -1015.6700439453125 & lcom > -815.7749938964844
    return : list of tuples [(feature, value)] where value of features are outside stated range
    """
    values = []
    props = rule.split(" & ")

    for prop in props:
        sep = prop.split(" ")
        if '>' in sep[1]:
            value = float(sep[2]) - 1
        elif '<' in sep[1]:
            value = float(sep[2]) + 1

        values.append((sep[0],value))
    return values


# Below are functions for evaluation metrics - given a test set test_x,test_y, 3 explanations each (100)

def prediction_fidelity(global_preds,lime_preds,py_preds,shap_preds):
    """
    shows level of (dis)agreement between task model predictions and surrogate model predictions using mcc, recall, precision
    """
    results = []

    lime_res = {}
    lime_res['method'] = 'lime'
    lime_res['mcc'] = metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(lime_preds)>0.5)
    lime_res['recall'] = metrics.recall_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5)
    lime_res['precision'] = metrics.precision_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5)
    results.append(lime_res)
    # print(lime_res)

    py_res = {}
    py_res['method'] = 'pyExplainer'
    py_res['mcc'] = metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(py_preds)>0.5)
    py_res['recall'] = metrics.recall_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5)
    py_res['precision'] = metrics.precision_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5)
    results.append(py_res)
    # print(py_res)

    shap_res = {}
    shap_res['method'] = 'shap'
    shap_res['mcc'] = metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(shap_preds)>0.5)
    shap_res['recall'] = metrics.recall_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5)
    shap_res['precision'] = metrics.precision_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5)
    results.append(shap_res)
    # print(shap_res)

    return results

def internal_fidelity(global_model, X_test, y_test, lime_explanations, py_explanations, shap_explanations,):
    """
    Shows how well the explanation reflects the decision making process of original task model by 
    comparing feature importance values generated by interpretable task model vs explanations
    
    global_model: interpretable task model (Logistic Regressor)
    return: 
        1 dict for each method with keys:
            method: method name
            avg_recall: average recall score across all test instances
            recalls: list of recall scores for all test instances
    """
    # 1. can extend to adapt to more interpretable models
    # 2. can possibly add precision score - how many of the top 10 features(by explanations) is truly important(weights within top quartile of task model)
    features = X_test.columns
    result = []

    exp_list = { 'LIME': lime_explanations, 'PyExplainer': py_explanations, 'SHAP':shap_explanations }
    true_f = [ np.abs(x) for x in global_model.coef_[0]]
    feature_importances = pd.Series(data=true_f)
    feature_importances = feature_importances.sort_values(ascending=False)

    for method in exp_list:
        method_res = {}
        method_res['method'] = method 
        recalls = []
        for exp_instance in exp_list[method]: # 100 instances
            true_features_indices = feature_importances[:10].index

            if (method == 'LIME'):
                rules = exp_instance['rule']
                exp_feature_indices = exp_instance['selected_feature_indices']
                exp_importances = [np.abs(rules.as_list()[i][1]) for i in range(len(rules.as_list()))]
                top_importance_boundary = np.quantile(exp_importances, .75)

                recall = len(set(exp_feature_indices) & set(true_features_indices))/len(true_features_indices)
                recalls.append(recall)
                # instance['precision'] = 
                # top_relevant_features = []
                # for i,imp in rules.as_map()[1]:
                #     if imp < top_importance_boundary:
                #         break
                #     top_relevant_features.append((i,imp)) # most relevant features (those where weights are in the top quartile) 
                
            elif (method == 'PyExplainer'):
                # method 1 - use features extracted from top 10 rules
                # if exp_instance['y_explain'][0] == True:
                #     top_rules = exp_instance['top_k_positive_rules'].head(10)
                # elif exp_instance['y_explain'][0] == False:
                #     top_rules = exp_instance['top_k_negative_rules'].head(10)
                # top_features_from_rules = get_features_from_rules(top_rules)
                # exp_feature_indices = [ i for i in range(len(features)) if features[i] in top_features_from_rules ]

                # method 2 - use indivdual feaatures and their assigned importance value (not top "features" in explanation)
                exp_features = exp_instance['local_model'].get_rules().head(20).sort_values(by='importance',ascending=False)
                exp_feature_indices = exp_features.index[:10]

                # set len of top features to min of exp features and top features (when num of top rules < 10 - only for method 1)
                if (len(exp_feature_indices)<10): 
                    true_features_indices = true_features_indices[:len(exp_feature_indices)]
                # debug 
                if (len(true_features_indices)==0):
                    print(exp_instance['name'])
                    print(exp_instance['y_explain'])
                    print(exp_instance['top_k_negative_rules'])
                    print(exp_instance['top_k_positive_rules'])
                
                recall = len(set(exp_feature_indices) & set(true_features_indices))/len(true_features_indices)
                recalls.append(recall)

            elif (method == 'SHAP'):
                exp_feature_indices = pd.Series(exp_instance['shap_values'])
                exp_feature_indices = exp_feature_indices.sort_values(ascending=False).index

                # set len of top features to min of exp features and top features
                if (len(exp_feature_indices)<10):
                    print("len of exp features: ",  exp_feature_indices)
                    true_features_indices = true_features_indices[:len(exp_feature_indices)]

                recall = len(set(exp_feature_indices) & set(true_features_indices))/len(true_features_indices)
                # top_importance_boundary = np.quantile(exp_importances, .75)

                recalls.append(recall)

        method_res['avg_recall'] = np.mean(recalls)
        method_res['recalls'] = recalls
        result.append(method_res)

    return result

def faithfulness(global_model, X_test, lime_explanations, py_explanations, shap_explanations):
    """
    indication of correctness of assigned feature importance values by comparing performance(changes in pred proba)
    when features are removed (tabular case - assign values outside important range) in order of their assigned importance. 
    """
    features = list(X_test.columns)
    def faithfulness_score(model,x,coefs,base,indices):
        """
        coefs: importance vals sorted according to decreasing importance
        indices: sorted indices according to decreasing importance
        """
        # original predicted class of the instance
        pred_class = np.argmax(model.predict_proba(x.reshape(1,-1)),axis=1)[0]

        # if clean class - negate coefficients (because pred_probs is wrt defect class)
        if pred_class == 0:
            coefs = [-c for c in coefs]
        pred_probs = []
        for ind in np.nditer(np.array(indices)):
            x_copy = x.copy()
            x_copy[ind] = base[ind]
            x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
            pred_probs.append(x_copy_pr[0][pred_class])
        corr = np.corrcoef(coefs, pred_probs)[0,1]

        if (np.isnan(corr)): # if no correlation - set to zero instead of nan
            # print(type(global_model))
            # print(pred_probs)
            return 0

        return -np.corrcoef(coefs, pred_probs)[0,1]

    result = []
    lime_faithfulness = []
    shap_faithfulness = []
    py_faithfulness = []
    
    for i in range(len(X_test)):
        # calculate for lime
        lime_exp = lime_explanations[i]['rule']
        sorted_indices = [np.abs(lime_exp.as_map()[1][i][0]) for i in range(len(lime_exp.as_map()[1]))] # sorted in order of decending importances
        coef_map = sorted(lime_exp.as_map()[1],key=itemgetter(0)) # top10 features sorted according to increasing index
        x = X_test.iloc[i].values # data row type ndarray
        coefs = [lime_exp.as_map()[1][i][1] for i in range(len(lime_exp.as_map()[1]))] # top 10 feature coefficients in descending order
        base = np.zeros(x.shape[0]) 
        fmlime = faithfulness_score(global_model,x,coefs,base,sorted_indices)
        lime_faithfulness.append(fmlime)

        # calculate for shap
        coefs_shap = np.array(shap_explanations[i]['shap_values'])
        count = np.count_nonzero(coefs_shap)
        sorted_indices = np.argsort(-np.abs(coefs_shap))[:count]
        # coefs = np.sort(coefs_shap[coefs_shap!=0])[::-1]
        coefs = [coefs_shap[i] for i in sorted_indices]
        fmshap = faithfulness_score(global_model,x,coefs,base,sorted_indices)
        shap_faithfulness.append(fmshap)

        # calculate for pyExplainer
        pred_probs = []
        pred_class = global_model.predict(x.reshape(1,-1))[0].astype(int)
        if pred_class ==1:
            top_rules = py_explanations[i]['top_k_positive_rules'].head(10)
        elif pred_class ==0:
            top_rules = py_explanations[i]['top_k_negative_rules'].head(10)
        coefs_pyexp = top_rules['importance'] # top 10 importance values

        rules = top_rules['rule']
        for rule in rules:
            x_copy = x.copy() 
            outside_range_values = feature_outside_boundary_values(rule)
            indices = [features.index(c[0]) for c in outside_range_values]
            for i,ind in enumerate(indices):
                x_copy[ind] = outside_range_values[i][1]

            x_copy_pr = global_model.predict_proba(x_copy.reshape(1,-1))
            pred_probs.append(x_copy_pr[0][pred_class])
            if (x_copy_pr[0][pred_class]==np.nan):
                print(x_copy_pr)
        fmpy = -np.corrcoef(coefs_pyexp, pred_probs)[0,1]
        if (np.isnan(fmpy)):
            fmpy = 0
        py_faithfulness.append(fmpy)

    result.append({ 'method':'LIME','avg_faithfulness':np.mean(lime_faithfulness),'faithfulness_scores':lime_faithfulness})
    result.append({ 'method':'SHAP','avg_faithfulness':np.mean(shap_faithfulness),'faithfulness_scores':shap_faithfulness})
    result.append({ 'method':'pyExplainer','avg_faithfulness':np.mean(py_faithfulness),'faithfulness_scores':py_faithfulness})

    return result

def monotonicity(global_model, X_test, lime_explanations, py_explanations, shap_explanations):
    features = list(X_test.columns)

    def monotonicity_score(model,x,coefs,base,indices):
        """
        coefs: importance vals sorted according to original index
        indices: sorted indices according to importance(descending)
        """
        ar = np.array(indices[::-1])
        # original predicted class of the instance
        pred_class = model.predict(x.reshape(1,-1))[0].astype(int)
        x_copy = base.copy()
        pred_probs = []

        for ind in np.nditer(ar): #increasing importance
            x_copy[ind] = x[ind]
            x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
            pred_probs.append(x_copy_pr[0][pred_class])

        return np.all(np.diff(pred_probs) >= 0)

    result = []
    lime_monotonicity = []
    shap_monotonicity = []
    py_monotonicity = []

    for i in range(len(X_test)):
        x = X_test.iloc[i].values # data row type ndarray
        pred_class = global_model.predict(x.reshape(1,-1))[0].astype(int)

        # calculate for lime
        lime_exp = lime_explanations[i]['rule']
        sorted_indices = [np.abs(lime_exp.as_map()[1][i][0]) for i in range(len(lime_exp.as_map()[1]))] # sorted in order of decending importances
        coef_map = sorted(lime_exp.as_map()[1],key=itemgetter(0))
        base = np.zeros(x.shape[0]) 
        coefs = [lime_exp.as_map()[1][i][1] for i in range(len(lime_exp.as_map()[1]))] # top 10 feature coefficients in descending order 
        if pred_class == 0:
            filtered_indices = [sorted_indices[i] for i in range(len(sorted_indices)) if coefs[i]<0]   
        else:
            filtered_indices = [sorted_indices[i] for i in range(len(sorted_indices)) if coefs[i]>0]   
        if len(filtered_indices) ==0:
            mlime = False # or ignore?
        else:
            mlime = monotonicity_score(global_model,x,coefs,base,sorted_indices)
        lime_monotonicity.append(mlime)

        # calculate for shap
        coefs_shap = np.array(shap_explanations[i]['shap_values'])
        count = np.count_nonzero(coefs_shap)
        sorted_indices = np.argsort(-coefs_shap)[:count]
        coefs = [coefs_shap[i] for i in sorted_indices] # coefficients in descending order of importance
        if pred_class == 0:
            filtered_indices = [sorted_indices[i] for i in range(len(sorted_indices)) if coefs[i]<0]   
        else:
            filtered_indices = [sorted_indices[i] for i in range(len(sorted_indices)) if coefs[i]>0] 
        if len(filtered_indices) ==0:
            mshap = False
        else:
            mshap = monotonicity_score(global_model,x,coefs_shap,base,sorted_indices)
        shap_monotonicity.append(mshap)

        # calculate for pyExplainer
        x_copy = base.copy()
        pred_probs = []
        pred_class = global_model.predict(x.reshape(1,-1))[0].astype(int)
        if pred_class ==1:
            top_rules = py_explanations[i]['top_k_positive_rules'].head(10)
        elif pred_class ==0:
            top_rules = py_explanations[i]['top_k_negative_rules'].head(10)
        coefs_pyexp = top_rules['importance'][::-1] # top 10 importance values assigned to rules -ascending order
        rules = top_rules['rule'][::-1] # ascending order

        # iterate rules ascending order -add more important one each round
        for rule in rules:
            boundary_values = feature_boundary_values(rule) # [(feature,value)]
            indices = [features.index(c[0]) for c in boundary_values]
            for i,ind in enumerate(indices):
                x_copy[ind] = boundary_values[i][1]
            x_copy_pr = global_model.predict_proba(x_copy.reshape(1,-1))
            pred_probs.append(x_copy_pr[0][pred_class])
        mpy = np.all(np.diff(pred_probs) >= 0)
        py_monotonicity.append(mpy)

    result.append({ 'method':'LIME','total_monotonicity':sum(lime_monotonicity),'monotonicity_scores': np.array(lime_monotonicity).astype(int)})
    result.append({ 'method':'SHAP','total_monotonicity':sum(shap_monotonicity),'monotonicity_scores':np.array(shap_monotonicity).astype(int)})
    result.append({ 'method':'pyExplainer','total_monotonicity':sum(py_monotonicity),'monotonicity_scores':np.array(py_monotonicity).astype(int)})
    
    return result

def uniqueness(global_model, X_test, lime_explanations, py_explanations, shap_explanations):
    """
    returns percentage uniqueness of top 1 most important factor in each explanation 
    """
    features = list(X_test.columns)
    result = []

    lime_top_exps = []
    shap_top_exps = []
    py_top_exps = []
    for i in range(len(X_test)):
        lime_exp = lime_explanations[i]['rule']
        top_lime = lime_exp.as_list()[0][0] # ex: loc > 543.00
        lime_top_exps.append(top_lime)
        pred_class = py_explanations[i]['local_model'].predict_proba(py_explanations[i]['X_explain'].values)[0][1]

        if pred_class > 0.5:
            py_exp = py_explanations[i]['top_k_positive_rules']['rule'].iloc[0].strip()
        else:
            py_exp = py_explanations[i]['top_k_negative_rules']['rule'].iloc[0].strip()
        
        top_py = py_exp.split(" & ") # ['loc > 19', 'wmc > 9']
        py_top_exps = py_top_exps + top_py

        shap_exp = shap_explanations[i]['shap_values']
        indices = np.argsort(-shap_exp)
        shap_top = features[indices[0]]
        shap_top_exps.append(shap_top)

    #uniqueness in percentage
    lime_uniqueness = (len(set(lime_top_exps))/len(lime_top_exps))*100
    shap_uniqueness = (len(set(shap_top_exps))/len(shap_top_exps))*100
    py_uniqueness = (len(set(py_top_exps))/len(py_top_exps))*100

    result.append({ 'method':'LIME','uniqueness': lime_uniqueness})
    result.append({ 'method':'SHAP','uniqueness': shap_uniqueness})
    result.append({ 'method':'pyExplainer','uniqueness': py_uniqueness})

    return result

def similarity(X_test, lime_explanations, py_explanations):
    """
    similarity of synthetic neighbours to instances (lime and pyExp)
    """
    result = []
    lime_euc_meds = []
    py_euc_meds = []
    results = pd.DataFrame()
    for i in range(len(X_test)):
        X_explain = X_test.iloc[[i]]
        row_index = py_explanations[i]['name']
        pyExplanation = py_explanations[i]
        limeExplanation = lime_explanations[i]

        #calculate euclidean distance between X_explain and synthetic data
        py_euc = euclidean_distances(X_explain.values,pyExplanation['synthetic_data'].values)
        lime_euc = euclidean_distances(X_explain.values,limeExplanation['synthetic_instance_for_global_model'])
        
        #calculate median 
        py_euc_med = np.median(py_euc)
        lime_euc_med = np.median(lime_euc)
        lime_euc_meds.append(lime_euc_med)
        py_euc_meds.append(py_euc_med)
        pyExp_series = pd.Series(data=[row_index,'pyExplainer',py_euc_med])
        limeExp_series = pd.Series(data=[row_index,'LIME',lime_euc_med])
       
        results = pd.concat([results,pyExp_series.to_frame(1).T], ignore_index = True)
        results = pd.concat([results,limeExp_series.to_frame(1).T], ignore_index = True)


    results.columns = ['name', 'method', 'euc_dist_med']
    result.append({ 'method':'LIME','euc_dist_med': lime_euc_meds})
    result.append({ 'method':'pyExplainer','euc_dist_med': py_euc_meds})

    return results

def show_model_performance(global_preds,lime_preds,py_preds,shap_preds):
    print('lime')
    print('precision: ', metrics.precision_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5))
    print('recall: ',metrics.recall_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5))
    print('mac: ',metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(lime_preds)>0.5))

    print('pyExplainer')
    print('precision: ',metrics.precision_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5))
    print('recall: ',metrics.recall_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5))
    print('mac: ',metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(py_preds)>0.5))

    print('shap')
    print('precision: ',metrics.precision_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5))
    print('recall: ',metrics.recall_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5))
    print('mac: ',metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(shap_preds)>0.5))
