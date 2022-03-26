from sklearn import metrics, tree
from aix360.metrics import *
from shap import KernelExplainer
from lime.lime_tabular import LimeTabularExplainer
from pyexplainer.pyexplainer_pyexplainer import *
import re
from operator import itemgetter

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
        # faithfulness = faithfulness_metric(explanation['local_model'],X_explain,)

        explanations.append(explanation)
    return explanations 

def get_explanations(project_name,explain_method,model_name,X_train,y_train,global_model,random_state=1,class_label =['clean','defect'] ):
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
    filepath = 'explanations/' + project_name + '_'+ model_name +'_'+explain_method+'_explanations' +'.pkl'

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
        explanations = pickle.load(open(filepath,'rb'))
    else:
        explanations = generate_explanations(explainer,test_data_x,test_data_y,global_model)
        pickle.dump(explanations,open(filepath,'wb'))

    return explainer,explanations

# util functions for evaluations

def get_features_from_rules(rules):
    features = []
    for rule in rules['rule']:
        rule_feats = re.findall('[a-zA-Z]+',rule)
        features = features + rule_feats
    # print(features)
    return set(features)

def feature_boundary_values(rule):
    """
    input : rfc > 36.65500068664551 & loc > -1015.6700439453125 & lcom > -815.7749938964844
    return list of tuples (feature, value)
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
    return list of tuples (feature, value)
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
    shows level of (dis)agreement between task model predictions and surrogate model predictions
    """
    results = []

    lime_res = {}
    lime_res['method'] = 'lime'
    lime_res['mcc'] = metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(lime_preds)>0.5)
    lime_res['recall'] = metrics.recall_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5)
    lime_res['precision'] = metrics.precision_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5)
    results.append(lime_res)
    print(lime_res)

    py_res = {}
    py_res['method'] = 'pyExplainer'
    py_res['mcc'] = metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(py_preds)>0.5)
    py_res['recall'] = metrics.recall_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5)
    py_res['precision'] = metrics.precision_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5)
    results.append(py_res)
    print(py_res)

    shap_res = {}
    shap_res['method'] = 'lime'
    shap_res['mcc'] = metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(shap_preds)>0.5)
    shap_res['recall'] = metrics.recall_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5)
    shap_res['precision'] = metrics.precision_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5)
    results.append(shap_res)
    print(shap_res)

    return results

def show_model_performance(global_preds,lime_preds,py_preds,shap_preds):
    print('lime')
    print('precision: ', metrics.precision_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5))
    print('recall: ',metrics.recall_score(np.array(global_preds)>0.5,np.array(lime_preds)>0.5))
    print('mac: ',metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(lime_preds)>0.5))
    # print(metrics.log_loss (np.array(global_preds)>0.5,np.array(lime_preds)>0.5))
    print('avg probability diff: ',avg_proba_diff(global_preds, lime_preds))

    print('pyExplainer')
    print('precision: ',metrics.precision_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5))
    print('recall: ',metrics.recall_score(np.array(global_preds)>0.5,np.array(py_preds)>0.5))
    print('mac: ',metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(py_preds)>0.5))
    print('avg probability diff: ',avg_proba_diff(global_preds, py_preds))

    print('shap')
    print('precision: ',metrics.precision_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5))
    print('recall: ',metrics.recall_score(np.array(global_preds)>0.5,np.array(shap_preds)>0.5))
    print('mac: ',metrics.matthews_corrcoef(np.array(global_preds)>0.5,np.array(shap_preds)>0.5))
    print('avg probability diff: ',avg_proba_diff(global_preds, shap_preds))

def internal_fidelity(global_model, X_test, y_test, lime_explanations, py_explanations, shap_explanations,):
    """
    shows how well the explanation reflects the decision making process of original task model by 
    comparing feature importance values generated by interpretable task model vs explanations
    
    global_model: interpretable task model (Logistic Regressor)
    return: 
        1 dict for each method with keys:
            method: method name
            avg_recall: average recall score across all test instances
            recalls: list of recall scores for all test instances
    """
    # have 1 global interpretable model ()
    # for each instance in (100) test instances, we calculate recall and precision measures using 
    # true features(from global model) and explanation features
    # if (global_model==None): # use decision tree
    #     global_model = tree.DecisionTreeClassifier(random_state=1)
    #     global_model.fit(X_test,y_test)
    features = X_test.columns
    result = []

    exp_list = { 'LIME': lime_explanations, 'PyExplainer': py_explanations, 'SHAP':shap_explanations }
    true_f = [ np.abs(x) for x in global_model.coef_[0]]
    # feature_importances = pd.Series(data=true_f,index=X_test.columns)
    feature_importances = pd.Series(data=true_f)
    feature_importances = feature_importances.sort_values(ascending=False)

    true_features_indices = feature_importances[:10].index

    for method in exp_list: # for each (method) explanations
        method_res = {}
        method_res['method'] = method 
        recalls = []
        for exp_instance in exp_list[method]: # 100 instances
            if (method == 'LIME'):
                rules = exp_instance['rule']
                exp_feature_indices = exp_instance['selected_feature_indices']
                exp_importances = [np.abs(rules.as_list()[i][1]) for i in range(len(rules.as_list()))]
                top_importance_boundary = np.quantile(exp_importances, .75)

                recall = len(set(exp_feature_indices) & set(true_features_indices))/len(true_features_indices)
                recalls.append(recall)
                # instance['precision'] = 
                # top_relevant_features = []
                # precision = 
                # for i,imp in rules.as_map()[1]:
                #     if imp < top_importance_boundary:
                #         break
                #     top_relevant_features.append((i,imp))
                # most relevant features (those where weights are in the top quartile) 

            elif (method == 'PyExplainer'):
                # FIXME to test fidelity of the explanation "features", do I use the features extracted from top important rules or filter individual features sorted according to importance values 
                
                # method 1 - use features extracted from top 10 rules
                top_rules = exp_instance['top_k_positive_rules'].head(10)
                top_features_from_rules = get_features_from_rules(top_rules)
                exp_feature_indices = [ i for i in range(len(features)) if features[i] in top_features_from_rules ]

                # method 2 - use indivdual feaatures and their assigned importance value (not top "features" in explanation)
                # exp_features = exp_instance['local_model'].get_rules().head(20).sort_values(by='importance',ascending=False)
                # exp_feature_indices = exp_features.index[:10]

                if (len(exp_feature_indices)<10): # set len of top features to min of exp features and top features
                    true_features_indices = true_features_indices[:,len(exp_feature_indices)]

                recall = len(set(exp_feature_indices) & set(true_features_indices))/len(true_features_indices)
                recalls.append(recall)

            elif (method == 'SHAP'):
                exp_feature_indices = pd.Series(exp_instance['shap_values'])
                exp_feature_indices = exp_feature_indices.sort_values(ascending=False).index

                if (len(exp_feature_indices)<10):
                    true_features_indices = true_features_indices[:,len(exp_feature_indices)]

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
        coefs: importance vals sorted according to original index
        indices: sorted indices according to importance(descending)
        """
        # original predicted class of the instance
        pred_class = model.predict(x.reshape(1,-1))[0].astype(int)
        
        #find indexs of coefficients in decreasing order of value
        # ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
        pred_probs = np.zeros(x.shape[0])
        for ind in np.nditer(np.array(indices)):
            x_copy = x.copy()
            x_copy[ind] = base[ind]
            x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
            pred_probs[ind] = x_copy_pr[0][pred_class]
            if (pred_probs[ind]==np.nan):
                print(x_copy_pr)
        print(pred_probs)
        pred_probs = pred_probs[pred_probs!=0]
        return -np.corrcoef(coefs, pred_probs)[0,1]

    result = []
    lime_faithfulness = []
    shap_faithfulness = []
    py_faithfulness = []
    
    for i in range(len(X_test)):
        # calculate for lime
        lime_exp = lime_explanations[i]['rule']
        sorted_indices = [np.abs(lime_exp.as_map()[1][i][0]) for i in range(len(lime_exp.as_map()[1]))] # sorted in order of decending importances
        coef_map = sorted(lime_exp.as_map()[1],key=itemgetter(0))
        x = X_test.iloc[i].values # data row type ndarray
        coefs = [np.abs(coef_map[i][1]) for i in range(len(coef_map))] # coefs for top 10 features (sorted by index)
        base = np.zeros(x.shape[0]) 
        fmlime = faithfulness_score(global_model,x,coefs,base,sorted_indices)
        lime_faithfulness.append(fmlime)

        # calculate for shap
        coefs_shap = np.array(shap_explanations[i]['shap_values'])
        count = np.count_nonzero(coefs_shap)
        sorted_indices = np.argsort(-coefs_shap)[:count]
        fmshap = faithfulness_score(global_model,x,coefs_shap,base,sorted_indices)
        shap_faithfulness.append(fmshap)

        # calculate for pyExplainer
        pred_probs = []
        pred_class = global_model.predict(x.reshape(1,-1))[0].astype(int)
        top_rules = py_explanations[i]['top_k_positive_rules'].head(10)
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
            if (pred_probs[ind]==np.nan):
                print(x_copy_pr)
        print(pred_probs)
        fmpy = -np.corrcoef(coefs_pyexp, pred_probs)[0,1]
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
        pred_probs = np.zeros(x.shape[0])
        for ind in np.nditer(ar):
            x_copy[ind] = x[ind]
            x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
            pred_probs[ind] = x_copy_pr[0][pred_class]

        pred_probs = pred_probs[pred_probs!=0]
        return np.all(np.diff(pred_probs[ar]) >= 0)

    result = []
    lime_monotonicity = []
    shap_monotonicity = []
    py_monotonicity = []

    for i in range(len(X_test)):
        # calculate for lime
        lime_exp = lime_explanations[i]['rule']
        sorted_indices = [np.abs(lime_exp.as_map()[1][i][0]) for i in range(len(lime_exp.as_map()[1]))] # sorted in order of decending importances
        coef_map = sorted(lime_exp.as_map()[1],key=itemgetter(0))
        x = X_test.iloc[i].values # data row type ndarray
        coefs = [np.abs(coef_map[i][1]) for i in range(len(coef_map))] # coefs for top 10 features (sorted by index)
        base = np.zeros(x.shape[0]) 
        mlime = monotonicity_score(global_model,x,coefs,base,sorted_indices)
        lime_monotonicity.append(mlime)

        # calculate for shap
        coefs_shap = np.array(shap_explanations[i]['shap_values'])
        count = np.count_nonzero(coefs_shap)
        sorted_indices = np.argsort(-coefs_shap)[:count]
        mshap = monotonicity_score(global_model,x,coefs_shap,base,sorted_indices)
        shap_monotonicity.append(mshap)

        # calculate for pyExplainer
        x_copy = base.copy()
        pred_probs = []
        pred_class = global_model.predict(x.reshape(1,-1))[0].astype(int)
        top_rules = py_explanations[i]['top_k_positive_rules'].head(10)
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


def uniqueness(X_test, lime_explanations, py_explanations, shap_explanations):
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

        py_exp = py_explanations[i]['top_k_positive_rules']['rule'][0].strip()
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

    result.append({ 'method':'LIME','lime_uniqueness': lime_uniqueness})
    result.append({ 'method':'SHAP','shap_uniqueness': shap_uniqueness})
    result.append({ 'method':'pyExplainer','py_uniqueness': py_uniqueness})

    return result

def similarity():
    pass















# ibm aix360 metrics - faithfulness and monotonicity
def faithfulness(model, x, coefs, base):
    #find predicted class
    pred_class = model.predict(x.reshape(1,-1))[0].astype(int)
    
    #find indexs of coefficients in decreasing order of value
    ar = np.argsort(-coefs)  #argsort returns indexes of values sorted in increasing order; so do it for negated array
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        x_copy = x.copy()
        x_copy[ind] = base[ind]
        x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]
        if (pred_probs[ind]==np.nan):
            print(x_copy_pr)
    print(pred_probs)
    return -np.corrcoef(coefs, pred_probs)[0,1]
    
def monotonicity(model, x, coefs, base):
    #find predicted class
    pred_class = model.predict(x.reshape(1,-1))[0].astype(int)

    x_copy = base.copy()

    #find indexs of coefficients in increasing order of value
    ar = np.argsort(coefs)
    pred_probs = np.zeros(x.shape[0])
    for ind in np.nditer(ar):
        x_copy[ind] = x[ind]
        x_copy_pr = model.predict_proba(x_copy.reshape(1,-1))
        pred_probs[ind] = x_copy_pr[0][pred_class]

    return np.all(np.diff(pred_probs[ar]) >= 0)

# calculate average percentage difference between prediction probabilities of global model and local model
def avg_proba_diff(global_pred_proba, local_pred_proba):
    # print('avg ', np.sum(local_pred_proba)/len(local_pred_proba))
    if len(global_pred_proba)!= len(local_pred_proba):
        print('unmatched length')
        return

    percentage_diffs = [(abs(global_pred_proba[i]-local_pred_proba[i])) for i in range(0, len(global_pred_proba))]
    
    return np.sum(percentage_diffs)/len(global_pred_proba)   

def faithfulness_and_monotonicity(test_data_x,lime_explanations, shap_explanations):
    lime_faithfulness = []
    lime_monotonicity = []
    shap_faithfulness = []
    shap_monotonicity = []
    for i in range(len(test_data_x)):
        local_preds = list(lime_explanations[i]['rule'].local_pred.values())[0][0]
        name = lime_explanations[i]['name']
        lime_exp = lime_explanations[i]['rule']
        x = test_data_x.iloc[i].values # data row type ndarray
        coefs = np.zeros(x.shape[0])  # coefficients (weights) corresponding to attribute importance
        # pred_class = int(np.round(global_preds[i]))
        for v in lime_exp.local_exp[1]:
            coefs[v[0]] = v[1]
        base = np.zeros(x.shape[0])
        fmlime = faithfulness(global_model,x,coefs,base)
        lime_faithfulness.append(fmlime)
        mlime = monotonicity(global_model,x,coefs,base)
        lime_monotonicity.append(mlime)

        coefs_shap = shap_explanations[i]['shap_values']
        fmshap = faithfulness(global_model,x,coefs_shap,base)
        shap_faithfulness.append(fmshap)
        mshap = monotonicity(global_model,x,coefs_shap,base)
        shap_monotonicity.append(mshap)
        # print('Lime local prediction ' , local_preds)
        # print('Global model prediction', lime_explanations[0]['rule'].predict_proba[1]) #global model prediction
    lime_avg_faithfulness = sum(lime_faithfulness)/len(lime_faithfulness)
    lime_percentage_monotonicity = sum(lime_monotonicity)/len(lime_monotonicity)
    shap_avg_faithfulness = sum(shap_faithfulness)/len(shap_faithfulness)
    shap_percentage_monotonicity = sum(shap_monotonicity)/len(shap_monotonicity)
    print("lime_avg_faithfulness: ",lime_avg_faithfulness)
    print("lime_percentage_monotonicity: " ,lime_percentage_monotonicity)
    print("shap_avg_faithfulness: ", shap_avg_faithfulness)
    print("shap_percentage_monotonicity: ", shap_percentage_monotonicity)
    return lime_faithfulness,lime_monotonicity,shap_faithfulness,shap_monotonicity

def get_percent_unique_explanation(explanation_list):
    total_exp = len(explanation_list)
    total_unique_exp = len(set(explanation_list))
    percent_unique = (total_unique_exp/total_exp)*100

    count_exp = Counter(explanation_list)
    max_exp_count = max(list(count_exp.values()))
    percent_dup_explanation = (max_exp_count/total_exp)*100

    print('% unique explanation is',round(percent_unique,2))
    print('% duplicate explanation is', round(percent_dup_explanation))
