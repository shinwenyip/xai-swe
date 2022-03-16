from sklearn import metrics
from aix360.metrics import *
from shap import KernelExplainer
from lime.lime_tabular import LimeTabularExplainer
from pyexplainer.pyexplainer_pyexplainer import *


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
            explanation = explainer.explain(X_explain, y_explain, search_function ='CrossoverInterpolation')
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

# Below are functions for evaluation metrics

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
