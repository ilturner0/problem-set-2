'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

def do_decision_tree(train, test):
    '''
    Performs decision tree classification

    Parameters:
          train : dataframe
               The portion of the dataset used for model training
          test : dataframe
               The portion of the dataset used for model testing 
    
    Returns:
          train : dataframe
               The training portion of df_arrests.
          test : dataframe
               The test dataframe that the regression model performed predictions with.
          prob_felony_charge_dt : Array
               The probability of each charge being a felony as determined by the model.


    '''
    param_grid_dt={'max_depth':[1, 10, 20]}
    dt_model=DTC()
    gs_cv_dt=GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5)
    gs_cv_dt.fit(train[['num_fel_arrests_last_year', 'current_charge_felony']], train['y'])
    print(f"Best depth is: {gs_cv_dt.best_params_}\n The amount of regularization is in the middle of the 3 values.")

    test['pred_dt']=gs_cv_dt.predict(test[['num_fel_arrests_last_year', 'current_charge_felony']])
    probs_dt=gs_cv_dt.predict_proba(test[['num_fel_arrests_last_year', 'current_charge_felony']])
    prob_felony_charge_dt=probs_dt[:, 1]
    #print(gs_cv_dt.classes_)
    print(gs_cv_dt.best_score_)
    train.to_csv(r'data/df_arrests_train.csv')
    test.to_csv(r'data/df_arrests_tested.csv')
    return train, test, prob_felony_charge_dt