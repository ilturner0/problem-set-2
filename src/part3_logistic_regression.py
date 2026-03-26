'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr



# Your code here
def do_log_reg(df_arrests):
    '''
    Performs logistic regression using gridsearch with 5 fold crossvalidation.

    Parameters:
        df_arrests : dataframe
    
    Returns:
        df_arrests_train : dataframe
            The training portion of df_arrests.
        df_arrests_test : dataframe
            The test dataframe that the regression model performed predictions with.
        prob_felony_charge_lr : Array
            The probability of each charge being a felony as determined by the model.
    '''

    df_arrests_train, df_arrests_test=train_test_split(df_arrests, test_size=0.3, shuffle=True, stratify=df_arrests['y'])
    features=['num_fel_arrests_last_year', 'current_charge_felony']
    param_grid={"C":[1, 10, 20]}
    lr_model=lr()
    gs_cv=GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5)
    #gs_cv.fit(df_arrests_train[['num_fel_arrests_last_year', 'current_charge_felony']], df_arrests_train['y'])
    gs_cv.fit(df_arrests_train[features], df_arrests_train['y'])

    print(f"Optimal C value is: {gs_cv.best_params_}\n It has the most regularization of all C values passed.\n Accuracy is {gs_cv.best_score_}")
    df_arrests_test['pred_lr']=gs_cv.predict(df_arrests_test[features])
    probs_lr=gs_cv.predict_proba(df_arrests_test[features])
    #print(type(probs_lr[0]))
    #print(probs_lr[:,0])
    prob_felony_charge_lr=probs_lr[:, 1]

    print(f"Classes {gs_cv.best_estimator_.classes_}")
    return df_arrests_train, df_arrests_test, prob_felony_charge_lr
    



