'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot


# Call functions / instanciate objects from the .py files
def main():
    '''
    Calls the associated functions from parts 1-5 of the problem set.

    Parameters:
        None
    Returns:
        None
    '''
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    #part1_etl.etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests=part2_preprocessing.transform()
    df_arrests=part2_preprocessing.repeat_felons(df_arrests)
    df_arrests=part2_preprocessing.make_features(df_arrests)

    # PART 3: Call functions/instanciate objects from logistic_regression
    arrests_train, arrests_test, probs_lr=part3_logistic_regression.do_log_reg(df_arrests)
    #print(arrests_pred.head(50))

    # PART 4: Call functions/instanciate objects from decision_tree
    full_train, full_test, probs_dt=part4_decision_tree.do_decision_tree(arrests_train, arrests_test)

    # PART 5: Call functions/instanciate objects from calibration_plot
    part5_calibration_plot.do_plots(arrests_test['charge_degree'], y_prob=probs_lr)
    part5_calibration_plot.do_plots(arrests_test['charge_degree'], y_prob=probs_dt)



if __name__ == "__main__":
    main()