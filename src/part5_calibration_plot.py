'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def do_plots(y_test, y_prob):
    '''
    Prepare data for and create calibration plots. Takes the classes in y_test and assigns binary values to them for use in calibration_plot

    Parameters:
        y_test : The target column ('charge_degree')
        y_prob : The probability that the charge is a felony
    '''
    binary=y_test.str.replace('felony', '1') 
    binary=binary.str.replace('misdemeanor', '0')
    binary=binary.astype('int')
    binary=binary.astype('bool')
    calibration_plot(binary, y_prob, n_bins=5)
