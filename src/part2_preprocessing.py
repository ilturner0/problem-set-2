'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import numpy as np




# Your code here

def transform():
    '''
    Reads in the pred_universe_raw and arrest_events_raw datasets and merges them on person_id.

    Parameters:
        None
    
    Returns:
        df_arrests : Dataframe resulting from merging the pred_universe_raw and arrest_events_raw datasets.
    '''
    pred_universe_raw=pd.read_csv(r'data/pred_universe_raw.csv')
    arrest_events_raw=pd.read_csv(r'data/arrest_events_raw.csv')
    df_arrests=arrest_events_raw.merge(pred_universe_raw, on='person_id', how='outer')
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
    print(df_arrests.head())
    return df_arrests


def t1():
    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)
    df_arrests=arrest_events_raw.merge(pred_universe_raw, on='person_id', how='outer')
    return df_arrests

    '''
    df_arrests['y']=0
    for person in df_arrests.groupby('person_id'):
    
    '''


def repeat_felons(df_arrests):
    '''
    Creates the column 'y' in df_arrests and calculates whether or not a person was arrested for a felony within 365 days of a prior arrest. Returns the dataframe with the added column.

    Parameters:
        df_arrests : dataframe

    Returns:
        df_arrests : dataframe
    
    '''

    df_arrests['date_diff']=df_arrests['arrest_date_event'].diff().dt.days
    df_arrests['date_diff']=df_arrests['date_diff'].fillna(9999999)
    df_arrests.sort_values(by='person_id', inplace=True)
    df_arrests['id_diff']=df_arrests['person_id'].diff()
    df_arrests['y']=df_arrests.apply(lambda x: 1 if ((abs(x['date_diff'])<365)&(x['charge_degree']=='felony')&(x['id_diff']==0)) else 0, axis=1)
    felons=len(df_arrests.loc[df_arrests['y']==1]['person_id'].unique()) #How many arrestees were arrested for a felony within 365 days of a prior arrest.
    total_arrestees=len(df_arrests['person_id'].unique()) #The number of unique arrestees in the dataset.
    percentage=np.round((felons/total_arrestees)*100, 2)
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?\n Answer: {percentage}")
    return df_arrests

def make_features(df_arrests):
    '''
    Creates the predictive features 'current_charge_felony' and 'num_fel_arrests_last_year'

    Parameters:
        df_arrests : dataframe 

    Returns:
        df_arrests : dataframe containing the added features
    
    '''

    #Make feature 1: 'current_charge_felony'
    current_charges=df_arrests.sort_values(by='arrest_date_event', ascending=False).drop_duplicates(subset='person_id')
    current_charges['current_charge_felony']=current_charges.apply(lambda x: 1 if x['charge_degree']=='felony' else 0, axis=1)
    current_fels=current_charges['current_charge_felony'].sum()
    total_current_charges=len(current_charges)
    percentage=np.round((current_fels/total_current_charges)*100, 2)
    df_arrests=df_arrests.merge(current_charges['current_charge_felony'], left_index=True, right_index=True, how='outer')
    #return df_arrests, current_charges
    df_arrests['current_charge_felony']=df_arrests['current_charge_felony'].fillna(0)
    print(f"What share of current charges are felonies?\n Answer: {percentage}")

    #Make feature 2: 'num_fel_arrests_last_year'

    df_arrests['num_fel_arrests_last_year']=0
    for person in df_arrests.groupby(['person_id']):
        id=person[0][0]
        current_charge=person[1]['arrest_date_event'].max()
        person[1]['num_fel_arrests_last_year']=person[1].apply(lambda x: 1 if((x['charge_degree']=='felony')&(abs((current_charge-x['arrest_date_event']).days)<=365)) else 0, axis=1)
        person[1].drop(person[1].loc[person[1]['arrest_date_event']==current_charge].index, inplace=True)
        num_fels=person[1]['num_fel_arrests_last_year'].sum()
        df_arrests.loc[df_arrests['person_id']==id, 'num_fel_arrests_last_year']=num_fels
        mean_fels_last_year=df_arrests.drop_duplicates('person_id')['num_fel_arrests_last_year'].mean() # The average number of felony arrests a person has within a year of their most recent charge. Dropped duplicate IDs to not sway average.
    print(f"What is the average number of felony arrests in the last year?\n Answer: {mean_fels_last_year}")
    print(mean_fels_last_year)
    
    return df_arrests