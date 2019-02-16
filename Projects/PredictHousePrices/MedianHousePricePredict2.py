import pandas as pd 
from fetchdata import fetch_housing_data, load_housing_data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin 
import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix 
import numpy as np 

fetch_housing_data() 
housingdf = load_housing_data()
#print(housingdf.head())
#print(housingdf.info())
#print(housingdf.head())
#print(housingdf.info())  
#print(housingdf['ocean_proximity'].value_counts())
# print(housingdf.describe()) #shows stats for each of the columns - null vals are ignored 
# print(housingdf.columns) 
# housingdf.hist(bins=50, figsize = (20,15))
# plt.show() 

##we need to do some stratified sampling in this dataset - since we are trying to train the model
##with a subset of all the data, we want train data that represents all categories of incomes 
	
def prep_data(housingdf): 
    
    housingdf['income_cat'] = np.ceil(housingdf['median_income']/1.5)
    housingdf['income_cat'].where(housingdf['income_cat'] < 5, 5.0, inplace=True) 
    # housingdf['income_cat'].hist() #shows proportions of new classes 
    # plt.show()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
    for train_index, test_index in split.split(housingdf, housingdf['income_cat']): 
        strat_train_set = housingdf.loc[train_index]
        strat_test_set = housingdf.loc[test_index]
    #print(strat_test_set['income_cat'].value_counts()/len(strat_test_set)) show proportions of stratified cats 
    for set_ in (strat_train_set, strat_test_set): 
        set_.drop('income_cat', axis=1, inplace=True) 
    
    return strat_train_set, strat_test_set

def plot_lat_long_housing(housing_df): 
    housing_df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, 
    s=housing_df['population']/100, label='population', figsize=(10,7), 
    c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
)
    plt.legend()
    plt.show()

def add_features(housing_df): 
    housing_df['rooms_per_household'] = housing_df['total_rooms'] / housing_df['households']
    housing_df['bedrooms_per_room'] = housing_df['total_bedrooms'] / housing_df['total_rooms']
    housing_df['population_per_household'] =housing_df['population'] / housing_df['households']

    return housing_df 

strat_train_set, strat_test_set = prep_data(housingdf) 

##set up training data with some training labels as well 
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy() 
##use imputer to set missing values to median 
imputer = Imputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1) #remove categorical vars 
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
##lets also remap the categorical values in ocean proximity 

oceanprox_map = {"ISLAND": 0, "NEAR BAY": 1, "NEAR OCEAN": 1, "<1H OCEAN": 2, "INLAND": 3} ##not using onehotencoder 
housing["ocean_proximity_num"] = housing.ocean_proximity.map(oceanprox_map)

print(len(housing_tr), len(housing))
#print(housing['ocean_proximity'].value_counts)
#final = housing_tr.append(housing['ocean_proximity_num'])
housing = housing.reset_index() ##important lesson - reset the goddam indices before merging cols between dataframes 
housing_tr['ocean_proximity_num'] = housing['ocean_proximity_num']
print(housing_tr.head())