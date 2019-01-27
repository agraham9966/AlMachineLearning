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
 
strat_train_set, strat_test_set = prep_data(housingdf) 
housing_ts = strat_train_set.copy()

#######plot to visualize population vs. median house prices 
# housing_ts.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
# s=housing_ts['population']/100, label='population', figsize=(10,7),
# c='median_house_value', cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show() 
####################################
#look for correlations - compute pearson's r 
# corr_matrix= housing_ts.corr() 
# print(corr_matrix['median_house_value'].sort_values(ascending=False))
# attributes= ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
# scatter_matrix(housing_ts[attributes], figsize=(12, 8))
# plt.show() #median income looks to be best correlated with median house value 
# housing_ts.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1) 
# plt.show()

####################feature engineering 
# housing_ts['rooms_per_household'] = housing_ts['total_rooms']/housing_ts['households']
# housing_ts['bedrooms_per_room'] = housing_ts['total_bedrooms']/housing_ts['total_rooms']
# housing_ts['population_per_household'] = housing_ts['population']/housing_ts['households']
# corr_matrix= housing_ts.corr() 
# print(corr_matrix['median_house_value'].sort_values(ascending=False)) ##got some better features to work with here 

# #split into target and labels 
# housing_ts=strat_train_set.drop('median_house_value', axis=1) 
# housing_labels =strat_train_set['median_house_value'].copy() 

##fill missing values in some of the attributes - let's fill it with the median from each attribute 
# imputer = Imputer(strategy='median') 
# housing_num= housing_ts.drop('ocean_proximity', axis=1) # can't compute mean of cat values duh! 
# imputer.fit(housing_num) #computes median of each attribute and stored res. in statistics_ instance variable. 
# #print(imputer.statistics_) 
# x = imputer.transform(housing_num) 
# housing_trans=pd.DataFrame(x, columns=housing_num.columns) #bring it back to a dataframe since imputer.trans outputs np array 
# housing_cat = housing_ts['ocean_proximity'] 
# housing_cat_encoded, housing_categories = housing_cat.factorize() # convert ocean categories to numerical 
# ##hotencoding 
# encoder = OneHotEncoder() 
# housing_cat_1hot= encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# ###############should I combine it back with the housing_trans? 


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6 #indices of columns in df 

class CombinedAttributesAdder(BaseEstimator, TransformerMixin): 
    def __init__(self, add_bedrooms_per_room = True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None): 
        return self 
    def transform(self, X, y=None): 
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]	
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room: 
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room] 
        else: 
            return np.c_[X, rooms_per_household, population_per_household] 

attr_adder = CombinedAttributesAdder() 
housing_extra_attribs = attr_adder.transform(housing_ts.values) 
print(housing_extra_attribs) 
num_pipeline = Pipeline([
    ('imputer', Imputer(strategy='median')), 
    ('attribs_adder', CombinedAttributesAdder()), 
    ('std_scalar', StandardScaler()), 
]) 

# housing_num_tr = num_pipeline.fit_transform(housing_num) 
# print(housing_num_tr) 
