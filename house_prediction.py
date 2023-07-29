#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import os
import warnings
import sys

import tarfile
import urllib

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


# ## Loading the data

# In[ ]:


DOWNLOAD_ROOT ="https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# In[ ]:


def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[ ]:


fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH)


# In[ ]:


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[ ]:


housing = load_housing_data(housing_path=HOUSING_PATH)


# In[ ]:


housing.head()


# ## Setting Mlflow server

# In[ ]:


remote_server_uri = "http://localhost:5000" # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)


# In[ ]:


mlflow.tracking.get_tracking_uri()


# In[ ]:


exp_name = "ElasticNet_house"
mlflow.set_experiment(exp_name)



from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household,population_per_household]

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])


def eval_metrics(actual, pred):
    # compute relevant metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(alpha=0.5, l1_ratio=0.5):
    # train a model with given parameters
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    housing = load_housing_data(housing_path=HOUSING_PATH)
    train_set, test_set = train_test_split(housing, test_size=0.2,random_state=42)

    with mlflow.start_run(run_name='Main') as parent_run:
        mlflow.log_param("Main", "yes")
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

        with mlflow.start_run(run_name='Data_Preparation', nested=True) as child_run:
            mlflow.log_param("Data Preparayion", "yes")

            housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])

            for train_index, test_index in split.split(housing,housing["income_cat"]):
                strat_train_set = housing.loc[train_index]
                strat_test_set = housing.loc[test_index]

            for set_ in (strat_train_set, strat_test_set):
                set_.drop("income_cat", axis=1, inplace=True)

            housing = strat_train_set.copy()

            housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
            housing["bedrooms_per_room"] =housing["total_bedrooms"]/housing["total_rooms"]
            housing["population_per_household"]=housing["population"]/housing["households"]

            housing = strat_train_set.drop("median_house_value", axis=1)
            housing_labels = strat_train_set["median_house_value"].copy()

            median = housing["total_bedrooms"].median() # option 3
            housing["total_bedrooms"].fillna(median, inplace=True)


            imputer = SimpleImputer(strategy="median")

            housing_num = housing.drop("ocean_proximity", axis=1)

            imputer.fit(housing_num)

            X = imputer.transform(housing_num)

            housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)

            housing_cat = housing[["ocean_proximity"]]

            ordinal_encoder = OrdinalEncoder()
            housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

            cat_encoder = OneHotEncoder()
            housing_cat_1hot = cat_encoder.fit_transform(housing_cat)



            attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
            housing_extra_attribs = attr_adder.transform(housing.values)

            housing_num_tr = num_pipeline.fit_transform(housing_num)

            num_attribs = list(housing_num)
            cat_attribs = ["ocean_proximity"]

            full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),])

            housing_prepared = full_pipeline.fit_transform(housing)

        with mlflow.start_run(run_name='Model_Training', nested=True) as child_run:
            mlflow.log_param("Model Training", "yes")

            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(housing_prepared, housing_labels)

            predicted_qualities = lr.predict(housing_prepared)
            (rmse, mae, r2) = eval_metrics(housing_labels, predicted_qualities)

        with mlflow.start_run(run_name='Model_Performance', nested=True) as child_run:
            mlflow.log_param("Model Performance", "yes")
            print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            mlflow.log_param(key="alpha", value=alpha)
            mlflow.log_param(key="l1_ratio", value=l1_ratio)
            mlflow.log_metric(key="rmse", value=rmse)
            mlflow.log_metrics({"mae": mae, "r2": r2})
            mlflow.log_artifact('datasets')
            print("Save to: {}".format(mlflow.get_artifact_uri()))

            mlflow.sklearn.log_model(lr, "model")




train(0.5, 0.5)




train(0.2, 0.2)






