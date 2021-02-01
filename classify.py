import argparse
import pandas as pd
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#parser
parser = argparse.ArgumentParser(description='Classifiy thrown grenades')
parser.add_argument('test', help = 'Test dataset to classify')
args = parser.parse_args()

#load data form parser
data = pd.read_csv(args.test)

#function to select specific columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
#numerical columns
num_cols = [
 'detonation_raw_x',
 'detonation_raw_y',
 'detonation_raw_z']

#categorical columns
cat_cols = ['TYPE','map_name']

#pipeline for numerical columns
num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(num_cols)),
        ("scale", StandardScaler())
    ])

#pipepline for categorical columns
cat_pipeline = Pipeline([
    ("select cat", DataFrameSelector(cat_cols)),
    ("One hot encoder", OneHotEncoder(sparse=False))
])

#Combine pipelines
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


X = preprocess_pipeline.fit_transform(data)

#classify new dataset
classifier = joblib.load('best_model.pkl')
labels = classifier.predict(X)
data['LABELS'] = labels

if __name__ == '__main__':
    data.to_csv(args.test, index = False)