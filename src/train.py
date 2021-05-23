import os
import yaml
import pandas as pd
import dill

import utils

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = []

    def fit(self, X, y=None):
        self.columns = [col for col in pd.get_dummies(
            X, prefix=self.key).columns]
        return self

    def transform(self, X):
        X = pd.get_dummies(X, prefix=self.key)
        test_columns = [col for col in X.columns]
        for col_ in self.columns:
            if col_ not in test_columns:
                X[col_] = 0
        return X[self.columns]


class Train():

    def __init__(self, data: str, features: str, params: dict) -> None:
        self.data = data
        self.features = features
        self.params = params
        self.pipeline = None
        self.best_params = None
        self.features_all = None

        os.makedirs("train", exist_ok=True)
        self.file_out_pipeline = os.path.join('train', 'pipeline.dill')
        self.file_out_report = os.path.join('train', 'report.md')
        
        super().__init__()

    def fit(self) -> None:
        final_transformers = list()

        for cat_col in self.features['categorical']:
            cat_transformer = Pipeline([
                ('selector', FeatureSelector(column=cat_col)),
                ('ohe', OHEEncoder(key=cat_col))
            ])
            final_transformers.append((cat_col, cat_transformer))

        for cont_col in self.features['continuous']:
            cont_transformer = Pipeline([
                ('selector', NumberSelector(key=cont_col)),
                ('standard', StandardScaler())
            ])
            final_transformers.append((cont_col, cont_transformer))

        feats = FeatureUnion(final_transformers)

        self.features_all = self.features['categorical'] + self.features['continuous']
        X = self.data[self.features_all]
        y = self.data[self.features['target']]

        rs = RandomizedSearchCV(
            Pipeline([
                ('features', feats),
                ('classifier', GradientBoostingClassifier(
                    **self.params['param_model'])),
            ]),
            {'classifier__' + key: value for key,
                value in self.params['param_distributions'].items()},
            **self.params['param_randomized_search'],
        )
        rs.fit(X, y)

        self.best_params = {key.split('__')[1]: value for key, value in rs.best_params_.items()}

        self.pipeline = Pipeline([
            ('features', feats),
            ('gb_clf', GradientBoostingClassifier(
                **self.params['param_model'],
                **self.best_params,
            )),
        ])
        self.pipeline.fit(X, y)
        del X, y

        return

    def make_report(self) -> None:
        score = cross_validate(
            self.pipeline,
            self.data[self.features_all],
            self.data[self.features['target']],
            **self.params['param_cross_validate']
        )
        
        best_params = pd.DataFrame.from_dict(
            self.best_params,
            orient='index'
        )
        
        feature_importances = pd.DataFrame(
            zip(self.features_all, self.pipeline.named_steps['gb_clf'].feature_importances_),
            columns=['feature', 'importance']
        )
        feature_importances.sort_values(
            by='importance',
            ascending=False,
            inplace=True
        )

        report = f"# Report\n\n" \
                 f"## Metrics\n\n" \
                 f"* roc_auc: {np.mean(score['test_roc_auc'])}\n" \
                 f"* f1: {np.mean(score['test_f1'])}\n" \
                 f"* precision: {np.mean(score['test_precision'])}\n" \
                 f"* recall: {np.mean(score['test_recall'])}\n\n" \
                 f"## Best parameters\n\n" \
                 f"{best_params.to_markdown(headers=['param', 'value'], tablefmt='github')}\n\n" \
                 f"## Feature importances\n\n" \
                 f"{feature_importances.to_markdown(tablefmt='github')}\n"

        utils.save_text_data(report, self.file_out_report)

        return

    def run(self) -> None:
        self.fit()
        self.make_report()

        with open(self.file_out_pipeline, 'wb') as file:
            dill.dump(self.pipeline, file)

        return


if __name__ == '__main__':
    params = yaml.safe_load(open(os.path.join('src', 'params.yaml')))
    df = pd.read_csv(
        params['data'],
    )

    train = Train(
        data=df,
        features=params['features'],
        params=params['train'],
    )
    train.run()
