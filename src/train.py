import os
import yaml
import dill
import pandas
import numpy

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


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
        feats = FeatureUnion([
            ('numeric', make_pipeline(FeatureSelector(column=self.features['continuous']), StandardScaler())),
            ('categorical', make_pipeline(FeatureSelector(column=self.features['categorical']), OneHotEncoder(handle_unknown='ignore')))
        ])

        self.features_all = self.features['categorical'] + \
            self.features['continuous']
        X = self.data[self.features_all]
        y = self.data[self.features['target']]

        rs = RandomizedSearchCV(
            Pipeline(
                steps=[
                    ('features', feats),
                    ('classifier', GradientBoostingClassifier(
                        **self.params['param_model'])),
                ]),
            {'classifier__' + key: value for key,
                value in self.params['param_distributions'].items()},
            **self.params['param_randomized_search'],
        )
        rs.fit(X, y)

        self.best_params={key.split('__')[1]: value for key, value in rs.best_params_.items()}

        self.pipeline=Pipeline(
            steps=[
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
        score=cross_validate(
            self.pipeline,
            self.data[self.features_all],
            self.data[self.features['target']],
            **self.params['param_cross_validate']
        )

        best_params=pandas.DataFrame.from_dict(
            self.best_params,
            orient='index'
        )

        feature_importances=pandas.DataFrame(
            zip(self.features_all,
                self.pipeline.named_steps['gb_clf'].feature_importances_),
            columns=['feature', 'importance']
        )
        feature_importances.sort_values(
            by='importance',
            ascending=False,
            inplace=True
        )

        report=  f"# Report\n\n" \
                 f"## Metrics\n\n" \
                 f"* roc_auc: {numpy.mean(score['test_roc_auc'])}\n" \
                 f"* f1: {numpy.mean(score['test_f1'])}\n" \
                 f"* precision: {numpy.mean(score['test_precision'])}\n" \
                 f"* recall: {numpy.mean(score['test_recall'])}\n\n" \
                 f"## Best parameters\n\n" \
                 f"{best_params.to_markdown(headers=['param', 'value'], tablefmt='github')}\n\n" \
                 f"## Feature importances\n\n" \
                 f"{feature_importances.to_markdown(tablefmt='github')}\n"

        with open(self.file_out_report, 'w') as file:
            file.write(report)

        return

    def run(self) -> None:
        self.fit()
        self.make_report()

        with open(self.file_out_pipeline, 'wb') as file:
            dill.dump(self.pipeline, file)

        return


if __name__ == '__main__':
    params=yaml.safe_load(open('params.yaml'))
    df=pandas.read_csv(
        params['data'],
    )

    train=Train(
        data=df,
        features=params['features'],
        params=params['train'],
    )
    train.run()
