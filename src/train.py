"""Train pipeline
"""

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
    """Feature selector class

    Args:
        BaseEstimator (class): base estimator class
        TransformerMixin (class): class for transformers
    """
    def __init__(self, column):
        self.column = column

    def fit(self, *args, **kwargs):
        """Fit

        Returns:
            self:
        """
        return self

    def transform(self, data, *args, **kwargs):
        """Transform

        Args:
            data (pandas.DataFrame): some data

        Returns:
            pandas.DataFrame:
        """
        return data[self.column]


class Train():
    """Train class
    """

    # pylint: disable=too-many-instance-attributes

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
        numeric_pipeline = make_pipeline(
            FeatureSelector(column=self.features['continuous']),
            StandardScaler()
        )

        categorical_pipeline = make_pipeline(
            FeatureSelector(column=self.features['categorical']),
            OneHotEncoder(handle_unknown='ignore')
        )
        feats = FeatureUnion([
            ('numeric', numeric_pipeline),
            ('categorical', categorical_pipeline)
        ])

        self.features_all = self.features['categorical'] + \
            self.features['continuous']
        data = self.data[self.features_all]
        target = self.data[self.features['target']]

        random_search = RandomizedSearchCV(
            Pipeline(
                steps=[
                    ('features', feats),
                    ('classifier', GradientBoostingClassifier(
                        **self.params['param_model'])),
                ]),
            {'classifier__' + k: v for k, v in self.params['param_distributions'].items()},
            **self.params['param_randomized_search'],
        )
        random_search.fit(data, target)

        self.best_params = {k.split('__')[1]: v for k, v in random_search.best_params_.items()}
        self.pipeline = Pipeline(
            steps=[
                ('features', feats),
                ('gb_clf', GradientBoostingClassifier(
                    **self.params['param_model'],
                    **self.best_params,
                )),
            ]
        )
        self.pipeline.fit(data, target)
        del data, target

        return

    def make_report(self) -> None:
        score = cross_validate(
            self.pipeline,
            self.data[self.features_all],
            self.data[self.features['target']],
            **self.params['param_cross_validate']
        )

        best_params = pandas.DataFrame.from_dict(
            self.best_params,
            orient='index'
        )

        feature_importances = pandas.DataFrame(
            zip(self.features_all,
                self.pipeline.named_steps['gb_clf'].feature_importances_),
            columns=['feature', 'importance']
        )
        feature_importances.sort_values(
            by='importance',
            ascending=False,
            inplace=True
        )

        report = f"# Report\n\n" \
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
    params = yaml.safe_load(open('params.yaml'))
    train = Train(
        data=pandas.read_csv(
            params['data'],
        ),
        features=params['features'],
        params=params['train'],
    )
    train.run()
