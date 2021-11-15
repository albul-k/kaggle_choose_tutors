"""Train pipeline
"""

import os
import yaml
import dill
import pandas as pd
import numpy as np

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

    # pylint: disable=invalid-name
    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        """Fit
        """
        return self

    def transform(self, data):
        """Transform
        """

        return data[self.column]


# pylint: disable=too-many-instance-attributes
class Train():
    """Train class
    """

    pipeline = None
    best_params = None
    features_all = None

    def __init__(self, data: str, features: str, params: dict) -> None:
        self.data = data
        self.features = features
        self.params = params

        os.makedirs("model", exist_ok=True)
        self.file_out_pipeline = os.path.join('model', 'pipeline.dill')
        self.file_out_report = os.path.join('model', 'report.md')

    def fit(self) -> None:
        """Fit model
        """

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

    def make_report(self) -> None:
        """Generate report
        """

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
                 f"* roc_auc: {np.mean(score['test_roc_auc'])}\n" \
                 f"* f1: {np.mean(score['test_f1'])}\n" \
                 f"* precision: {np.mean(score['test_precision'])}\n" \
                 f"* recall: {np.mean(score['test_recall'])}\n\n" \
                 f"## Best parameters\n\n" \
                 f"{best_params.to_markdown(headers=['param', 'value'], tablefmt='github')}\n\n" \
                 f"## Feature importances\n\n" \
                 f"{feature_importances.to_markdown(tablefmt='github')}\n"

        with open(self.file_out_report, 'w', encoding='utf-8') as file:
            file.write(report)

    def run(self) -> None:
        """Run train
        """

        self.fit()
        self.make_report()

        with open(self.file_out_pipeline, 'wb') as file:
            dill.dump(self.pipeline, file)


def main():
    """Main
    """

    with open('params.yaml', 'r', encoding='utf-8') as file:
        params = yaml.safe_load(file)

    train = Train(
        data=pd.read_csv(
            params['data'],
        ),
        features=params['features'],
        params=params['train'],
    )
    train.run()


if __name__ == '__main__':
    main()
