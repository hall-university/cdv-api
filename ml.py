import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Modeler:
    def __init__(
        self,
        data_source: str = None,
        classifier=None
    ):
        self.data_source = data_source
        self.classifier = classifier
        self.classification_report = None
        self._df = None

    @property
    def df(self):
        return self._df

    def create_dataset(self) -> pd.DataFrame:
        """
        Read csv data and create DataFrame object.
        """
        return pd.read_csv(self.data_source)

    def na_values_present(self, df: pd.DataFrame) -> bool:
        return df.isna().values.any()

    def replace_na_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO I think there is no point in checking if na values are in the dataset.
        return df.fillna(df.mean()) if self.na_values_present(df) else df

    def replace_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df, drop_first=True)

    def create_binary_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        df['binary_quality'] = [0 if x < 7 else 1 for x in df.quality]
        return df

    def split_dependent_independent(self, df: pd.DataFrame) -> tuple:
        dependent = df.drop(['quality', 'binary_quality'], axis=1)
        independent = df['binary_quality']
        return dependent, independent

    def standardize(self, data):
        return StandardScaler().fit_transform(data)

    def train_test_split(self, dependent, independent) -> tuple:
        return train_test_split(dependent, independent, test_size=.2)

    def fit(self, dependent_train, independent_train):
        self.classifier.fit(dependent_train, independent_train)

    def predict(self, dependent):
        independent = self.classifier.predict(dependent)
        return independent

    def model_summary(self, independent_test, independent_predicted):
        self.classification_report = classification_report(
            independent_test, independent_predicted
        )

    def serialize(self, file_path: str = None):
        file_path = file_path or self.classifier.__class__.__name__
        pickle.dump(self.classifier, open(file_path, 'wb'))

    @classmethod
    def deserialize(cls, file_path):
        with open(file_path) as f:
            classifier = pickle.load(f)
        return cls(classifier=classifier)

    def pipeline(self):
        self._df = df = self.create_binary_quality(
            self.replace_dummies(self.replace_na_values(self.create_dataset()))
        )
        x, y = self.split_dependent_independent(df)
        x = self.standardize(x)
        x_train, x_test, y_train, y_test = self.train_test_split(x, y)
        self.fit(x_train, y_train)
        y_predict = self.predict(x_test)
        self.model_summary(y_test, y_predict)


model = Modeler(data_source='winequalityN.csv', classifier=RandomForestClassifier())
model.pipeline()
print(model.classification_report)
model.serialize()
