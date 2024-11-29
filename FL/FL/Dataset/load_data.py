import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler


class LoadDataset:
    def load_dataset(preferences):
        if preferences.dataset == "adult":
            _X, _Z, _y = LoadDataset.load_adult(preferences.dataset_path)
            return _X, _Z, _y
        if preferences.dataset == "dutch":
            _X, _Z, _y = LoadDataset.load_dutch(preferences.dataset_path)
            return _X, _Z, _y
        else:
            raise ValueError(f"Dataset {preferences.dataset} not supported.")

    def load_adult(dataset_path):
        adult_feat_cols = [
            "workclass",
            "education",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "sex_binary",
            "race_binary",
            "age_binary",
        ]

        adult_columns_names = (
            "age",
            "workclass",  # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
            "fnlwgt",  # "weight" of that person in the dataset (i.e. how many people does that person represent) -> https://www.kansascityfed.org/research/datamuseum/cps/coreinfo/keyconcepts/weights
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        )
        df_adult = pd.read_csv(dataset_path + "adult.data", names=adult_columns_names)
        # df_adult = pd.DataFrame(df_adult[0]).astype("int32")

        df_adult["sex_binary"] = np.where(df_adult["sex"] == " Male", 1, 0)
        df_adult["race_binary"] = np.where(df_adult["race"] == " White", 1, 0)
        df_adult["age_binary"] = np.where(
            (df_adult["age"] > 25) & (df_adult["age"] < 60), 1, 0
        )

        y = np.zeros(len(df_adult))

        y[df_adult["income"] == " >50K"] = 1
        df_adult["income_binary"] = y
        del df_adult["income"]
        del df_adult["race"]
        del df_adult["sex"]
        del df_adult["age"]

        metadata_adult = {
            "name": "Adult",
            "protected_atts": ["sex_binary", "race_binary", "age_binary"],
            "protected_att_values": [0, 0, 0],
            "code": ["AD1", "AD2", "AD3"],
            "protected_att_descriptions": [
                "Gender = Female",
                "Race \\neq white",
                "Age <25 | >60",
            ],
            "target_variable": "income_binary",
        }

        return LoadDataset.dataset_to_numpy(
            df_adult,
            adult_feat_cols,
            metadata_adult,
            num_sensitive_features=1,
        )

    def load_dutch(dataset_path):
        dutch_df = pd.read_csv(dataset_path + "dutch_census.csv")
        # dutch_df = pd.DataFrame(data[0]).astype("int32")

        dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
        dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

        del dutch_df["sex"]
        del dutch_df["occupation"]

        dutch_df_feature_columns = [
            "age",
            "household_position",
            "household_size",
            "prev_residence_place",
            "citizenship",
            "country_birth",
            "edu_level",
            "economic_status",
            "cur_eco_activity",
            "Marital_status",
            "sex_binary",
        ]

        metadata_dutch = {
            "name": "Dutch census",
            "code": ["DU1"],
            "protected_atts": ["sex_binary"],
            "protected_att_values": [0],
            "protected_att_descriptions": ["Gender = Female"],
            "target_variable": "occupation_binary",
        }

        return LoadDataset.dataset_to_numpy(
            dutch_df,
            dutch_df_feature_columns,
            metadata_dutch,
            num_sensitive_features=1,
        )

    def dataset_to_numpy(
        _df,
        _feature_cols: list,
        _metadata: dict,
        num_sensitive_features: int = 1,
        sensitive_features_last: bool = True,
    ):
        """Args:
        _df: pandas dataframe
        _feature_cols: list of feature column names
        _metadata: dictionary with metadata
        num_sensitive_features: number of sensitive features to use
        sensitive_features_last: if True, then sensitive features are encoded as last columns
        """

        # transform features to 1-hot
        _X = _df[_feature_cols]
        # take sensitive features separately
        print(
            f'Using {_metadata["protected_atts"][:num_sensitive_features]} as sensitive feature(s).'
        )
        if num_sensitive_features > len(_metadata["protected_atts"]):
            num_sensitive_features = len(_metadata["protected_atts"])
        _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]
        # _X = _X.drop(columns=_metadata["protected_atts"][:num_sensitive_features])
        # 1-hot encode and scale features
        if "dummy_cols" in _metadata.keys():
            dummy_cols = _metadata["dummy_cols"]
        else:
            dummy_cols = None
        _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
        esc = MinMaxScaler()
        _X = esc.fit_transform(_X2)

        # current implementation assumes each sensitive feature is binary
        for i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
            assert len(_Z[tmp].unique()) == 2, "Sensitive feature is not binary!"

        # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
        _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
        # print(_Z2.head(), _Z2.shape)
        if sensitive_features_last:
            for i, tmp in enumerate(_Z.columns):
                assert (
                    _metadata["protected_att_values"][i] in _Z[tmp].unique()
                ), "Protected attribute value not found in data!"
                if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                    # swap columns
                    _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
        # change booleans to floats
        # _Z2 = _Z2.astype(float)
        # _Z = _Z2.to_numpy()
        _y = _df[_metadata["target_variable"]].values
        return _X, np.array([sv[0] for sv in _Z.values]), _y
