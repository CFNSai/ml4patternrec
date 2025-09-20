import uproot as ur
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple

class GradientBoostHybrid:
    """
    A hybrid AI class model using uproot data and gradient boosting.
    Class combines numerical variables from a TTree and binned data from a TH1 histogram.
    """

    def __init__(
        self,
        inputfile_name: str,
        tree_name: str,
        branch_name: Tuple[str,str],
        histogram_name: str,
        features: List[str],
        target: str,
        x_edges: np.array = None,
        y_edges: np.array = None,
        r_max: float = 650,
        chunck_size: int = 100_000,
        max_events: int = 250_000 #safety cap for WSL2
    ):
        """
        Initializes the HybridModel.

        Args:
            inputfile_name (str): Path to the ROOT file.
            tree_name (str): Name of the TTree to read numerical variables from.
            histogram_name (str): Name of the TH2 histogram to use.
            features (list): A list of numerical branch names from the TTree.
            target (str): The name of the target variable branch from the TTree.
        """
        self._inputfile = inputfile_name
        self.tree_name = tree_name
        self.branch_name = branch_name
        self.histogram_name = histogram_name
        self.features = features
        self.target = target
        self.chunck_size = chunck_size
        self.r_max = r_max
        self.max_events = max_events
        #This is the hist range
        self.x_edges = np.linspace(-650,650,50) if x_edges is None else np.asarray(x_edges)
        self.y_edges = np.linspace(-650,650,50) if y_edges is None else np.asarray(y_edges)
        self.model = None
        self.scaler = None  # Add a scaler for potential feature scaling

    def _get_th2_bin_index(self,row: pd.Series) -> int:
        x_var,y_var = self.branch_name
        x_val = row[x_var]
        y_val = row[y_var]

        if self.r_max and (x_val**2 + y_val**2 > self.r_max**2):
            return -1

        x_bin = np.digitize(x_val, self.x_edges) - 1
        y_bin = np.digitize(y_val, self.y_edges) - 1

        if not (0 <= x_bin < len(self.x_edges) - 1):
            return -1
        if not (0 <= y_bin < len(self.y_edges) - 1):
            return -1

        return y_bin * (len(self.x_edges) -1) + x_bin

    def _compute_radius(self, row: pd.Series) -> float:
        r_xy = np.sqrt(row['posX']**2 + row['posY']**2)
        return r_xy

    def _load_data(self) -> pd.DataFrame:
        """
        Loads data from the ROOT file using uproot.
        Return a single pandas DataFrame with features, target,
        and derived columns ['bin_index', 'radius']
        """
        dfs = []
        processed = 0
        branches = self.features + list(self.branch_name) + [self.target]
        for arrays in ur.iterate(
            f"{self._inputfile}:{self.tree_name}",
            branches,
            step_size=self.chunck_size,
            library="np" # ensures dict of numpy arrays
        ):
            #Convert awkward to pandas
            df = pd.DataFrame({k: arrays[k] for k in arrays.keys()})

            #Verify that expected columns exist
            missing = [c for c in branches if c not in df.columns]
            if missing:
                raise KeyError(f"Missing branches in TTree: {missing}")
            #Add derived columns
            df['bin_index'] = df.apply(self._get_th2_bin_index, axis=1)
            df['radius'] = df.apply(self._compute_radius, axis=1)

            #Keep only valid entries
            df = df[df['bin_index'] >= 0]

            dfs.append(df)
            processed += len(df)

            if processed >= self.max_events:
                print(f"Reached max number of event: {self.max_events}. Stopping early...")
                break

        if not dfs:
            raise RuntimeError("***No valid events found in ROOT file!***")

        df_tree = pd.concat(dfs, ignore_index=True)
        #Apply final cap (in case las chunk overshoots)
        if len(df_tree) > self.max_events:
            df_tree = df_tree.iloc[:self.max_events]
        return df_tree

    def xgb_train(self,test_size=0.2, random_state=42):
        '''
        Train XGBoost Classifier
        '''
        df = self._load_data()
        X = df[self.features + ['bin_index', 'radius']]
        y = df[self.target]

        #Map labels -> 0..N-1
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoder = le

        #Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_classes = len(le.classes_)

        #XGBoost params:
        xgb_params={
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'num_class': num_classes,
            #'num_class': len(np.unique(y)),
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'tree_method': 'hist'
        }

        print("Training XGBoost model...")
        self.model = xgb.train(
            xgb_params, 
            dtrain, 
            num_boost_round=200,
            evals=[(dtrain,'train'),(dtest,'test')],
            verbose_eval=True
        )

        #Evaluate
        preds = self.model.predict(dtest)
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy: {acc: 4f}")

        return self.model
    def gbhm_train(self,test_size=0.3,random_state=42):
        """
        Trains the Gradient Boosting model.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data.
        """
        #Load and preprocess data
        data=self._load_data()
        X=data[self.features]
        y=data[self.target]

        #Split data into training and testing sets
        X_train, X_test, y_train, y_test=train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        #Initialize and train the Gradient Boosting model
        print("Training Gradient Boosting model...")
        self.model=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=random_state)
        self.model.fit(X_train,y_train)

        #Evaluate the model
        train_accuracy=accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy=accuracy_score(y_test, self.model.predict(X_test))
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

    def predict(self, new_data):
        """
        Makes predictions on new data.

        Args:
            new_data (pd.DataFrame): DataFrame containing the same features as the training data.

        Returns:
            np.ndarray: The predicted class labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        # Ensure the new data has the same columns as the training data
        missing_cols=set(self.features) - set(new_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in new data: {missing_cols}")

        return self.model.predict(new_data[self.features])

    def get_feature_importances(self):
        """Returns the feature importances from the trained model."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        importances=pd.Series(self.model.feature_importances_,index=self.features).sort_values(ascending=False)
        return importances

