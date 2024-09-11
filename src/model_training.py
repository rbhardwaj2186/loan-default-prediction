from sklearn.ensemble import RandomForestClassifier
import joblib

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def train(self):
        self.model = RandomForestClassifier(random_state=123)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
        return self.model