import joblib
import os
from sklearn.neural_network import MLPClassifier
from orientation.utils import to_ml_format


class MlpClassifier:

    def __init__(self, args):
        self.clf = MLPClassifier(max_iter=30000)
        self.args = args
        self.ckpt_path = os.path.join(
            self.args.work_dir, "mlpClf.joblib"
        )

    def dump(self, path):
        joblib.dump(self.clf, path)

    def load(self, path):
        self.clf = joblib.load(path)

    def train(self, dataset):
        x, y ,_= to_ml_format(dataset)
        self.clf.fit(x, y)
        self.dump(self.ckpt_path)

    def test(self, dataset):
        x, _, z = to_ml_format(dataset)
        if os.path.isfile(self.ckpt_path):
            self.load(self.ckpt_path)
        return self.clf.predict(x), z
