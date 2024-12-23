from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Class that, depending on the dataset and preprocessing performed,  
# allows comparing the results of different Machine Learning algorithms, returning only the results

class TraditionalMLTechniques:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

    def knn(self, n_neighbors=10):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        knn.fit(self.X_train, self.y_train)

        score = knn.score(self.X_test, self.y_test)
        predict = knn.predict(self.X_test)

        return score, predict

    def svm(self, C=1.0, kernel='rbf'):
        svm = SVC(C=C, kernel=kernel)

        svm.fit(self.X_train, self.y_train)

        score = svm.score(self.X_test, self.y_test)
        predict = svm.predict(self.X_test)

        return score, predict

    def decision_tree(self, max_depth=None):
        dtc = DecisionTreeClassifier(max_depth=max_depth)

        dtc.fit(self.X_train, self.y_train)

        score = dtc.score(self.X_test, self.y_test)
        predict = dtc.predict(self.X_test)

        return score, predict

    def logistic_regression(self, max_iter=100):
        l_reg = LogisticRegression(max_iter=max_iter)

        l_reg.fit(self.X_train, self.y_train)

        score = l_reg.score(self.X_test, self.y_test)
        predict = l_reg.predict(self.X_test)

        return score, predict

    def compare_all(self):
        results = {
            "KNN": self.knn(),
            "SVM": self.svm(),
            "Decision Tree": self.decision_tree(),
            "Logistic Regression": self.logistic_regression(),
        }
        return results