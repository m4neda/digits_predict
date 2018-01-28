from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def get_parmeters():
    return [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
             'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


def reshape(digits):
    n_samples = len(digits.images)
    return digits.images.reshape((n_samples, -1))


def fit(X_train, y_train, score):
    clf = GridSearchCV(SVC(C=1), get_parmeters(), cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    return clf


def main():
    digits = datasets.load_digits()
    X = reshape(digits)
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5,
        random_state=0,
    )

    parameters = get_parmeters()
    # 適合率で最適化
    score = 'precision'
    clf = fit(X_train, y_train, score)
    y_true, y_pred = y_test, clf.predict(X_test)

    predict_results = confusion_matrix(y_true, y_pred)
    print predict_results


if __name__ == '__main__':
    main()
