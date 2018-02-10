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
    # score_wighted

    # 交差検証を行う
    # cv=5 ５分割する
    # precision 適合率を重視する
    # C=1 誤った分類をどの程度許容するか
    clf = GridSearchCV(SVC(C=1), get_parmeters(), cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    return clf


def main():
    # データ読み込み
    digits = datasets.load_digits()
    # imagesの8x8のarrayを1次元のarrayに変更
    # sklearnに読み込ませるため。8x8も1x64も同じ
    X = reshape(digits)
    y = digits.target

    # テストデータと検証データに分割
    # テストサイズは半分
    # random stateを設定することで、何度やっても同じ分割が得られるようにする
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5,
        random_state=0,
    )

    # 適合率で最適化
    # 100件のうち、正解が60だったら60÷100で0.6
    score = 'precision'
    clf = fit(X_train, y_train, score)
    y_true, y_pred = y_test, clf.predict(X_test)
    # 混同行列を表示
    predict_results = confusion_matrix(y_true, y_pred)
    print 'predict test data'
    print predict_results


if __name__ == '__main__':
    main()
