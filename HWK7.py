import numpy as np
import matplotlib.pyplot as plt
from numba.np.arrayobj import np_repeat
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt


def cross_entropy(x, y):
    p = 1 / (1 + np.exp(-x))
    return - (y * np.log(p) + (1 - y) * np.log(1 - p))


# 2.
def fun2():
    x = np.linspace(-10, 10, 200)
    y0 = cross_entropy(x, 0)  # y=0
    y1 = cross_entropy(x, 1)  # y=1

    plt.plot(x, y0, label='y=0', linewidth=3, linestyle='--')
    plt.plot(x, y1, label='y=1', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Log-loss function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fun3():
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_clusters_per_class=2,
        n_informative=2,
        n_repeated=0,
        n_redundant=0,
        random_state=5805
    )
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=5805)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # i.
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    # plt.show()

    # ii.
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # iii.
    print('accuracy:', accuracy_score(y_test, y_pred))
    print('precision:', precision_score(y_test, y_pred))
    print('recall:', recall_score(y_test, y_pred))


# fun2()
fun3()
