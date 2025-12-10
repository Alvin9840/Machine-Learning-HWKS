import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve

df = sns.load_dataset('titanic')
df.dropna(how='any', inplace=True)


def fun1():
    p = np.linspace(0, 1, 1000)
    p_safe = np.clip(p, 1e-10, 1 - 1e-10)
    Entropy = -p_safe * np.log2(p_safe) - (1 - p_safe) * np.log2(1 - p_safe)
    Gini = 2 * p_safe * (1 - p_safe)

    plt.plot(p, Entropy, linewidth=3)
    plt.plot(p, Gini, linewidth=3)
    plt.title('Entropy versus Gini index')
    plt.xlabel('p')
    plt.ylabel('Entropy/Gini')
    plt.legend(['Entropy', 'Gini impurity'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fun2(X_train, X_test, y_train, y_test):
    # 4.
    clf = DecisionTreeClassifier(random_state=5805)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print('Train Accuracy: ', round(train_acc, 2))
    print('Test Accuracy: ', round(test_acc, 2))

    print("Decision tree parameters:", clf.get_params())
    plt.figure(figsize=(12, 8))
    plot_tree(clf,
              feature_names=num_features,
              class_names=['not survived', 'survived'],
              filled=True,
              rounded=True)
    plt.tight_layout()
    plt.show()

    # 5.
    param_grid = {
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [20, 30, 40],
        'min_samples_leaf': [10, 20, 30],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=5805),
        param_grid,
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print('Best params(pre-pruned): ', grid_search.best_params_)
    clf_pruned = grid_search.best_estimator_
    train_acc_pruned = clf_pruned.score(X_train, y_train)
    test_acc_pruned = clf_pruned.score(X_test, y_test)
    print('Train Accuracy(pre-pruned): ', round(train_acc_pruned, 2))
    print('Test Accuracy(pre-pruned): ', round(test_acc_pruned, 2))

    plt.figure(figsize=(12, 8))
    plot_tree(clf_pruned,
              feature_names=num_features,
              class_names=['not survived', 'survived'],
              filled=True,
              rounded=True)
    plt.tight_layout()
    plt.show()

    y_pred_pruned = clf_pruned.predict(X_test)
    y_prob_pruned = clf_pruned.predict_proba(X_test)[:, 1]

    # 6.
    path = (DecisionTreeClassifier(random_state=5805)
            .cost_complexity_pruning_path(X_train, y_train))
    ccp_alphas = path.ccp_alphas

    train_acc_postpruned = []
    test_acc_postpruned = []
    for alpha in ccp_alphas:
        clf_postpruned = DecisionTreeClassifier(random_state=5805,
                                                ccp_alpha=alpha)
        clf_postpruned.fit(X_train, y_train)
        train_acc_postpruned.append(clf_postpruned.score(X_train, y_train))
        test_acc_postpruned.append(clf_postpruned.score(X_test, y_test))

    plt.figure(figsize(8, 6))
    plt.plot(ccp_alphas, train_acc_postpruned, marker='o', label='Train')
    plt.plot(ccp_alphas, test_acc_postpruned, label='Test')
    plt.xlabel('alpha')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.title("Accuracy vs. ccp_alphas for PostPruning")
    plt.show()

    best_alpha_idx = np.argmax(test_acc_postpruned)
    best_alpha = ccp_alphas[best_alpha_idx]
    print('Best alpha(post-pruned): ', round(best_alpha, 2))

    clf_postpruned = DecisionTreeClassifier(random_state=5805,
                                            ccp_alpha=best_alpha)
    clf_postpruned.fit(X_train, y_train)
    print('Train Accuracy(post-pruned): ', round(clf_postpruned.score(X_train, y_train), 2))
    print('Test Accuracy(post-pruned): ', round(clf_postpruned.score(X_test, y_test), 2))
    plt.figure(figsize=(12, 8))
    plot_tree(clf_postpruned,
              feature_names=num_features,
              class_names=['not survived', 'survived'],
              filled=True,
              rounded=True)
    plt.tight_layout()
    plt.show()

    y_pred_postpruned = clf_postpruned.predict(X_test)
    y_prob_postpruned = clf_postpruned.predict_proba(X_test)[:, 1]

    return clf_pruned, y_pred_pruned, y_prob_pruned, clf_postpruned, y_pred_postpruned, y_prob_postpruned


def fun3(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(max_iter=500, random_state=5805)
    lr.fit(X_train, y_train)
    train_acc_lr = lr.score(X_train, y_train)
    test_acc_lr = lr.score(X_test, y_test)
    print('Train Accuracy (Logistic Regression):', round(train_acc_lr, 2))
    print('Test Accuracy (Logistic Regression):', round(test_acc_lr, 2))

    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    return lr, y_pred_lr, y_prob_lr


def fun4(y_test, y_pred_pruned, y_prob_pruned, y_pred_postpruned, y_prob_postpruned, y_pred_lr, y_prob_lr):
    # 8.
    # Accuracy
    acc_pruned = accuracy_score(y_test, y_pred_pruned)
    acc_postpruned = accuracy_score(y_test, y_pred_postpruned)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    # Recall
    recall_pruned = recall_score(y_test, y_pred_pruned)
    recall_postpruned = recall_score(y_test, y_pred_postpruned)
    recall_lr = recall_score(y_test, y_pred_lr)
    # AUC
    auc_pruned = roc_auc_score(y_test, y_prob_pruned)
    auc_postpruned = roc_auc_score(y_test, y_prob_postpruned)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    # Confusion matrix
    cm_pruned = confusion_matrix(y_test, y_pred_pruned)
    cm_postpruned = confusion_matrix(y_test, y_pred_postpruned)
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    print("\nModel comparison (Test set):")
    print("---------------------------------------------------")
    print(f"{'Model':<18}{'Accuracy':<10}{'Recall':<10}{'AUC':<10}")
    print("---------------------------------------------------")
    print(f"{'DT Pre-Pruned':<18}{acc_pruned:.2f}{recall_pruned:>10.2f}{auc_pruned:>10.2f}")
    print(f"{'DT Post-Pruned':<18}{acc_postpruned:.2f}{recall_postpruned:>10.2f}{auc_postpruned:>10.2f}")
    print(f"{'Logistic Reg.':<18}{acc_lr:.2f}{recall_lr:>10.2f}{auc_lr:>10.2f}")
    print("---------------------------------------------------")
    print("\nConfusion matrices:")
    print("DT Pre-Pruned:\n", cm_pruned)
    print("DT Post-Pruned:\n", cm_postpruned)
    print("Logistic Regression:\n", cm_lr)

    # ROC curves
    fpr_pruned, tpr_pruned, _ = roc_curve(y_test, y_prob_pruned)
    fpr_postpruned, tpr_postpruned, _ = roc_curve(y_test, y_prob_postpruned)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_pruned, tpr_pruned, label=f'DT Pre-pruned (AUC={auc_pruned:.2f})')
    plt.plot(fpr_postpruned, tpr_postpruned, label=f'DT Post-pruned (AUC={auc_postpruned:.2f})')
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curves Comparison')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    num_features = [f for f in df.select_dtypes(include='number').columns.tolist() if f != 'survived']
    X = df[num_features]
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=5805)

    fun1()
    clf_pruned, y_pred_pruned, y_prob_pruned, clf_postpruned, y_pred_postpruned, y_prob_postpruned = fun2(X_train,
                                                                                                          X_test,
                                                                                                          y_train,
                                                                                                          y_test)
    lr, y_pred_lr, y_prob_lr = fun3(X_train, X_test, y_train, y_test)

    fun4(y_test, y_pred_pruned, y_prob_pruned, y_pred_postpruned, y_prob_postpruned, y_pred_lr, y_prob_lr)
