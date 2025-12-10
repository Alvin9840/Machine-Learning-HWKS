import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)


pf = fetch_lfw_people(min_faces_per_person=17)


def func5():
    # a.
    num_classes = len(pf.target_names)
    # b.
    num_images = pf.data.shape[0]
    # c.
    dimensionality = pf.data.shape[1]
    # d.
    names = pf.target_names

    print("number of unique politicians (classes):\n", num_classes)
    print("total number of images:\n", num_images)
    print("dimensionality of the dataset:\n", dimensionality)
    print("name of unique politician:\n")
    for name in names:
        print(name)

def func6():
    fig, axes = plt.subplots(5, 4, figsize=(8, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(pf.images[i], cmap='gray')
        ax.set_title(pf.target_names[pf.target[i]], fontsize=8)

    plt.tight_layout()
    plt.show()

def func7():
    counts = np.bincount(pf.target)

    plt.figure(figsize=(15, 10))
    plt.bar(pf.target_names, counts, color='orange', edgecolor='black')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("number of image")
    plt.title("number of images per politician")
    plt.tight_layout()
    plt.show()

def func8():
    X = pf.data
    y = pf.target
    classes = np.unique(y)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    #Sw
    n_features = X.shape[1]
    Sw = np.zeros((n_features, n_features))

    for c in classes:
        X_c = X[y == c]
        mean_c = X_c.mean(axis=0)
        for x in X_c:
            diff = (x - mean_c).reshape(-1, 1)
            Sw += diff @ diff.T

#Sb
    overall_mean = X.mean(axis=0)
    Sb = np.zeros((n_features, n_features))

    for c in classes:
        X_c = X[y == c]
        mean_c = X_c.mean(axis=0)
        n_c = X_c.shape[0]
        diff = (mean_c - overall_mean).reshape(-1, 1)
        Sb += n_c * (diff @ diff.T)

    print("Sw:\n", np.round(Sw[:5, :5], 2))
    print("Sb:\n", np.round(Sb[:5, :5], 2))
    return Sw, Sb

def func9( Sw, Sb):
    A = np.linalg.inv(Sw) @ Sb
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    idx_desc = np.argsort(eigvals)[::-1]
    idx_asc = np.argsort(eigvals)

    def show_eigs(indices, title, k=5, preview=10):
        print(title)
        for rank, i in enumerate(indices[:k], start=1):
            val = eigvals[i]
            vec = eigvecs[:, i]
            print(f"{rank}. eigenvalue = {val:.2f}")
            print(f"   eigenvector (first {preview} entries): {vec[:preview]}")
        print()

    show_eigs(idx_desc, "five largest eigenvalues and their corresponding eigenvectors:", k=5, preview=10)
    show_eigs(idx_asc, "five smallest eigenvalues and their corresponding eigenvectors:", k=5, preview=10)


# func5()
# func6()
# func7()
# func8()
# Sw, Sb = func8()
Sw = [[2322.4,  2169.98, 1882.1,  1591.92, 1387.71],
 [2169.98, 2283.25, 2123.59, 1797.16, 1535.32],
 [1882.1,  2123.59, 2275.36, 2107.61, 1804.46],
 [1591.92, 1797.16, 2107.61, 2293.39, 2147.38],
 [1387.71 ,1535.32 ,1804.46 ,2147.38, 2324.52]]
Sb = [[1059.59 ,1072.23 ,1050.9,   994.53  ,908.68],
 [1072.23, 1098.76, 1091.23 ,1044.25 , 962.55],
 [1050.9 , 1091.23, 1106.64, 1081.28, 1013.88],
 [ 994.53, 1044.25, 1081.28, 1088.61, 1052.1 ],
 [ 908.68,  962.55, 1013.88, 1052.1,  1057.48]]
func9(Sw, Sb)
