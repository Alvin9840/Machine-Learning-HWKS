import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.precision', 2)
np.random.seed(5805)

sample = 1000
mu_x = 1
sigma2_x = 2
sigma_x = np.sqrt(sigma2_x)

mu_epsilon = 2
sigma2_epsilon = 3
sigma_epsilon = np.sqrt(sigma2_epsilon)

x = np.random.normal(loc=mu_x, scale=sigma_x, size=sample)
epsilon = np.random.normal(loc=mu_epsilon, scale=sigma_epsilon, size=sample)

y = x + epsilon

X = np.column_stack((x, y))
means = np.mean(X, axis=0)
centered = X - means
cov_matrix = (centered.T @ centered) / (X.shape[0] - 1)

print ("variance of x =", sigma2_x)
print("variance of y = {:.2f}".format(np.var(y, ddof=1)))

#a
df_cov = pd.DataFrame(cov_matrix, columns=['x', 'y'], index=['x', 'y'])
print("Estimated Covariance Matrix:")
print(df_cov)

#b
e_vals, e_vecs = np.linalg.eig(cov_matrix)
eigen_df = pd.DataFrame({
    "eigenvalue": [e_vals[0], e_vals[1]],
    "eigenvector": [e_vecs[:, 0], e_vecs[:, 1]]
}, index=["lambda1", "lambda2"])

print("\n eigenvalue & eigenvector table")
print(eigen_df)

#c
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)

for i in range(2):
    vec = e_vecs[:, i] * np.sqrt(e_vals[i]) * 2
    plt.quiver(*means, *vec, angles='xy', scale_units='xy', scale=1, color=['red', 'blue'][i], label=f'eigenvector {i+1}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('scatter plot with eigenvectors')
plt.legend()
plt.grid(True)
plt.show()

#d
u, s, vt = np.linalg.svd(centered)

svd_df = pd.DataFrame({
    "singular value": [f"{s[0]:.4f}", f"{s[1]:.4f}"]
}, index=["s1", "s2"])

print(svd_df)

#e
df = pd.DataFrame(X, columns=['x', 'y'])
corr_matrix = df.corr()

print("\n correlation matrix:")
print(corr_matrix)

pcc = corr_matrix.loc['x', 'y']
print(f"\n sample Pearson correlation coefficient between x and y: {pcc:.2f}")