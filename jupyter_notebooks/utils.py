from collections import defaultdict
from sklearn import metrics
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

# to visualize high-dimensional data into lower-dimension(2D)
def visualize_data(X)->None:
    """
    this method takes high-dimensional data and change it into lower-dimension
    using PCA. (2D)
    """
    pca = PCA(n_components=2, svd_solver="auto")
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:,0], X_pca[:,1])
    plt.title("Scatter plot for the given data")

def fit_and_evaluate(km, X, name=None, n_runs=5)->None:
    """
    this method fit and evaluate the model on the given dataset.
    
    returns
    -------
    It returns scores
    """
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        # as we don't have any labels to validate
        # we will validate our clusters using silhouette Coefficient
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
        # calinski-harabasz index is also know as Variance Ratio Criterion
        # The score is defined as ration of the sum of between-cluster dispersion and of within-cluster dispersion
        scores["Calinski-Harabasz Index"].append(
            metrics.calinski_harabasz_score(X, km.labels_)
        )
        scores["Davies-Bouldin Index"].append(
            metrics.davies_bouldin_score(X, km.labels_)
        )

    train_times = np.asarray(train_times)

    print(f"Clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s")

    evaluation = {
        "estimator": name, 
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }

    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")

    return scores