import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from time import time

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer


def prediction(dataset_path, output_filepath, label=False):

    start_time = time()
    # load dataset
    # as the dataset is in the form of xlsx
    print("[INFO] loading dataset...")
    if "xlsx" in dataset_path:
        dataset = pd.read_excel(open(dataset_path, "rb"))
    else:
        dataset = pd.read_csv(dataset_path)
    print("[INFO] dataset loaded.")

    if "Text" not in dataset.columns:
        raise ValueError("dataset must have a column name `Text`")

    if label is True and "label" not in dataset.columns:
        raise ValueError("You have made `label=True` but dataset doesn't contain `label` column.")

    # a little bit processing to the dataset
    if label==True:
        dataset = dataset[["Text", "label"]]
    else:
        dataset = dataset[["Text"]]
    dataset.fillna("None",inplace=True)
    
    # load vectorizer
    # print("[INFO] loading vectorizer...")
    # vectorizer = joblib.load("./model/tfidf_vectorizer.pickle")

    # doing featurization usign TfidfVectorizer
    vectorizer = TfidfVectorizer(
        # if the word is present in more than 80% of documents, will not be considered
        max_df=0.8,
        # minimum number of docs which consists particular word
        min_df=5,
        stop_words="english",
        ngram_range=(1,3),
        use_idf=True,
    )
    t0 = time()
    X = vectorizer.fit_transform(dataset["Text"])
    print(f"vectorization done in {time() - t0:.5f} s")
    print(f"n_samples: {X.shape[0]}, n_features: {X.shape[1]}")

    # transform the dataset
    # print("[INFO] Getting TF-IDF embedding for the dataset...")
    # X = vectorizer.fit_transform(dataset["Text"])

    # load lsa pipeline to reduce the dimension
    # print("[INFO] loading LSA model...")
    # lsa = joblib.load("./model/lsa.pickle")

    # print("[INFO] reducing dimension...")
    # X = lsa.fit_transform(X)

    print(f"[INFO] Reducing dimensionality...")
    # As TruncatedSVD result is not Normalized, we'll Normalize the result also
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    t0 = time()
    X = lsa.fit_transform(X)
    explained_variance = lsa[0].explained_variance_ratio_.sum()
    print(f"LSA done in {time() - t0:.3f} s")
    print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

    # # load MiniBatchKMeans model
    # print("[INFO] loading MiniBatchKMeans model...")
    # kmeans = joblib.load("./model/best_mb_kmeans.pickle")

    print("[INFO] fitting MiniBatchKMeans...")
    from sklearn.cluster import MiniBatchKMeans
    K = 156 # best K found
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        max_iter=100,
        n_init=10,
    )
    kmeans.fit(X)
    # getting predictions on the dataset
    print("[INFO] Getting predictions...")
    dataset["predictions"] = pd.Series(kmeans.predict(X))

    if label==True:
        print("[METRICS] Calculating metrices using groundtruth labels...")
        # get the scores which needs label of the dataset
        print(f"adjusted mutual info score [perfect is 1.0]: {metrics.adjusted_mutual_info_score(dataset.label, dataset.predictions)}")

        # get the homogeneity (each cluster contains only members of a single class.)
        print(f"homogeneity [each cluster contains only members of a single class]: {metrics.homogeneity_score(dataset.label, dataset.predictions)}")

        # get the comleteness (all members of a given class are assigned to the same cluster.)
        print(f"Completeness [all members of a given class are assigned to the same cluster]: {metrics.completeness_score(dataset.label, dataset.predictions)}")

        # get the V-measure (harmonic mean of homogeneity and completeness)
        print(f"V-measure (harmonic mean of homogeneity and completeness): {metrics.v_measure_score(dataset.label, dataset.predictions)}")


    print("[UNSUPERVISED-METRIC] Calculating Silhouette Coefficient")
    # getting "Silhouette Coefficient value"
    print(f"Silhouette Coefficient is: {metrics.silhouette_score(X, dataset.predictions)}")

    # saving the dataset to the output path
    print(f"[INFO] Saving the dataset to {output_filepath}")
    dataset.to_csv(output_filepath, index=False)

    print(f"[TIME-TAKEN] total time taken: {time()-start_time}s")

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--dataset_path",
        help="filepath of the dataset containing Text",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_filepath",
        help="output filepath to store the dataset with predictions",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--label",
        help="whether to use the label of original dataset to validate the cluster or not. Boolean value",
        type=bool,
        default=False,
        required=False
    )
    args = parser.parse_args()

    prediction(
        args.dataset_path,
        args.output_filepath,
        args.label,
    )