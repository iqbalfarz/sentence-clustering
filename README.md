# Sentence Clustering
Cluster Sentences into their dedicated group/cluster.
## Table of Contents
* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Applications of Clustering sentences](#applications-of-clustering-sentences)
* [Source and Useful Links](#source-and-useful-links)
* [Real-world/business Objectives and Constraints](#real-world-business-objectives-and-constraints)
* [Mapping to Machine Learning Problem](#mapping-to-machine-learning-problem)
* [Model Training](#model-training)
* [Technical Aspect](#technical-aspect)
* [Installation](#installation)
* [Run](#run)
* [Directory Tree](#directory-tree)
* [Future Work](#future-work)
* [Technologies used](#technologies-used)



## Overview
This repository contains different approaches to cluster sentences and showing their performance.

## Problem Statement
The task is to write an algorithm to cluster the texts given.

## Applications of Clustering Sentences
Sentence Clustering can be used to:
1. Label data faster
2. Reduced the number of labels needed
3. Debug structural model mistakes
4. Monitor Data Drift
5. Help Clean up noise from labelled datasets.

#### 1. Label data faster:
    One of the most time-consuming process of any Machine Learning workflow is data labeling. Raw data available but high-quality data is scarce. 
    A better approach can be while dealing with sentence is to first cluster the sentences and then show them in order, so the semantically similar sentences will be grouped.


#### 2. Reduction in the number of labels needed
    Some time we many need to put out the noisy datasets. So, we can use the sentence clustering to put out the noisy datapoints out of the dataset.


#### 3. Model Debugging
    Building machine learning models is not a linear but cyclical process. A model can always be improved and needs to be re-trained when new (labelled) data comes in. Sentence clustering can help in a manual error analysis of the model assessing whether model mistakes are structural errors (i.e. recurring for similar patterns) or not. By manually checking the performance of the model on evaluation samples and grouping them by semantic similarity, we can potentially find “hard” clusters. This can then inform us about new model improvement ideas such as different model architectures or pre- or post-processing steps.

#### 4. Data and Model Drift Monitoring

    An important issue with machine learning models that are used in production is data and model drift. This occurs when the distribution of the production data shifts relatively to the training data distribution. This could happen for many reasons, such as new types of customers, change in user behavior, or change in pre-processing pipelines of incoming data. This can be harmful as the models were not trained on this new type of data and might not be able to generalize well. It can often go unnoticed if not monitored correctly as ML teams often rely on fixed test sets which are not updated frequently to account for this distribution shift. Sentence clustering can be helpful here too. By checking whether sentence clusters in the production data align with the clusters in the training and test sets we have some proxy for detecting distribution shift.

#### 5. Data Cleaning
    The last application where sentence clustering proved to be useful for us is data cleaning. As machine learning engineers we often consider our labelled data as the ground truth or gold standard labels. However, in many cases, this ground truth data contains many mistakes and inconsistencies that could confuse our model during training. This is only natural as humans, like machine learning models, make mistakes. Often, large datasets are labelled at multiple time periods by distinct annotators, leading to inconsistencies.

    We can employ noise-robust training techniques that try to reduce the impact of noise in a labelled dataset, but sometimes it is better to inspect our labelled dataset manually and remove or correct noisy labels. By grouping our labelled dataset by sentence similarity, we can quickly inspect thousands of samples and look for inconsistencies.

## Source and Useful Links

### Data Source: 
- Download the dataset from [here!](https://docs.google.com/spreadsheets/d/1fRNebu-9wWT_dOfAqAiMd_rkE1XWtmmv/edit?usp=sharing&ouid=104304060814414753396&rtpof=true&sd=true)
- Dataset contains around 46K sentences.
- Dataset doesn't contains any labels
### Example
|Sentence |Cluster No.|
|:--------|-----------|
|Yesterday england won the cricket world cup| 1|
|India beat bangladesh in Fifa world cup qualifiers| 1|
|E-way bill up in june after May’s sharp drop| 2|
|India is heading towards v-shaped recovery according to
economic survey| 2|

### Other beneficial links
- Different Clustering Algorithms: [Click to see!](https://scikit-learn.org/stable/modules/clustering.html#)
- HDBSCAN: Faster than DBSCAN of sklearn ([Why choose HDBSCAN!](https://nbviewer.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb))
- Clustering Performance Evaluation: [clustering metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- MiniBatchKMeans: [Faster than KMeans](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans)
- Word2Vec: [Understanding Word Embeddings](https://jalammar.github.io/illustrated-word2vec/)
- Clustering millions of sentence: [Click Here!](https://ntropy.com/post/clustering-millions-of-sentences-to-optimize-the-ml-workflow#:~:text=With%20sentence%20clustering%2C%20we%20refer,a%20predefined%20list%20of%20topics.)

## Mapping to Machine Learning Problem
This is an Unsupervised Learning Problem known as Clustering.

### Instructions:
1. You can use any publicly available dataset for training, if required(__TIP:__ use wikipedia or some news related dataset.)
2. Not supposed to use the pre-trained embedding models like Word2Vec, Fasttext, or SentenceBERT, etc. (You can train the Word2Vec, Fasttext on new datasets as mentioned above).
### Guidelines
1. Neatly documented approach note on algorithm and how to run, and a csv file with the labels against sentences.
### BONUS: 
Time complexity will matter while doing assessment of your code and how are validating your clusters.


## Model Training
We will train different Clustering Algorithms like `MiniBatchKMeans` (faster version of KMeans), and `community_detection` (by sentence_transformers) which uses cosine distance to get similarity of two sentences.

- ### For Evaluation of the model we will use Silhouette Score as we are not having labels to validate the cluster

- ### We will also write an script to validate the model using labels (If someone has!)

## Technical Aspect
For in-detail technical aspects that I have tried, please go to the [`jupyter_notebooks\`]() folder.

## Installation
The code is written in python==3.9.5. If you want to install python,  [Click here](https://www.python.org/downloads/). Don't forget to upgrade python if you are using a lower version. upgrade using `pip install --upgrade python`. 
1. Clone the repository
2. install requirements.txt:
    ```
    pip isntall -r requirements.txt
    ```
## Run
You will have to manually run each jupyter notebooks and see the result.

- ### To get the label on the clustering dataset with the best found clustering, please use below script
```
python get_labels.py --dataset_path <path_to_the_dataset_csv_or_xlsx_file> --output_filepath <output_filepath_with_filename> --label <whether_to_use_groundtruth_label_or_not>
```
__NOTE:__
- `--dataset_path`: filepath of the dataset containing text.
- `--output_filepath`: Filepath to store the resultant dataset containing predictions for each sentence in the dataset. Predicted column name is `predictions`
- `--label`: Whether to use groundtruth label or not to judge the clustering model. The default is `False`.
- dataset should have a column `Text` to get the text data from the dataset csv file.
- If you want to validate the performance of the model on the labeled dataset, dataset should have column `Text`, and `label`.



## Directory Tree
```
├───dataset
│   └───clustering.xlsx
├───images
├───jupyter_notebooks
│   ├───1.data_analysis_preprocessing.ipynb
│   ├───2.MiniBatchKMeans_clustering_on_tfidf.ipynb
│   ├───3.MiniBatchKMeans_clustering_tfidf_lemmatization.ipynb
│   ├───4.training-word2vec-on-amazon-reviews-dataset.ipynb
│   ├───5.MiniBatchKmeans_clustering_on_word2vec.ipynb
├───model
│   ├─── contains vectorizer, LSA pipeline, and best MiniBatchKMeans model
├───result
│   ├─── containts images and HTML file for the 3D visualization
├───README.md
└───LICENSE
```

## Future Work
1. Try scalable version of HDBSCAN
2. Come up with better understanding of the dataset
3. Try different Embedding techniques like [`Doc2Vec`](https://radimrehurek.com/gensim/models/doc2vec.html), [`Sense2Vec`](https://spacy.io/universe/project/sense2vec)   , etc. 
4. Come up with better pre-processing also.

## Technologies used
[![](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org)
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width=200 height=100>](https://scikit-learn.org/stable/)
[<img src="https://numpy.org/images/logo.svg" width=200>](https://www.numpy.org)
[<img src="https://www.sbert.net/_static/logo.png" width=200>](https://www.sbert.net/)


## Summary
1. It is hard to cluster sentences without prior Knowledge about the type of data coming.
2. It is very hard to get the value of K for MiniBatchKMeans Because the dataset is too messy.
3. It is better to come up with prior assumption for the data to select the appropriate no.of clusters.
4. I found the best algorithm is `community_detection` because their parameters are very intuitive. Parameters are `threshold` and `min_community_size`.
    - `min_community_size`: Minimum no.of sentences/docs to be present in the cluster.
    - `threshold`: threshold is cosine similarity. greater than the threshold will be clustered.
    - The problem is that It doesn't guarantee for each sentence to be clustered.

5. Even tuning the `community_detection` is hard as lack of Problem definition that what we wants out of the dataset given.