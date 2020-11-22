## Fitting a diagonal covariance Gaussian mixture model to text data

In a previous assignment, we explored k-means clustering for a high-dimensional Wikipedia dataset. We can also model this data with a mixture of Gaussians, though with increasing dimension we run into two important issues associated with using a full covariance matrix for each component.
 * Computational cost becomes prohibitive in high dimensions: score calculations have complexity cubic in the number of dimensions M if the Gaussian has a full covariance matrix.
 * A model with many parameters require more data: observe that a full covariance matrix for an M-dimensional Gaussian will have M(M+1)/2 parameters to fit. With the number of parameters growing roughly as the square of the dimension, it may quickly become impossible to find a sufficient amount of data to make good inferences.

Both of these issues are avoided if we require the covariance matrix of each component to be diagonal, as then it has only M parameters to fit and the score computation decomposes into M univariate score calculations. Recall from the lecture that the M-step for the full covariance is:

\begin{align*}
\hat{\Sigma}_k &= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_i-\hat{\mu}_k)(x_i - \hat{\mu}_k)^T
\end{align*}

Note that this is a square matrix with M rows and M columns, and the above equation implies that the (v, w) element is computed by

\begin{align*}
\hat{\Sigma}_{k, v, w} &= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_{iv}-\hat{\mu}_{kv})(x_{iw} - \hat{\mu}_{kw})
\end{align*}

When we assume that this is a diagonal matrix, then non-diagonal elements are assumed to be zero and we only need to compute each of the M elements along the diagonal independently using the following equation. 

\begin{align*}
\hat{\sigma}^2_{k, v} &= \hat{\Sigma}_{k, v, v}  \\
&= \frac{1}{N_k^{soft}} \sum_{i=1}^N r_{ik} (x_{iv}-\hat{\mu}_{kv})^2
\end{align*}

In this section, we will use an EM implementation to fit a Gaussian mixture model with **diagonal** covariances to a subset of the Wikipedia dataset. The implementation uses the above equation to compute each variance term. 

We'll begin by importing the dataset and coming up with a useful representation for each article. After running our algorithm on the data, we will explore the output to see whether we can give a meaningful interpretation to the fitted parameters in our model.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import necessary packages


```python
from __future__ import print_function # to conform python 2.x print to python 3.x
import turicreate
```

We also have a Python file containing implementations for several functions that will be used during the course of this assignment.


```python
from em_utilities import *
```

## Load Wikipedia data and extract TF-IDF features

Load Wikipedia data and transform each of the first 5000 document into a TF-IDF representation.


```python
wiki = turicreate.SFrame('people_wiki.sframe/').head(5000)
wiki['tf_idf'] = turicreate.text_analytics.tf_idf(wiki['text'])
```

Using a utility we provide, we will create a sparse matrix representation of the documents. This is the same utility function you used during the previous assignment on k-means with text data.


```python
wiki = wiki.add_row_number()
tf_idf, map_word_to_index = sframe_to_scipy(wiki, 'tf_idf')
map_index_to_word = dict([[map_word_to_index[i], i] for i in map_word_to_index.keys()])
```

As in the previous assignment, we will normalize each document's TF-IDF vector to be a unit vector. 


```python
%%time
tf_idf = normalize(tf_idf)
```

    CPU times: user 15.6 ms, sys: 0 ns, total: 15.6 ms
    Wall time: 10.6 ms


We can check that the length (Euclidean norm) of each row is now 1.0, as expected.


```python
for i in range(5):
    doc = tf_idf[i]
    print(np.linalg.norm(doc.todense()))
```

    1.0
    1.0
    0.9999999999999997
    1.0000000000000004
    0.9999999999999999


## EM in high dimensions

EM for high-dimensional data requires some special treatment:
 * E step and M step must be vectorized as much as possible, as explicit loops are dreadfully slow in Python.
 * All operations must be cast in terms of sparse matrix operations, to take advantage of computational savings enabled by sparsity of data.
 * Initially, some words may be entirely absent from a cluster, causing the M step to produce zero mean and variance for those words.  This means any data point with one of those words will have 0 probability of being assigned to that cluster since the cluster allows for no variability (0 variance) around that count being 0 (0 mean). Since there is a small chance for those words to later appear in the cluster, we instead assign a small positive variance (~1e-10). Doing so also prevents numerical overflow.
 
We provide the complete implementation for you in the file `em_utilities.py`. For those who are interested, you can read through the code to see how the sparse matrix implementation differs from the previous assignment. 

You are expected to answer some quiz questions using the results of clustering.

**Initializing mean parameters using k-means**

Recall from the lectures that EM for Gaussian mixtures is very sensitive to the choice of initial means. With a bad initial set of means, EM may produce clusters that span a large area and are mostly overlapping. To eliminate such bad outcomes, we first produce a suitable set of initial means by using the cluster centers from running k-means.  That is, we first run k-means and then take the final set of means from the converged solution as the initial means in our EM algorithm.


```python
%%time 

from sklearn.cluster import KMeans

np.random.seed(5)
num_clusters = 25

# Use scikit-learn's k-means to simplify workflow
#kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1) # uncomment to use parallelism -- may break on your installation
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=1)
kmeans_model.fit(tf_idf)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

means = [centroid for centroid in centroids]
```

    /home/ruben/venv/lib/python3.6/site-packages/sklearn/cluster/_kmeans.py:939: FutureWarning: 'n_jobs' was deprecated in version 0.23 and will be removed in 0.25.
      " removed in 0.25.", FutureWarning)


    CPU times: user 11.9 s, sys: 578 ms, total: 12.5 s
    Wall time: 11.8 s


**Initializing cluster weights**

We will initialize each cluster weight to be the proportion of documents assigned to that cluster by k-means above.


```python
cluster_assignment
```




    array([22, 23, 15, ...,  2,  2,  8], dtype=int32)




```python
%%time 

num_docs = tf_idf.shape[0]
print('num docs: ' + str(num_docs))
weights = []
counted_docs = 0
for i in range(num_clusters):
    # Compute the number of data points assigned to cluster i:
    num_assigned = len(cluster_assignment[cluster_assignment == i]) # YOUR CODE HERE
    print('cluster: ' + str(i) + ' num_assigned: ' + str(num_assigned))
    w = float(num_assigned) / num_docs
    print('weight of cluster ' + str(i) + ': ' + str(w))
    counted_docs += num_assigned
    print('counted docs: ' + str(counted_docs))
    weights.append(w)
```

    num docs: 5000
    cluster: 0 num_assigned: 256
    weight of cluster 0: 0.0512
    counted docs: 256
    cluster: 1 num_assigned: 128
    weight of cluster 1: 0.0256
    counted docs: 384
    cluster: 2 num_assigned: 159
    weight of cluster 2: 0.0318
    counted docs: 543
    cluster: 3 num_assigned: 66
    weight of cluster 3: 0.0132
    counted docs: 609
    cluster: 4 num_assigned: 50
    weight of cluster 4: 0.01
    counted docs: 659
    cluster: 5 num_assigned: 106
    weight of cluster 5: 0.0212
    counted docs: 765
    cluster: 6 num_assigned: 391
    weight of cluster 6: 0.0782
    counted docs: 1156
    cluster: 7 num_assigned: 141
    weight of cluster 7: 0.0282
    counted docs: 1297
    cluster: 8 num_assigned: 838
    weight of cluster 8: 0.1676
    counted docs: 2135
    cluster: 9 num_assigned: 85
    weight of cluster 9: 0.017
    counted docs: 2220
    cluster: 10 num_assigned: 98
    weight of cluster 10: 0.0196
    counted docs: 2318
    cluster: 11 num_assigned: 263
    weight of cluster 11: 0.0526
    counted docs: 2581
    cluster: 12 num_assigned: 133
    weight of cluster 12: 0.0266
    counted docs: 2714
    cluster: 13 num_assigned: 235
    weight of cluster 13: 0.047
    counted docs: 2949
    cluster: 14 num_assigned: 240
    weight of cluster 14: 0.048
    counted docs: 3189
    cluster: 15 num_assigned: 415
    weight of cluster 15: 0.083
    counted docs: 3604
    cluster: 16 num_assigned: 84
    weight of cluster 16: 0.0168
    counted docs: 3688
    cluster: 17 num_assigned: 128
    weight of cluster 17: 0.0256
    counted docs: 3816
    cluster: 18 num_assigned: 83
    weight of cluster 18: 0.0166
    counted docs: 3899
    cluster: 19 num_assigned: 254
    weight of cluster 19: 0.0508
    counted docs: 4153
    cluster: 20 num_assigned: 91
    weight of cluster 20: 0.0182
    counted docs: 4244
    cluster: 21 num_assigned: 169
    weight of cluster 21: 0.0338
    counted docs: 4413
    cluster: 22 num_assigned: 70
    weight of cluster 22: 0.014
    counted docs: 4483
    cluster: 23 num_assigned: 344
    weight of cluster 23: 0.0688
    counted docs: 4827
    cluster: 24 num_assigned: 173
    weight of cluster 24: 0.0346
    counted docs: 5000
    CPU times: user 15.6 ms, sys: 15.6 ms, total: 31.2 ms
    Wall time: 39.3 ms


**Initializing covariances**

To initialize our covariance parameters, we compute $\hat{\sigma}_{k, j}^2 = \sum_{i=1}^{N}(x_{i,j} - \hat{\mu}_{k, j})^2$ for each feature $j$.  For features with really tiny variances, we assign 1e-8 instead to prevent numerical instability. We do this computation in a vectorized fashion in the following code block.


```python
covs = []
for i in range(num_clusters):
    member_rows = tf_idf[cluster_assignment==i]
    cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[i]**2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)
```

**Running EM**

Now that we have initialized all of our parameters, run EM.


```python
out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
```


```python
out['loglik']
```




    [3855847476.7012835, 4844053202.46348, 4844053202.46348]



## Interpret clustering results

In contrast to k-means, EM is able to explicitly model clusters of varying sizes and proportions. The relative magnitude of variances in the word dimensions tell us much about the nature of the clusters.

Write yourself a cluster visualizer as follows.  Examining each cluster's mean vector, list the 5 words with the largest mean values (5 most common words in the cluster). For each word, also include the associated variance parameter (diagonal element of the covariance matrix). 

A sample output may be:
```
==========================================================
Cluster 0: Largest mean parameters in cluster 

Word        Mean        Variance    
football    1.08e-01    8.64e-03
season      5.80e-02    2.93e-03
club        4.48e-02    1.99e-03
league      3.94e-02    1.08e-03
played      3.83e-02    8.45e-04
...
```


```python
# Fill in the blanks
def visualize_EM_clusters(tf_idf, means, covs, map_index_to_word):
    print('')
    print('==========================================================')
    
    num_clusters = len(means)
    for c in range(num_clusters):
        print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
        print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
        
        # The k'th element of sorted_word_ids should be the index of the word 
        # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
        sorted_word_ids = np.argsort(means[c])  # YOUR CODE HERE

        for i in sorted_word_ids[-5:]:
            print('{0: <12}{1:<10.2e}{2:10.2e}'.format(map_index_to_word[i], 
                                                       means[c][i],
                                                       covs[c][i]))
        print('\n==========================================================')
```


```python
'''By EM'''
visualize_EM_clusters(tf_idf, out['means'], out['covs'], map_index_to_word)
```

    
    ==========================================================
    Cluster 0: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    elected     2.91e-02    8.95e-04
    liberal     2.93e-02    4.55e-03
    party       5.89e-02    2.61e-03
    election    5.89e-02    3.21e-03
    minister    7.57e-02    7.42e-03
    
    ==========================================================
    Cluster 1: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    directed    3.39e-02    2.22e-03
    feature     3.69e-02    1.81e-03
    festival    4.66e-02    3.60e-03
    films       5.50e-02    2.97e-03
    film        1.76e-01    6.07e-03
    
    ==========================================================
    Cluster 2: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    design      3.20e-02    4.59e-03
    artist      3.61e-02    1.44e-03
    gallery     3.65e-02    3.40e-03
    museum      5.62e-02    7.27e-03
    art         1.26e-01    6.83e-03
    
    ==========================================================
    Cluster 3: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    team        4.68e-02    1.30e-03
    coach       5.57e-02    5.91e-03
    points      6.25e-02    5.92e-03
    nba         1.01e-01    1.22e-02
    basketball  1.86e-01    7.78e-03
    
    ==========================================================
    Cluster 4: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    league      4.31e-02    1.53e-03
    season      5.05e-02    2.52e-03
    ice         6.40e-02    2.97e-03
    nhl         1.56e-01    1.64e-02
    hockey      2.45e-01    1.64e-02
    
    ==========================================================
    Cluster 5: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    democratic  4.46e-02    3.02e-03
    district    4.60e-02    2.37e-03
    house       4.64e-02    2.41e-03
    senate      5.41e-02    6.28e-03
    republican  7.93e-02    5.20e-03
    
    ==========================================================
    Cluster 6: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    womens      1.21e-02    1.46e-03
    women       1.43e-02    1.36e-03
    miss        2.22e-02    7.76e-03
    her         1.00e-01    3.14e-03
    she         1.60e-01    4.65e-03
    
    ==========================================================
    Cluster 7: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         4.18e-02    5.99e-03
    medal       4.28e-02    2.44e-03
    olympics    4.69e-02    2.59e-03
    m           4.70e-02    7.58e-03
    championships7.78e-02    5.17e-03
    
    ==========================================================
    Cluster 8: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    he          1.04e-02    6.05e-05
    novel       1.07e-02    1.43e-03
    that        1.10e-02    1.73e-04
    published   1.23e-02    6.16e-04
    book        1.45e-02    9.38e-04
    
    ==========================================================
    Cluster 9: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    drama       5.03e-02    6.40e-03
    film        5.98e-02    3.44e-03
    actress     7.65e-02    4.29e-03
    her         8.99e-02    2.74e-03
    she         1.37e-01    4.25e-03
    
    ==========================================================
    Cluster 10: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    league      3.07e-02    2.01e-03
    coach       3.09e-02    4.45e-03
    team        4.13e-02    2.15e-03
    chess       4.52e-02    1.66e-02
    soccer      1.15e-01    2.86e-02
    
    ==========================================================
    Cluster 11: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    executive   2.15e-02    1.23e-03
    served      2.24e-02    6.99e-04
    committee   2.34e-02    2.38e-03
    chairman    2.44e-02    1.97e-03
    president   2.52e-02    1.29e-03
    
    ==========================================================
    Cluster 12: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    chinese     3.12e-02    5.33e-03
    kong        3.50e-02    8.64e-03
    hong        3.78e-02    9.92e-03
    jazz        6.07e-02    1.14e-02
    music       7.26e-02    3.48e-03
    
    ==========================================================
    Cluster 13: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    studies     2.41e-02    1.95e-03
    professor   2.74e-02    1.08e-03
    philosophy  2.86e-02    5.35e-03
    history     3.38e-02    2.81e-03
    university  3.47e-02    8.89e-04
    
    ==========================================================
    Cluster 14: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    comedy      2.86e-02    3.91e-03
    film        2.93e-02    1.16e-03
    television  3.21e-02    1.67e-03
    actor       3.56e-02    2.91e-03
    theatre     4.93e-02    6.17e-03
    
    ==========================================================
    Cluster 15: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    song        2.50e-02    1.81e-03
    released    3.13e-02    1.11e-03
    music       4.18e-02    1.96e-03
    band        5.35e-02    4.21e-03
    album       6.76e-02    4.78e-03
    
    ==========================================================
    Cluster 16: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    formula     6.06e-02    1.31e-02
    championship6.27e-02    4.54e-03
    racing      8.45e-02    8.26e-03
    pga         1.08e-01    2.65e-02
    tour        1.14e-01    1.92e-02
    
    ==========================================================
    Cluster 17: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    chef        3.27e-02    1.18e-02
    bbc         3.63e-02    7.41e-03
    show        3.75e-02    2.56e-03
    radio       5.18e-02    4.62e-03
    news        5.76e-02    8.06e-03
    
    ==========================================================
    Cluster 18: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    quarterback 4.02e-02    7.16e-03
    coach       6.74e-02    7.85e-03
    nfl         6.98e-02    9.15e-03
    yards       7.37e-02    1.72e-02
    football    1.11e-01    5.60e-03
    
    ==========================================================
    Cluster 19: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    cup         4.22e-02    2.46e-03
    rugby       4.35e-02    8.18e-03
    season      4.77e-02    2.30e-03
    club        5.04e-02    2.64e-03
    league      5.21e-02    3.13e-03
    
    ==========================================================
    Cluster 20: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    philharmonic4.96e-02    3.27e-03
    conductor   8.16e-02    1.01e-02
    symphony    8.70e-02    1.08e-02
    music       1.23e-01    6.15e-03
    orchestra   1.31e-01    1.06e-02
    
    ==========================================================
    Cluster 21: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    district    3.72e-02    4.20e-03
    attorney    3.74e-02    4.30e-03
    judge       4.59e-02    4.44e-03
    court       6.84e-02    5.24e-03
    law         9.52e-02    8.35e-03
    
    ==========================================================
    Cluster 22: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    season      5.58e-02    1.83e-03
    club        5.93e-02    1.76e-03
    australian  7.91e-02    1.58e-03
    afl         9.58e-02    1.31e-02
    football    1.21e-01    6.14e-03
    
    ==========================================================
    Cluster 23: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    physics     2.61e-02    5.43e-03
    professor   3.20e-02    1.26e-03
    university  3.34e-02    7.14e-04
    science     3.50e-02    2.95e-03
    research    5.70e-02    2.68e-03
    
    ==========================================================
    Cluster 24: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    sox         4.55e-02    6.28e-03
    games       4.66e-02    1.93e-03
    major       5.09e-02    1.19e-03
    league      1.03e-01    3.63e-03
    baseball    1.16e-01    5.57e-03
    
    ==========================================================


**Quiz Question**. Select all the topics that have a cluster in the model created above. [multiple choice]

- Baseball x
- Basketball x
- Soccer/Football x
- Music x
- Politics
- Law x
- Finance

## Comparing to random initialization

Create variables for randomly initializing the EM algorithm. Complete the following code block.


```python
np.random.seed(5) # See the note below to see why we set seed=5.
num_clusters = len(means)
num_docs, num_words = tf_idf.shape

random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    # YOUR CODE HERE
    mean = np.random.normal(0,1,num_words)
    
    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    # YOUR CODE HERE
    cov = np.random.uniform(1,5,num_words)

    # Initially give each cluster equal weight.
    # YOUR CODE HERE
    weight = 1/num_clusters
    
    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)
```

**Quiz Question**: Try fitting EM with the random initial parameters you created above. (Use `cov_smoothing=1e-5`.) Store the result to `out_random_init`. What is the final loglikelihood that the algorithm converges to? 2362457265.2187605


```python
out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, random_weights, cov_smoothing=1e-5)
out_random_init['loglik']
```




    [-764086029.088758,
     2282599968.73394,
     2362197958.6081905,
     2362457265.2184424,
     2362457265.2187605,
     2362457265.2187605]



**Quiz Question:** Is the final loglikelihood larger or smaller than the final loglikelihood we obtained above when initializing EM with the results from running k-means? smaller


```python
4844053202.46348 > 2362457265.2187605
```




    True



**Quiz Question**: For the above model, `out_random_init`, use the `visualize_EM_clusters` method you created above. Are the clusters more or less interpretable than the ones found after initializing using k-means? less


```python
# YOUR CODE HERE. Use visualize_EM_clusters, which will require you to pass in tf_idf and map_index_to_word.
visualize_EM_clusters(tf_idf, out_random_init['means'], out_random_init['covs'], map_index_to_word)
```

    
    ==========================================================
    Cluster 0: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    law         1.27e-02    2.59e-03
    university  1.43e-02    6.32e-04
    music       1.45e-02    1.29e-03
    award       1.53e-02    1.18e-03
    she         2.24e-02    3.54e-03
    
    ==========================================================
    Cluster 1: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    he          1.36e-02    1.13e-04
    football    1.53e-02    2.15e-03
    season      1.59e-02    1.17e-03
    league      1.91e-02    2.14e-03
    she         1.99e-02    2.80e-03
    
    ==========================================================
    Cluster 2: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    opera       1.03e-02    3.26e-03
    de          1.08e-02    1.92e-03
    music       1.13e-02    9.50e-04
    her         5.21e-02    5.62e-03
    she         5.65e-02    6.39e-03
    
    ==========================================================
    Cluster 3: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    her         1.24e-02    1.20e-03
    she         1.31e-02    1.69e-03
    hockey      1.33e-02    5.33e-03
    he          1.38e-02    1.07e-04
    film        2.98e-02    5.79e-03
    
    ==========================================================
    Cluster 4: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    university  1.31e-02    5.36e-04
    her         1.35e-02    1.09e-03
    he          1.36e-02    1.14e-04
    film        1.38e-02    1.72e-03
    she         1.96e-02    3.21e-03
    
    ==========================================================
    Cluster 5: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    court       9.54e-03    1.47e-03
    law         9.93e-03    1.96e-03
    he          1.31e-02    1.21e-04
    her         1.37e-02    1.09e-03
    she         2.03e-02    2.65e-03
    
    ==========================================================
    Cluster 6: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    state       1.03e-02    5.08e-04
    served      1.11e-02    3.79e-04
    he          1.18e-02    1.06e-04
    her         1.44e-02    1.18e-03
    she         2.70e-02    3.62e-03
    
    ==========================================================
    Cluster 7: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    marathon    1.25e-02    5.52e-03
    album       1.52e-02    2.38e-03
    music       1.73e-02    2.16e-03
    her         2.17e-02    1.89e-03
    she         3.41e-02    4.33e-03
    
    ==========================================================
    Cluster 8: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    played      1.35e-02    6.32e-04
    season      1.38e-02    8.80e-04
    he          1.46e-02    1.17e-04
    she         1.73e-02    2.67e-03
    league      1.87e-02    2.01e-03
    
    ==========================================================
    Cluster 9: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    cup         1.39e-02    1.44e-03
    played      1.45e-02    7.99e-04
    team        1.46e-02    8.86e-04
    her         1.52e-02    1.23e-03
    she         3.38e-02    4.81e-03
    
    ==========================================================
    Cluster 10: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    she         1.22e-02    2.06e-03
    university  1.23e-02    5.72e-04
    he          1.31e-02    1.01e-04
    york        1.49e-02    8.13e-04
    music       1.54e-02    1.51e-03
    
    ==========================================================
    Cluster 11: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    university  1.36e-02    3.43e-04
    music       1.36e-02    1.71e-03
    film        1.55e-02    2.19e-03
    her         1.61e-02    1.67e-03
    she         2.07e-02    2.65e-03
    
    ==========================================================
    Cluster 12: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    music       1.25e-02    1.40e-03
    her         1.26e-02    1.11e-03
    he          1.42e-02    1.22e-04
    tour        1.46e-02    3.64e-03
    she         2.06e-02    2.98e-03
    
    ==========================================================
    Cluster 13: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    music       1.44e-02    1.41e-03
    nixon       1.50e-02    9.78e-03
    art         1.60e-02    2.76e-03
    her         1.65e-02    1.35e-03
    she         2.12e-02    2.25e-03
    
    ==========================================================
    Cluster 14: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    university  1.04e-02    2.96e-04
    research    1.08e-02    1.06e-03
    he          1.31e-02    9.45e-05
    film        1.69e-02    2.75e-03
    music       1.85e-02    2.31e-03
    
    ==========================================================
    Cluster 15: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    season      1.22e-02    1.01e-03
    her         1.30e-02    1.32e-03
    league      1.32e-02    1.34e-03
    he          1.35e-02    1.11e-04
    she         2.14e-02    3.05e-03
    
    ==========================================================
    Cluster 16: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    show        1.14e-02    9.72e-04
    he          1.17e-02    1.01e-04
    film        1.36e-02    1.26e-03
    her         1.65e-02    1.29e-03
    she         2.12e-02    2.73e-03
    
    ==========================================================
    Cluster 17: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    her         1.05e-02    1.17e-03
    she         1.11e-02    1.28e-03
    he          1.18e-02    9.96e-05
    music       1.28e-02    1.23e-03
    film        1.66e-02    3.18e-03
    
    ==========================================================
    Cluster 18: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    university  1.19e-02    3.84e-04
    music       1.24e-02    9.29e-04
    band        1.64e-02    2.43e-03
    her         2.28e-02    1.75e-03
    she         4.03e-02    5.17e-03
    
    ==========================================================
    Cluster 19: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    club        1.11e-02    8.05e-04
    she         1.33e-02    1.71e-03
    he          1.33e-02    9.20e-05
    season      1.43e-02    1.04e-03
    league      1.51e-02    1.26e-03
    
    ==========================================================
    Cluster 20: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    law         1.24e-02    2.35e-03
    he          1.26e-02    1.05e-04
    her         1.45e-02    1.43e-03
    film        1.82e-02    1.93e-03
    she         2.63e-02    3.91e-03
    
    ==========================================================
    Cluster 21: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    her         1.21e-02    1.09e-03
    season      1.22e-02    1.09e-03
    league      1.37e-02    1.27e-03
    he          1.39e-02    1.30e-04
    she         1.86e-02    2.57e-03
    
    ==========================================================
    Cluster 22: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    her         1.00e-02    1.08e-03
    party       1.12e-02    8.99e-04
    he          1.29e-02    1.08e-04
    music       1.49e-02    1.63e-03
    she         1.50e-02    2.21e-03
    
    ==========================================================
    Cluster 23: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    york        1.18e-02    6.68e-04
    band        1.20e-02    1.10e-03
    music       1.62e-02    1.32e-03
    her         1.79e-02    1.67e-03
    she         2.27e-02    2.67e-03
    
    ==========================================================
    Cluster 24: Largest mean parameters in cluster 
    
    Word        Mean        Variance    
    soccer      1.16e-02    4.19e-03
    he          1.28e-02    1.26e-04
    album       1.50e-02    2.47e-03
    her         1.77e-02    1.67e-03
    she         2.62e-02    3.89e-03
    
    ==========================================================


**Note**: Random initialization may sometimes produce a superior fit than k-means initialization. We do not claim that random initialization is always worse. However, this section does illustrate that random initialization often produces much worse clustering than k-means counterpart. This is the reason why we provide the particular random seed (`np.random.seed(5)`).

## Takeaway

In this assignment we were able to apply the EM algorithm to a mixture of Gaussians model of text data. This was made possible by modifying the model to assume a diagonal covariance for each cluster, and by modifying the implementation to use a sparse matrix representation. In the second part you explored the role of k-means initialization on the convergence of the model as well as the interpretability of the clusters.
