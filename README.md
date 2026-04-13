# Weekly-Challenge-12-K-Means-Clustering
For Week 12, I am exploring **Unsupervised Machine Learning**. Last week, I built KNN, which requires labeled data (supervised). This week, I built **K-Means Clustering** completely from scratch, an algorithm that receives raw, unlabeled data and finds hidden patterns and groupings entirely on its own.

This algorithm is widely used in customer segmentation, anomaly detection, and image compression.

## ⚙️ How it works
The algorithm follows a mathematically elegant optimization loop:
1. **Initialization:** Randomly place $K$ centroids (the "leaders" of the clusters) on the map.
2. **Assignment Step:** Measure the Euclidean distance from every data point to every centroid. Assign each point to its closest centroid.
3. **Update Step:** Calculate the average (mean) coordinates of all the points inside a cluster, and move the centroid to that exact center.
4. **Convergence:** Repeat steps 2 and 3 until the centroids stop moving. 

## 🚀 Complexity Analysis
* **Time Complexity:** $O(I \times K \times N)$ - Where $I$ is the number of iterations, $K$ is the number of clusters, and $N$ is the number of data points.
* **Space Complexity:** $O(N + K)$ - We need to store the dataset coordinates and the coordinates of the $K$ centroids in memory.

## 💻 Code Snippet (The Core Loop)
```python
# Calculate Euclidean distance and assign points to the closest centroid
distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
labels = np.argmin(distances, axis=1)

# Move the centroids to the mean of their assigned points
new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
