import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist
import seaborn

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

class KMeans(object):
    def __init__(self, K, iters=10):
        self.K = K
        self.iters = iters
        self.means = None
        self.data = None
        self.clusters = None
        self.N = None

    def fit(self, X):
        # randomly assign clusters
        self.data = X
        self.N = X.shape[0]
        self.clusters = np.random.randint(0, self.K, self.N)

        loss = []
        for _ in range(self.iters):
            self.means = np.array([np.mean(self.data[self.clusters == i], axis=0) for i in range(self.K)])
            self.clusters = np.array([np.argmin(np.linalg.norm(self.data[i:i+1] - self.means, axis=1)) for i in range(self.N)])
            loss.append(np.sum((self.data - self.means[self.clusters]) ** 2))
        return np.array(loss)

    def get_mean_images(self):
        return self.means

    def get_counts(self):
        return np.unique(self.clusters, return_counts=True)[1]

    def get_clusters(self):
        return self.clusters

# linkage types for HAC
LINKAGES = ['max', 'min', 'centroid']

class HAC(object):
    def __init__(self, linkage, num_clusters=10):
        assert linkage in LINKAGES, "unknown linkage: please use max, min, or centroid"
        self.linkage = linkage
        self.num_clusters = num_clusters
        self.clusters = None
        self.data = None

    def fit(self, X):
        assert self.num_clusters <= X.shape[0], f"too few data points for {self.num_clusters} clusters"
        self.data = X
        clusters = [np.array([i]) for i in range(self.data.shape[0])]
        self.dists = cdist(self.data, self.data)
        while len(clusters) != self.num_clusters:
            min_dist = self.__compute_dist(clusters[0], clusters[1])
            # if pair is (u,v), then u must be strictly less than v
            closest_pair = (0, 1)
            for i in range(2, len(clusters)):
                for j in range(i):
                    dist = self.__compute_dist(clusters[j], clusters[i])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (j, i)
            # pop the higher index first since it doesn't affect the lower index
            clusters.append(np.concatenate([clusters.pop(closest_pair[1]), clusters.pop(closest_pair[0])]))
            if len(clusters) % 10 == 0:
                print(len(clusters))
        self.clusters = clusters

    def __compute_dist(self, cluster1, cluster2):
        if self.linkage == 'centroid':
            return np.linalg.norm(np.mean(self.data[cluster1], axis=0) - np.mean(self.data[cluster2], axis=0))

        # index into precomputed matrix of distances instead of recomputing each time
        a, b = np.meshgrid(cluster1, cluster2)
        dists = self.dists[a.flatten(), b.flatten()]
        return np.max(dists) if self.linkage == 'max' else np.min(dists)

    def get_mean_images(self):
        return np.vstack([np.mean(self.data[cluster], axis=0) for cluster in self.clusters])

    def get_counts(self):
        return np.array([cluster.shape[0] for cluster in self.clusters])

    def get_clusters(self):
        clusters = np.zeros(self.data.shape[0])
        for i, cluster in enumerate(self.clusters):
            clusters[cluster] = i
        return clusters


# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.show()

# Part 1
def part_1():
    KMeansClassifier = KMeans(K=10, iters=30)
    loss = KMeansClassifier.fit(large_dataset)
    plt.plot(np.arange(1, 31), loss)
    plt.title("Loss over Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.savefig("plots/kmeans_loss.png")
    plt.show()

part_1()

# Part 2
make_mean_image_plot(large_dataset, False)

# Part 3
std = np.std(large_dataset, axis=0)
std[std == 0] = 1
large_dataset_standardized = (large_dataset - np.mean(large_dataset, axis=0)) / std
make_mean_image_plot(large_dataset_standardized, True)

# Plotting code for part 4
def part_4():
    n_clusters = 10
    fig = plt.figure(figsize=(10,10))
    plt.suptitle("HAC mean images with max, min, and centroid linkages")
    for l_idx, l in enumerate(LINKAGES):
        # Fit HAC
        hac = HAC(l, num_clusters=n_clusters)
        hac.fit(small_dataset)
        mean_images = hac.get_mean_images()

        # Pickle HAC model to avoid re-clustering
        with open(f"pickle jar/{l}_HAC.pkl", mode="wb") as f:
            pickle.dump(hac, f)

        # Make plot
        for m_idx in range(mean_images.shape[0]):
            m = mean_images[m_idx]
            ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if m_idx == 0: plt.title(l)
            if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
            plt.imshow(m.reshape(28,28), cmap='Greys_r')
    plt.show()

part_4()

def retrieve_models():
    with open("pickle jar/max_HAC.pkl", mode="rb") as f:
        max_HAC = pickle.load(f)
    with open("pickle jar/min_HAC.pkl", mode="rb") as f:
        min_HAC = pickle.load(f)
    with open("pickle jar/centroid_HAC.pkl", mode="rb") as f:
        centroid_HAC = pickle.load(f)
    KMeansClassifier = KMeans(K=10)
    KMeansClassifier.fit(small_dataset)
    return (max_HAC, min_HAC, centroid_HAC, KMeansClassifier)

# Part 5
def part_5():
    max_HAC, min_HAC, centroid_HAC, KMeansClassifier = retrieve_models()
    plt.bar(np.arange(10)-0.3, max_HAC.get_counts(), 0.2, color="red")
    plt.bar(np.arange(10)-0.1, min_HAC.get_counts(), 0.2, color="blue")
    plt.bar(np.arange(10)+0.1, centroid_HAC.get_counts(), 0.2, color="purple")
    plt.bar(np.arange(10)+0.3, KMeansClassifier.get_counts(), 0.2, color="teal")
    plt.title("Image Distribution across 10 Clusters")
    plt.ylabel("Number of Images in Cluster")
    plt.xlabel("Cluster")
    plt.xticks(np.arange(10))
    plt.legend(["Max Linkage", "Min Linkage", "Centroid Linkage", "K-Means"], loc=2)
    plt.savefig("plots/cluster_distribution.png")

part_5()

def counts(a, n_clusters=10):
    return np.array([sum(a==i) for i in range(n_clusters)])

def confusion(a, b, n_clusters=10):
    return np.vstack([counts(b[a==j], n_clusters) for j in range(n_clusters)]).T

# Part 6
def part_6():
    max_cluster, min_cluster, centroid_cluster, kmeans_cluster = [model.get_clusters() for model in retrieve_models()]
    fig, axs = plt.subplots(3,2)
    fig.tight_layout()
    axs[0, 0].set_title("Max Linkage vs Min Linkage")
    seaborn.heatmap(confusion(max_cluster, min_cluster), ax=axs[0, 0])
    axs[1, 0].set_title("Max Linkage vs Centroid Linkage")
    seaborn.heatmap(confusion(max_cluster, centroid_cluster), ax=axs[1, 0])
    axs[2, 0].set_title("Max Linkage vs K-Means")
    seaborn.heatmap(confusion(max_cluster, kmeans_cluster), ax=axs[2, 0])
    axs[0, 1].set_title("Min Linkage vs Centroid Linkage")
    seaborn.heatmap(confusion(min_cluster, centroid_cluster), ax=axs[0, 1])
    axs[1, 1].set_title("Min Linkage vs K-Means")
    seaborn.heatmap(confusion(min_cluster, kmeans_cluster), ax=axs[1, 1])
    axs[2, 1].set_title("Centroid Linkage vs K-Means")
    seaborn.heatmap(confusion(centroid_cluster, kmeans_cluster), ax=axs[2, 1])
    plt.savefig("plots/confusion_matrices.png")
    plt.show()

part_6()