import numpy as np
import random as rn
import matplotlib.pyplot as plt

class k_means():
    def __init__(self, data, k_clusters, max_iterations):
        self.data = data
        self.k_clusters = k_clusters
        self.max_iterations = max_iterations

    def create_clusters(self):
        clustersDic = {'cluster'+str(i): [] for i in range(self.k_clusters)}
        centroides = [[rn.uniform(np.min(self.data[0]),np.max(self.data[0])), rn.uniform(np.min(self.data[1]),np.max(self.data[1]))] for i in range(self.k_clusters)]
        return clustersDic, centroides
    
    def distance(self,point1, point2):
        distace = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return distace
    
    def findCentroid(self,clusters):
        centroids = []
        for points in clusters:
            points = np.array(points).T
            centroids.append([np.mean(points[0]), np.mean(points[1])])
        return centroids

    def update_cluster(self, centroids):
        clustersDis = [[] for i in range(self.k_clusters)]
        for i in range(len(self.data[0])):
            dis = [self.distance(([self.data[0][i],self.data[1][i]]), centroids[j]) for j in range(self.k_clusters)]
            minDist = dis.index(np.min(dis))
            clustersDis[minDist].append([self.data[0][i],self.data[1][i]])
            clustersDis = np.array(clustersDis)
        centroids = self.findCentroid(clustersDis)
        return clustersDis, centroids

    def main(self):
        color = []
        for i in range(self.k_clusters):
            color.append('#%06X' % rn.randint(0, 0xFFFFFF))
        clusters, centroids = self.create_clusters()
        for i in range(self.max_iterations):
            clusters, centroids = self.update_cluster(centroids)
        for j in range(self.k_clusters):
            for point in clusters[j]:
                plt.plot(point[0],point[1],'o', color = color[j])
            plt.scatter(centroids[j][0], centroids[j][1], c='black', s=200, alpha=0.8)