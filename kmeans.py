import numpy as np
import math
import matplotlib.pyplot as plt


# Khởi tạo cluster ban đầu

num_clusters = 4
initial_centroids = np.random.uniform(-500, 500, size=(num_clusters, 2))

# Tìm khoảng cách nhỏ nhất giữa các tâm ban đầu
min_pairwise_distance = 1e18
for i in range(num_clusters):
    for j in range(num_clusters):
        if i == j:
            continue
        distance = math.sqrt((initial_centroids[i][0] - initial_centroids[j][0])**2 +
                             (initial_centroids[i][1] - initial_centroids[j][1])**2)
        if distance < min_pairwise_distance:
            min_pairwise_distance = distance

print("Bound size:", min_pairwise_distance)

plt.scatter(initial_centroids[:,0], initial_centroids[:,1], color="purple", label="Init centers")


# Sinh dữ liệu quanh các tâm

cluster_data_1 = np.random.uniform(
    low=[initial_centroids[0][0] - min_pairwise_distance, initial_centroids[0][1] - min_pairwise_distance],
    high=[initial_centroids[0][0] + min_pairwise_distance, initial_centroids[0][1] + min_pairwise_distance],
    size=(50, 2)
)
cluster_data_2 = np.random.uniform(
    low=[initial_centroids[1][0] - min_pairwise_distance, initial_centroids[1][1] - min_pairwise_distance],
    high=[initial_centroids[1][0] + min_pairwise_distance, initial_centroids[1][1] + min_pairwise_distance],
    size=(50, 2)
)
cluster_data_3 = np.random.uniform(
    low=[initial_centroids[2][0] - min_pairwise_distance, initial_centroids[2][1] - min_pairwise_distance],
    high=[initial_centroids[2][0] + min_pairwise_distance, initial_centroids[2][1] + min_pairwise_distance],
    size=(50, 2)
)
cluster_data_4 = np.random.uniform(
    low=[initial_centroids[3][0] - min_pairwise_distance, initial_centroids[3][1] - min_pairwise_distance],
    high=[initial_centroids[3][0] + min_pairwise_distance, initial_centroids[3][1] + min_pairwise_distance],
    size=(50, 2)
)

plt.scatter(cluster_data_1[:,0], cluster_data_1[:,1], color="blue", alpha=0.5)
plt.scatter(cluster_data_2[:,0], cluster_data_2[:,1], color="blue", alpha=0.5)
plt.scatter(cluster_data_3[:,0], cluster_data_3[:,1], color="blue", alpha=0.5)
plt.scatter(cluster_data_4[:,0], cluster_data_4[:,1], color="blue", alpha=0.5)

# gộp tất cả điểm dữ liệu
data_points = np.concatenate((initial_centroids, cluster_data_1, cluster_data_2, cluster_data_3, cluster_data_4), axis=0)
num_points = len(data_points)

plt.show()


# KMeans++ Initialization

min_distances = np.zeros(num_points)
centroids = []

# chọn tâm đầu tiên ngẫu nhiên
rand_idx = np.random.randint(0, num_points)
first_centroid = data_points[rand_idx]
centroids.append(first_centroid)

# chọn thêm các tâm còn lại
while len(centroids) < num_clusters:
    for i in range(num_points):
        min_distance = 1e18
        for center in centroids:
            distance = math.sqrt((data_points[i][0] - center[0])**2 + (data_points[i][1] - center[1])**2)
            if distance < min_distance:
                min_distance = distance
        min_distances[i] = min_distance

    total_distance = np.sum(min_distances)
    probs = min_distances / total_distance

    next_idx = np.random.choice(num_points, p=probs)
    centroids.append(data_points[next_idx])

centroids = np.array(centroids)
print("Init centers (KMeans++):", centroids)

plt.scatter(data_points[:,0], data_points[:,1], color="gray", alpha=0.5)
plt.scatter(centroids[:,0], centroids[:,1], color="red", marker="x", s=200)
plt.title("KMeans++ Init")
plt.show()


# KMeans update

max_iterations = 25
for iteration in range(max_iterations):
    assignments = []
    for i in range(num_points):
        best_cluster = 0
        best_distance = 1e18
        for k in range(num_clusters):
            distance = math.sqrt((data_points[i][0] - centroids[k][0])**2 +
                                 (data_points[i][1] - centroids[k][1])**2)
            if distance < best_distance:
                best_distance = distance
                best_cluster = k
        assignments.append(best_cluster)
    assignments = np.array(assignments)

    updated_centroids = []
    for k in range(num_clusters):
        cluster_points = data_points[assignments == k]
        if len(cluster_points) > 0:
            new_x = np.mean(cluster_points[:,0])
            new_y = np.mean(cluster_points[:,1])
            updated_centroids.append((new_x, new_y))
        else:
            updated_centroids.append(centroids[k])  # giữ nguyên nếu cluster rỗng
    updated_centroids = np.array(updated_centroids)

    if np.allclose(centroids, updated_centroids):
        print("Converged at iteration", iteration)
        break
    centroids = updated_centroids


# Vẽ kết quả cuối

colors = ["blue", "green", "orange", "purple"]
for k in range(num_clusters):
    cluster_points = data_points[assignments == k]
    plt.scatter(cluster_points[:,0], cluster_points[:,1], alpha=0.5, color=colors[k])
plt.scatter(centroids[:,0], centroids[:,1], color="black", marker="x", s=200)
plt.title("Final KMeans clustering")
plt.show()
