import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def isolate_character(src_image):
    image = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (1, 1, image.shape[1]-1, image.shape[0]-1)

    cv.grabCut(image, mask, rect, bgd_model, fgd_model, 7, cv.GC_INIT_WITH_RECT)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask[:, :, np.newaxis]
    pixels = result.reshape(-1, 3)

    num_clusters = 3 # shape, letter, and bg colors
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)
    labels = kmeans.predict(pixels)
    labels = labels.reshape(image.shape[:2])

    clustered_images = [np.zeros_like(result) for _ in range(num_clusters)]
    for i in range(num_clusters):
        clustered_images[i][labels == i] = result[labels == i]

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(result)
    axes[1].set_title("Background Removed")
    axes[2].imshow(clustered_images[2])
    axes[2].set_title("Final")
    
    plt.savefig('cluster_results.jpg')
    return clustered_images[2]
