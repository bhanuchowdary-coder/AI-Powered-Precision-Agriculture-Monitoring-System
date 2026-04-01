import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_ndvi(image):
    # Using Green as approximation for NIR
    nir = image[:, :, 1].astype(float)
    red = image[:, :, 2].astype(float)

    ndvi = (nir - red) / (nir + red + 1e-5)
    return ndvi

def classify_ndvi(ndvi):
    classified = np.zeros(ndvi.shape)

    classified[ndvi > 0.5] = 2        # Healthy
    classified[(ndvi > 0.2) & (ndvi <= 0.5)] = 1  # Moderate
    classified[ndvi <= 0.2] = 0       # Stressed

    return classified

def show_results(ndvi, classified):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("NDVI Heatmap")
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Crop Classification")
    plt.imshow(classified, cmap='viridis')
    plt.colorbar()

    plt.show()

def main():
    image = cv2.imread("sample.jpg")

    if image is None:
        print("Image not found")
        return

    ndvi = calculate_ndvi(image)
    classified = classify_ndvi(ndvi)

    show_results(ndvi, classified)

if __name__ == "__main__":
    main()
