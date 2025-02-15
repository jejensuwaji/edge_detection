import imageio
import numpy as np
import matplotlib.pyplot as plt

# Menggunakan raw string untuk path
image_path = r'C:/Users/User/Documents/unsia/semester 7/pengolahan citra/UAS/fb.jpg'

# Membaca citra dengan mode 'L' untuk hasil integer
image = imageio.imread(image_path, mode='L')

# Fungsi untuk deteksi tepi menggunakan operator Robert
def robert_edge_detection(image):
    robert_cross_v = np.array([[1, 0], [0, -1]])
    robert_cross_h = np.array([[0, 1], [-1, 0]])

    vertical = np.zeros(image.shape, dtype=np.float32)
    horizontal = np.zeros(image.shape, dtype=np.float32)

    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            vertical[i, j] = np.sum(np.multiply(robert_cross_v, image[i:i+2, j:j+2]))
            horizontal[i, j] = np.sum(np.multiply(robert_cross_h, image[i:i+2, j:j+2]))

    edge_magnitude = np.sqrt(np.square(vertical) + np.square(horizontal))
    edge_magnitude *= 255.0 / edge_magnitude.max()
    return edge_magnitude

# Fungsi untuk deteksi tepi menggunakan operator Sobel
def sobel_edge_detection(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    vertical = np.zeros(image.shape, dtype=np.float32)
    horizontal = np.zeros(image.shape, dtype=np.float32)

    for i in range(image.shape[0] - 2):
        for j in range(image.shape[1] - 2):
            vertical[i, j] = np.sum(np.multiply(sobel_y, image[i:i+3, j:j+3]))
            horizontal[i, j] = np.sum(np.multiply(sobel_x, image[i:i+3, j:j+3]))

    edge_magnitude = np.sqrt(np.square(vertical) + np.square(horizontal))
    edge_magnitude *= 255.0 / edge_magnitude.max()
    return edge_magnitude

# Deteksi tepi menggunakan operator Robert
robert_edges = robert_edge_detection(image)

# Deteksi tepi menggunakan operator Sobel
sobel_edges = sobel_edge_detection(image)

# Menampilkan hasil
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(robert_edges, cmap='gray')
plt.title('Deteksi Tepi Robert')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Deteksi Tepi Sobel')
plt.axis('off')

plt.tight_layout()
plt.show()
