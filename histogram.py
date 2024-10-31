import imageio
import numpy as np
import matplotlib.pyplot as plt

image_rgb = imageio.imread('D:\\Perkuliahan\\S5\\Pengolahan Citra Digital\\s3\\Tugas Histogram\\lawu.jpg')

def rgb_to_grayscale(rgb_image):
    grayscale_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale_image

image_gray = rgb_to_grayscale(image_rgb)

hist, bins = np.histogram(image_gray, bins=256, range=(0, 255))

print("Jumlah total piksel untuk setiap intensitas:")
for intensity, count in enumerate(hist):
    print(f"Intensitas {intensity}: {count} piksel")

dominant_intensity = np.argmax(hist)
dominant_count = hist[dominant_intensity]
print(f"\nIntensitas dominan: {dominant_intensity} dengan {dominant_count} piksel\n")

fig, axs = plt.subplots(2, 2, figsize=(12, 8), 
                        gridspec_kw={'height_ratios': [1, 0.4], 'hspace': 0.4})

axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title('Gambar Asli (RGB)')
axs[0, 0].axis('off')

axs[0, 1].imshow(image_gray, cmap='gray')
axs[0, 1].set_title('Gambar Grayscale')
axs[0, 1].axis('off')

axs[1, 0].plot(hist, color='black')
axs[1, 0].set_title('Histogram Gambar Grayscale')
axs[1, 0].set_xlabel('Pixel Intensity (0-255)')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].grid(True)

fig.delaxes(axs[1, 1])

axs[1, 0].set_position([0.25, 0.1, 0.5, 0.3])  # [left, bottom, width, height]

plt.tight_layout()
plt.show()
