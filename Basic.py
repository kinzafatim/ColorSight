import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (change to your image filename)
img = cv2.imread('/home/kinzaa/Desktop/Digital-Image-Processing/images/test5.jpg')          
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
img = img.astype(np.float32) / 255.0        # Normalize to [0, 1]

# Color transformation matrices
normal = np.eye(3)

# Anomalous trichromacy
protanomaly = np.array([[0.817, 0.183, 0],
                        [0.333, 0.667, 0],
                        [0,     0.125, 0.875]])

deuteranomaly = np.array([[0.8,   0.2,   0],
                          [0.258, 0.742, 0],
                          [0,     0.142, 0.858]])

tritanomaly = np.array([[0.967, 0.033, 0],
                        [0,     0.733, 0.267],
                        [0,     0.183, 0.817]])

# Dichromatic
protanopia = np.array([[0.567, 0.433, 0],
                       [0.558, 0.442, 0],
                       [0,     0.242, 0.758]])

deuteranopia = np.array([[0.625, 0.375, 0],
                         [0.7,   0.3,   0],
                         [0,     0.3,   0.7]])

tritanopia = np.array([[0.95,  0.05,  0],
                       [0,     0.433, 0.567],
                       [0,     0.475, 0.525]])

# transformation
def apply_matrix(img, matrix):
    reshaped = img.reshape(-1, 3)
    transformed = reshaped @ matrix.T
    return np.clip(transformed.reshape(img.shape), 0, 1)

# convert to grayscale RGB
def to_grayscale_rgb(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return np.stack([gray]*3, axis=-1) / 255.0

#  transformations
img_normal        = img
img_protanomaly   = apply_matrix(img, protanomaly)
img_deuteranomaly = apply_matrix(img, deuteranomaly)
img_tritanomaly   = apply_matrix(img, tritanomaly)
img_protanopia    = apply_matrix(img, protanopia)
img_deuteranopia  = apply_matrix(img, deuteranopia)
img_tritanopia    = apply_matrix(img, tritanopia)
img_monochrome    = to_grayscale_rgb(img)

titles = [
    "Normal (Trichromatic)", "Protanomaly (Anomalous)", "Protanopia (Dichromatic)", "Monochromatic",
    "", "Deuteranomaly (Anomalous)", "Deuteranopia (Dichromatic)", "",
    "", "", "Tritanomaly (Anomalous)", "Tritanopia (Dichromatic)", "Blue Cone Monochromacy"
]

images = [
    img_normal, img_protanomaly, img_protanopia, img_monochrome,
    None, img_deuteranomaly, img_deuteranopia, None,
    None, None, img_tritanomaly, img_tritanopia, img_monochrome
]

plt.figure(figsize=(15, 8))
for i, (title, image) in enumerate(zip(titles, images)):
    if image is not None:
        plt.subplot(3, 5, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
plt.tight_layout()
plt.savefig("color_vision_simulation_output.png")
print("Saved output to color_vision_simulation_output.png")
