import cv2
import numpy as np


def linear_transform(img, a=1.2, b=30):
    """Brightness + contrast adjustment with scalar gain and bias"""
    return np.clip(a * img + b, 0, 255).astype(np.uint8)


def spatial_transform(img):
    """Gain and bias vary across the image (e.g., simulate vignetting)"""
    h, w, c = img.shape
    y = np.linspace(0, 1, h)[:, None]  # vertical gradient
    a = np.tile(y, (1, w))[:, :, None]  # shape: (h, w, 1)
    b = np.tile(50 * (1 - y), (1, w))[:, :, None]
    return np.clip(a * img + b, 0, 255).astype(np.uint8)


def blend_color(img0, img1, alpha=0.5):
    """Linear blend between two images"""
    img1_resized = cv2.resize(img1, (img0.shape[1], img0.shape[0]))
    return cv2.addWeighted(img0, 1 - alpha, img1_resized, alpha, 0)


def gamma_correction(img, gamma=1.22):
    """Apply inverse gamma correction"""
    img_norm = img / 255.0
    corrected = np.power(img_norm, 1 / gamma)
    return np.clip(corrected * 255, 0, 255).astype(np.uint8)


# --- Load inputs ---
img = cv2.imread("input_imgs/fruits.jpg")
skyline_img = cv2.imread("input_imgs/skyline.jpg")

# --- Run operations and save outputs ---
linear_output = linear_transform(img)
cv2.imwrite("output_imgs/linear_transform.jpg", linear_output)

spatial_output = spatial_transform(skyline_img)
cv2.imwrite("output_imgs/spatial_transform.jpg", spatial_output)

blended = blend_color(img, skyline_img, alpha=0.7)
cv2.imwrite("output_imgs/blended.jpg", blended)

gamma_corrected = gamma_correction(img)
cv2.imwrite("output_imgs/gamma_corrected.jpg", gamma_corrected)
