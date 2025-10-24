import cv2
import numpy as np

# --- Helpers ---
def _ensure_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# --- Q1 ------------------------------------------------
def image_color_seperation(img):
    b, g, r = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    b_image = cv2.merge([b, zeros, zeros])
    g_image = cv2.merge([zeros, g, zeros])
    r_image = cv2.merge([zeros, zeros, r])
    return {
        "Blue": _ensure_uint8(b_image),
        "Green": _ensure_uint8(g_image),
        "Red": _ensure_uint8(r_image),
    }

def color_transformation(img):
    b, g, r = cv2.split(img)
    cv_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_gray = (b / 3 + g / 3 + r / 3).astype(np.uint8)     # (b + g + r)/3 causes overflow
    return {
        "cv_gray": _ensure_uint8(cv_gray),
        "avg_gray": _ensure_uint8(avg_gray),
    }

def color_extration(img):
    lower_bound = np.array([18, 0, 25])
    upper_bound = np.array([85, 255, 255])
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    mask_inverse = cv2.bitwise_not(mask)
    extracted_image = cv2.bitwise_and(img, img, mask=mask_inverse)
    return {
        "The Yellow-Green Mask": _ensure_uint8(mask),
        "The Inverse Mask": _ensure_uint8(extracted_image),
    }

# --- Q2 ------------------------------------------------
def gaussian_blur(img):
    m = 5
    blur_img = img if m == 0 else cv2.GaussianBlur(img, (2*m+1, 2*m+1), 0, 0)
    return {"Gaussian blur": _ensure_uint8(blur_img)}

def bilateral_filter(img):
    m = 5
    d = 2 * m + 1
    sigmaColor, sigmaSpace = 90, 90
    bilateral_img = img if m == 0 else cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    return {"Bilateral Filter": _ensure_uint8(bilateral_img)}

def median_filter(img):
    m = 5
    median_img = img if m == 0 else cv2.medianBlur(img, 2*m + 1)
    return {"Median Filter": _ensure_uint8(median_img)}

# --- Q3 ------------------------------------------------
def blur(img, m, sigmaX, sigmaY):
    if m == 0:
        return img
    kernel_size = 2 * m + 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX, sigmaY)

def sobel_edge_detection(img, sobel):
    height, width = img.shape
    sobel_image = np.zeros_like(img, dtype=np.float32)
    padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            region = padded[i - 1:i + 2, j - 1:j + 2]
            sobel_image[i - 1, j - 1] = np.sum(sobel * region)
    sobel_image = np.abs(sobel_image)
    sobel_image = (sobel_image / (np.max(sobel_image) + 1e-8)) * 255.0
    return sobel_image.astype(np.uint8)

def Sobel_x(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = blur(gray, 1, 0, 0)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_x_image = sobel_edge_detection(blur_img, sobel_x)
    return {
        "Sobel X Image": _ensure_uint8(sobel_x_image),
    }

def Sobel_y(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = blur(gray, 1, 0, 0)
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)
    sobel_y_image = sobel_edge_detection(blur_img, sobel_y)
    return {
        "Sobel Y Image": _ensure_uint8(sobel_y_image),
    }

def combination_and_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = blur(gray, 1, 0, 0)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    sx = sobel_edge_detection(blur_img, sobel_x)
    sy = sobel_edge_detection(blur_img, sobel_y)

    combo = np.sqrt(sx.astype(np.float32)**2 + sy.astype(np.float32)**2)
    combo = (combo / (combo.max() + 1e-8) * 255.0).astype(np.uint8) # 1e-8 prevents dividing by 0

    normalized = cv2.normalize(combo, None, 0, 255, cv2.NORM_MINMAX)
    _, result128 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
    _, result28 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)

    return {
        "Combination": _ensure_uint8(combo),
        "Threshold=128": _ensure_uint8(result128),
        "Threshold=28": _ensure_uint8(result28),
    }

def gradient_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = blur(gray, 2, 0, 0)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    sx = sobel_edge_detection(blur_img, sobel_x).astype(np.float32)
    sy = sobel_edge_detection(blur_img, sobel_y).astype(np.float32)

    gradient_angle = (np.arctan2(sy, sx) * (180 / np.pi) + 180) # +180 shifts to [0, 360]

    mask1 = np.zeros_like(gradient_angle, dtype=np.uint8)
    mask2 = np.zeros_like(gradient_angle, dtype=np.uint8)
    mask1[(gradient_angle >= 170) & (gradient_angle <= 190)] = 255
    mask2[(gradient_angle >= 260) & (gradient_angle <= 280)] = 255

    result1 = cv2.bitwise_and(sx.astype(np.uint8), mask1)
    result2 = cv2.bitwise_and(sy.astype(np.uint8), mask2)

    return {
        "Result of 170~190": _ensure_uint8(result1),
        "Result of 260~280": _ensure_uint8(result2),
    }

# --- Q4 ------------------------------------------------
def transform(img, rotation, center, scaling, Tx, Ty):
    (height, width) = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, rotation, scaling)
    M[0, 2] += Tx
    M[1, 2] += Ty
    result = cv2.warpAffine(img, M, (width, height))
    return {"Result": _ensure_uint8(result)}


# --- Q5 ------------------------------------------------
def global_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    return {"Original": _ensure_uint8(img), "Threshold image": _ensure_uint8(threshold_image)}


def local_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1)
    return {"Original": _ensure_uint8(img), "Adaptive threshold image": _ensure_uint8(threshold_image)}

