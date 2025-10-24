import sys
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
from _config import SCALE_FACTOR
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout,
    QGridLayout, QWidget, QLineEdit, QDialog, QScrollArea, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import func


# Can't get opencv working with PyQt5 on venv, so switched to opencv-headless.
# Instead displaying images in PyQt5 instead
def qpixmap_from_ndarray(img):
    """Convert NumPy image (BGR or grayscale) to QPixmap for PyQt display."""
    if img.ndim == 2:  # Grayscale
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    elif img.ndim == 3 and img.shape[2] == 3:  # BGR
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")



class ImageDialog(QDialog):
    """Auto-sized dialog showing scaled images (nearest-neighbor, pixel-perfect)."""
    def __init__(self, images_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image view")

        scroll = QScrollArea(self)  # have scroll
        scroll.setWidgetResizable(False)    # but not scale the images

        container = QWidget()
        layout = QVBoxLayout(container)

        total_height = 0
        max_width = 0

        for title, img in images_dict.items():
            t = QLabel(title)
            layout.addWidget(t)

            # we convert
            pixmap = qpixmap_from_ndarray(img)

            # Scale from _config if set
            scaled_pixmap = pixmap.scaled(
                int(pixmap.width() * SCALE_FACTOR),
                int(pixmap.height() * SCALE_FACTOR),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )

            lbl = QLabel()
            lbl.setPixmap(scaled_pixmap)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setScaledContents(False)
            layout.addWidget(lbl)

            max_width = max(max_width, scaled_pixmap.width())
            total_height += scaled_pixmap.height() + 40  # margin for title+spacing

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.addWidget(scroll)
        self.setLayout(outer)

        # Window size after all content
        content_w = max_width + 60
        content_h = total_height + 80

        # Set a max window size
        screen_geom = QApplication.primaryScreen().availableGeometry()
        window_w = min(content_w, int(screen_geom.width() * 0.95))
        window_h = min(content_h, int(screen_geom.height() * 0.95))

        self.resize(window_w, window_h)





# ---------------- Main Window ----------------
class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = [None] * 4
        self.image[3] = cv2.imread("Dataset/Q4_image/burger.png")
        self.setWindowTitle('Image Processing GUI')
        self.setGeometry(350, 500, 600, 500)

        # Layout Setup
        self.central_widget = QWidget()
        self.main_layout = QGridLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        # 2 "Load Image" buttons at 1/3 and 2/3 screen
        self.left_column_layout = QVBoxLayout()
        self.left_column_layout.addStretch(1)
        self.load_image1_button = QPushButton('Load Image 1', self)
        self.load_image1_button.clicked.connect(lambda: self.load_image(0))
        self.left_column_layout.addWidget(self.load_image1_button)
        self.left_column_layout.addStretch(1)
        self.load_image2_button = QPushButton('Load Image 2', self)
        self.load_image2_button.clicked.connect(lambda: self.load_image(1))
        self.left_column_layout.addWidget(self.load_image2_button)
        self.left_column_layout.addStretch(1)
        self.main_layout.addLayout(self.left_column_layout, 0, 0, 4, 1)


        # Q1. Image Processing
        self.image_processing_group = QGroupBox("1. Image Processing")
        self.image_processing_layout = QVBoxLayout()
        self.image_processing_group.setLayout(self.image_processing_layout)

        self.color_separation_button = QPushButton('1.1 Color Separation')
        self.image_processing_layout.addWidget(self.color_separation_button)
        self.color_separation_button.clicked.connect(
            lambda: self.show_result(func.image_color_seperation(self.image[0])) if self.image[0] is not None else None
        )

        self.color_transformation_button = QPushButton('1.2 Color Transformation')
        self.image_processing_layout.addWidget(self.color_transformation_button)
        self.color_transformation_button.clicked.connect(
            lambda: self.show_result(func.color_transformation(self.image[0])) if self.image[0] is not None else None
        )

        self.color_extraction_button = QPushButton('1.3 Color Extraction')
        self.image_processing_layout.addWidget(self.color_extraction_button)
        self.color_extraction_button.clicked.connect(
            lambda: self.show_result(func.color_extration(self.image[0])) if self.image[0] is not None else None
        )

        self.main_layout.addWidget(self.image_processing_group, 0, 1)
        self.main_layout.setRowMinimumHeight(1, 30)


        # Q2. Image Smoothing
        self.image_smoothing_group = QGroupBox("2. Image Smoothing")
        self.image_smoothing_layout = QVBoxLayout()
        self.image_smoothing_group.setLayout(self.image_smoothing_layout)

        self.gaussian_blur_button = QPushButton('2.1 Gaussian blur')
        self.image_smoothing_layout.addWidget(self.gaussian_blur_button)
        self.gaussian_blur_button.clicked.connect(
            lambda: self.show_result(func.gaussian_blur(self.image[0])) if self.image[0] is not None else None
        )

        self.bilateral_filter_button = QPushButton('2.2 Bilateral filter')
        self.image_smoothing_layout.addWidget(self.bilateral_filter_button)
        self.bilateral_filter_button.clicked.connect(
            lambda: self.show_result(func.bilateral_filter(self.image[0])) if self.image[0] is not None else None
        )

        # 不知道為啥 2.3 是 image 2
        self.median_filter_button = QPushButton('2.3 Median filter')
        self.image_smoothing_layout.addWidget(self.median_filter_button)
        self.median_filter_button.clicked.connect(
            lambda: self.show_result(func.median_filter(self.image[1])) if self.image[1] is not None else None
        )

        self.main_layout.addWidget(self.image_smoothing_group, 1, 1)
        self.main_layout.setRowMinimumHeight(1, 30)


        # Q3. Edge Detection
        self.edge_detection_group = QGroupBox("3. Edge Detection")
        self.edge_detection_layout = QVBoxLayout()
        self.edge_detection_group.setLayout(self.edge_detection_layout)

        self.sobel_x_button = QPushButton('3.1 Sobel X')
        self.edge_detection_layout.addWidget(self.sobel_x_button)
        self.sobel_x_button.clicked.connect(
            lambda: self.show_result(func.Sobel_x(self.image[0])) if self.image[0] is not None else None
        )

        self.sobel_y_button = QPushButton('3.2 Sobel Y')
        self.edge_detection_layout.addWidget(self.sobel_y_button)
        self.sobel_y_button.clicked.connect(
            lambda: self.show_result(func.Sobel_y(self.image[0])) if self.image[0] is not None else None
        )

        self.combination_threshold_button = QPushButton('3.3 Combination and Threshold')
        self.edge_detection_layout.addWidget(self.combination_threshold_button)
        self.combination_threshold_button.clicked.connect(
            lambda: self.show_result(func.combination_and_threshold(self.image[0])) if self.image[0] is not None else None
        )

        self.gradient_angle_button = QPushButton('3.4 Gradient Angle')
        self.edge_detection_layout.addWidget(self.gradient_angle_button)
        self.gradient_angle_button.clicked.connect(
            lambda: self.show_result(func.gradient_angle(self.image[0])) if self.image[0] is not None else None
        )

        self.main_layout.addWidget(self.edge_detection_group, 2, 1)
        self.main_layout.setRowMinimumHeight(1, 30)

        # Q4. Transforms
        self.transforms_group = QGroupBox("4. Transforms")
        self.transforms_layout = QVBoxLayout()
        self.transforms_group.setLayout(self.transforms_layout)

        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Rotation: deg")
        self.transforms_layout.addWidget(self.rotation_input)

        self.scaling_input = QLineEdit()
        self.scaling_input.setPlaceholderText("Scaling:")
        self.transforms_layout.addWidget(self.scaling_input)

        self.tx_input = QLineEdit()
        self.tx_input.setPlaceholderText("Tx: pixel")
        self.transforms_layout.addWidget(self.tx_input)

        self.ty_input = QLineEdit()
        self.ty_input.setPlaceholderText("Ty: pixel")
        self.transforms_layout.addWidget(self.ty_input)

        self.transforms_button = QPushButton('Transforms')
        self.transforms_layout.addWidget(self.transforms_button)
        self.transforms_button.clicked.connect(self.run_transform)

        self.main_layout.addWidget(self.transforms_group, 0, 2)
        self.main_layout.setRowMinimumHeight(1, 30)


        # Q5. Adaptive Threshold
        self.adaptive_threshold_group = QGroupBox("5. Adaptive Threshold")
        self.adaptive_threshold_layout = QVBoxLayout()
        self.adaptive_threshold_group.setLayout(self.adaptive_threshold_layout)

        self.global_threshold_button = QPushButton('5.1 Global Threshold')
        self.adaptive_threshold_layout.addWidget(self.global_threshold_button)
        self.global_threshold_button.clicked.connect(
            lambda: self.show_result(func.global_threshold(self.image[0])) if self.image[0] is not None else None
        )

        self.local_threshold_button = QPushButton('5.2 Local Threshold')
        self.adaptive_threshold_layout.addWidget(self.local_threshold_button)
        self.local_threshold_button.clicked.connect(
            lambda: self.show_result(func.local_threshold(self.image[0])) if self.image[0] is not None else None
        )

        self.main_layout.addWidget(self.adaptive_threshold_group, 1, 2)
        self.main_layout.setRowMinimumHeight(2, 30)



    def load_image(self, img_num):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open Image File', '',
            'Images (*.png *.xpm *.jpg *.bmp *.jpeg);;All Files (*)', options=options)
        if file_name:
            self.image[img_num] = cv2.imread(file_name)

    def show_result(self, result):
        if result is None:
            return
        if isinstance(result, dict):
            dlg = ImageDialog(result, self)
        else:
            dlg = ImageDialog({"Result": result}, self)
        dlg.exec_()

    # initials before calling func.transform
    def run_transform(self):
        if self.image[0] is None:
            return
        def get_float(text, default):
            try:
                return float(text)
            except ValueError:
                return default

        rot = get_float(self.rotation_input.text(), 0)

        center = (240, 200)

        scale = get_float(self.scaling_input.text(), 1.0)
        tx = get_float(self.tx_input.text(), 0) 
        ty = get_float(self.ty_input.text(), 0)

        result = func.transform(self.image[0], rot, center, scale, tx, ty)
        self.show_result(result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ImageProcessingApp()
    main_window.show()
    sys.exit(app.exec_())
