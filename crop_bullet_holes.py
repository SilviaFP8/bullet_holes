import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import os

class BulletHoleProcessor:
    def __init__(self, min_hole_size: int = 2000, max_hole_size: int = 1500000):
        """
        Initialize bullet hole processor.
        
        Args:
            min_hole_size: Minimum hole size in pixels
            max_hole_size: Maximum hole size in pixels
        """
        self.min_hole_size = min_hole_size
        self.max_hole_size = max_hole_size
        
    def detect_hole(self, image_path: str, 
                    edge_method: str = 'sobel', 
                    # Sobel/Laplacian parameters
                    sobel_ksize: int = 3,
                    laplacian_ksize: int = 3,
                    # Canny parameters
                    canny_thresh1: int = 30,
                    canny_thresh2: int = 90,
                    # Adaptive threshold parameters
                    adaptive_block: int = 11,
                    adaptive_C: int = 2,
                    # Threshold for Sobel/Laplacian (if None, use Otsu)
                    binary_thresh: Optional[int] = 25,
                    # Median filter parameter
                    median_kernel_size: int = 0   # (0 = no median filter)
                   ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect bullet hole in an image.
        
        Returns:
            Tuple of (image, bounding_rect) or None if no hole found
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        # Apply median filter first if requested
        if median_kernel_size > 1 and median_kernel_size % 2 == 1:
            gray = cv2.medianBlur(gray, median_kernel_size)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- Generate binary image according to chosen method ---
        if edge_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C
            )

        elif edge_method == 'canny':
            binary = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        elif edge_method == 'sobel':
            # Compute gradient magnitude
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            mag = np.sqrt(sobel_x**2 + sobel_y**2)
            mag = np.uint8(np.clip(mag, 0, 255))
            # Threshold to get binary edges
            if binary_thresh is None:
                # Use Otsu's automatic threshold
                _, binary = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(mag, binary_thresh, 255, cv2.THRESH_BINARY)

        elif edge_method == 'laplacian':
            lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=laplacian_ksize)
            lap = np.uint8(np.clip(np.abs(lap), 0, 255))
            if binary_thresh is None:
                _, binary = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, binary = cv2.threshold(lap, binary_thresh, 255, cv2.THRESH_BINARY)

        else:
            raise ValueError(f"Unknown edge_method: {edge_method}. Choose from 'adaptive','canny','sobel','laplacian'.")

        # Optional: morphological closing to connect broken edges
        # (helps especially for Sobel/Laplacian)
        # kernel = np.ones((3,3), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No contours found with method '{edge_method}'")
            return None
        
        # Find the largest contour that fits our size criteria
        largest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_hole_size < area < self.max_hole_size:
                if area > max_area:
                    max_area = area
                    largest_contour = contour
        
        if largest_contour is None:
            print("No suitable hole found")
            return None
        
        # Get bounding rectangle
        x, y, w_0, h_0 = cv2.boundingRect(largest_contour)

        # Get image dimensions
        height, width = image.shape[:2]
        # Make bounding square
        w = np.min([width, w_0])
        h = np.min([height, h_0])
        if w > h:
            diff_dist = int(np.floor((w - h) / 2))
            y = np.max([y - diff_dist, 0])
            h = np.min([w, height])
        if h > w:
            diff_dist = int(np.floor((h - w) / 2))
            x = np.max([x - diff_dist, 0])
            w = np.min([h, width])
        else:
            w = w_0
            h = h_0
        return image, (x, y, w, h)
    
    def crop_and_square_hole(self, image_path: str, output_path: str = None, 
                           padding: int = 10, visualize: bool = False) -> Optional[np.ndarray]:
        """
        Detect, crop, and square the bullet hole.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            padding: Extra padding around the hole
            visualize: Show intermediate steps
            
        Returns:
            Squared hole image or None if processing failed
        """
        # Detect hole
        result = self.detect_hole(image_path)
        if result is None:
            return None
            
        image, (x, y, w, h) = result
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)
        
        # Crop the hole region
        hole_crop = image[y:y+h, x:x+w]
        
        # Make it square by taking the larger dimension
        max_dim = max(w, h)
        
        # Create a black square canvas
        if len(hole_crop.shape) == 3:  # Color image
            square_hole = np.zeros((max_dim, max_dim, 3), dtype=np.uint8) + 255
        else:  # Grayscale
            square_hole = np.zeros((max_dim, max_dim), dtype=np.uint8) + 255
        
        # Calculate position to paste the cropped hole in the center
        if w > h:
            # Hole is wider than tall (landscape)
            y_offset = (max_dim - h) // 2
            square_hole[y_offset:y_offset+h, 0:w] = hole_crop
        elif h > w:
            # Hole is taller than wide (portrait)
            x_offset = (max_dim - w) // 2
            square_hole[0:h, x_offset:x_offset+w] = hole_crop
        else:
            # Already square
            square_hole = hole_crop
        
        # Optional: Apply border to see the extent
        if visualize:
            square_hole = cv2.copyMakeBorder(
                square_hole, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 255]
            )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, square_hole)
            print(f"Saved squared hole to: {output_path}")
        
        return square_hole
    
    def process_multiple_images(self, input_dir: str, output_dir: str, 
                               padding: int = 10) -> None:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            padding: Extra padding around holes
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for img_file in input_path.iterdir():
            if img_file.suffix.lower() in supported_formats:
                print(f"Processing: {img_file.name}")
                
                output_file = output_path / f"squared_{img_file.name}"
                
                result = self.crop_and_square_hole(
                    str(img_file), 
                    str(output_file),
                    padding=padding,
                    visualize=False
                )
                
                if result is not None:
                    print(f"  ✓ Success: {img_file.name}")
                else:
                    print(f"  ✗ Failed: {img_file.name}")


def visualize_process(image_path: str):
    """
    Visualize the entire processing pipeline for a single image.
    """
    processor = BulletHoleProcessor()
    
    # Read and display original
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect hole
    result = processor.detect_hole(image_path)
    if result is None:
        print("No hole detected!")
        return
    
    image, (x, y, w, h) = result
    
    # Draw rectangle on original
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
    img_with_rect_rgb = cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB)
    
    # Get squared hole
    squared_hole = processor.crop_and_square_hole(
        image_path, 
        padding=50,
        visualize=False
    )
    
    if squared_hole is None:
        return
    
    # Convert to RGB for display
    if len(squared_hole.shape) == 3:
        squared_hole_rgb = cv2.cvtColor(squared_hole, cv2.COLOR_BGR2RGB)
    else:
        squared_hole_rgb = squared_hole
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(img_with_rect_rgb)
    axes[1].set_title('Detected Hole')
    axes[1].axis('off')
    
    axes[2].imshow(squared_hole_rgb)
    axes[2].set_title('Squared Hole')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_median_filter(image_path: str, kernel_size: int = 3, sobel_ksize: int = 3, binary_thresh: Optional[int] = 100):
    """
    Display the original grayscale image and the image after median filtering.
    
    Parameters:
        image_path: path to the input image (tilde ~ is expanded automatically)
        kernel_size: size of the median filter kernel (must be odd, e.g., 3, 5, 7)
    """
    # Expand user home directory if needed
    image_path = os.path.expanduser(image_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Convert to grayscale (median filter is often applied to grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel_x1 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y1 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
    mag1 = np.uint8(np.clip(mag1, 0, 255))
    # Threshold to get binary edges
    if binary_thresh is None:
        # Use Otsu's automatic threshold
        _, binary1 = cv2.threshold(mag1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary1 = cv2.threshold(mag1, binary_thresh, 255, cv2.THRESH_BINARY)
    
    # Apply median filter
    filtered = cv2.medianBlur(gray, kernel_size)
    
    sobel_x2 = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y2 = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
    mag2 = np.uint8(np.clip(mag2, 0, 255))
    # Threshold to get binary edges
    if binary_thresh is None:
        # Use Otsu's automatic threshold
        _, binary2 = cv2.threshold(mag2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary2 = cv2.threshold(mag2, binary_thresh, 255, cv2.THRESH_BINARY)
    
    blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
    
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    mag = np.uint8(np.clip(mag, 0, 255))
    # Threshold to get binary edges
    if binary_thresh is None:
        # Use Otsu's automatic threshold
        _, binary = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(mag, binary_thresh, 255, cv2.THRESH_BINARY)
    
    
    # Display side by side
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title(f'Original Grayscale')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title(f'Median Filter (k={kernel_size})')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(blurred, cmap='gray')
    plt.title(f'Smoothed (k={(5,5)})')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(binary1, cmap='gray')
    plt.title(f'Sobel Org. Edges')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(binary2, cmap='gray')
    plt.title(f'Sobel Md. Edges')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(binary, cmap='gray')
    plt.title(f'Sobel Sm. Edges')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = BulletHoleProcessor()
    
    # Process a single image    
    class1_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_1/15gr_1.jpg"
    class1_plate = os.path.expanduser(class1_plate)
    #class2_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_2/30gr_6.jpg"
    class2_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_2/30gr_9.jpg"
    #class2_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_2/30gr_10_.jpg"
    class2_plate = os.path.expanduser(class2_plate)
    class3_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_3/45gr_9_.jpg"
    class3_plate = os.path.expanduser(class3_plate)
    class4_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_4/60gr_1.jpg"
    class4_plate = os.path.expanduser(class4_plate)
    class5_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_5/75gr_25_.jpg"
    class5_plate = os.path.expanduser(class5_plate)
    class6_plate = "~/Documentos/Silvia_FP/original_1800_fotos/dataset3/class_6/90gr_1.jpg"
    class6_plate = os.path.expanduser(class6_plate)
    
    class1_drywall = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_1/IMG_9840.png"
    class1_drywall = os.path.expanduser(class1_drywall)
    class2_drywall = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_2/IMG_9424.png"
    class2_drywall = os.path.expanduser(class2_drywall)
    class3_drywall = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_3/IMG_9090.png"
    class3_drywall = os.path.expanduser(class3_drywall)
    class4_drywall = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_4/IMG_8723.png"
    class4_drywall = os.path.expanduser(class4_drywall)
    class5_drywall = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_5/IMG_8407.png"
    class5_drywall = os.path.expanduser(class5_drywall)
    class6_drywall = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_6/IMG_7993.png"
    class6_drywall = os.path.expanduser(class6_drywall)
    
    class1_particleboard = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_1/IMG_9605.png"
    class1_particleboard = os.path.expanduser(class1_particleboard)
    class2_particleboard = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_2/IMG_9256.png"
    class2_particleboard = os.path.expanduser(class2_particleboard)
    class3_particleboard = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_3/IMG_8928.png"
    class3_particleboard = os.path.expanduser(class3_particleboard)
    class4_particleboard = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_4/IMG_8557.png"
    class4_particleboard = os.path.expanduser(class4_particleboard)
    class5_particleboard = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_5/IMG_8228.png"
    class5_particleboard = os.path.expanduser(class5_particleboard)
    class6_particleboard = "~/Documentos/Silvia_FP/original_1800_fotos/dataset/class_6/IMG_7827.png"
    class6_particleboard = os.path.expanduser(class6_particleboard)
    
    # Method 1: Basic processing
    # result = processor.crop_and_square_hole(
    #     image_path=input_image,
    #     output_path=output_image,
    #     padding=15,  # Add 15 pixels padding around the hole
    #     visualize=False
    # )
    
    # if result is not None:
    #     print("Successfully processed the hole!")
    
    # Method 2: Process a directory of images
    # processor.process_multiple_images(
    #     input_dir="./input_images/",
    #     output_dir="./output_images/",
    #     padding=10
    # )
    
    # Method 3: Visualize the process
    visualize_process(class2_plate)
    #compare_median_filter(class5_drywall)
