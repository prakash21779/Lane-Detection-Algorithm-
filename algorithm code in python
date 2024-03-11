import cv2
import numpy as np
def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough Transform to detect lines in the image
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

    # Draw detected lane lines on the original image
    lane_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combine the original image with the lane lines
    detected_lanes = cv2.addWeighted(image, 0.8, lane_image, 1, 0)

    return detected_lanes


# Read an input image
image_path = '\Java Programs\Projectstage\istockphoto-1254721353-612x612.jpg'
image = cv2.imread(image_path)

# Call the lane detection function
result = detect_lanes(image)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
