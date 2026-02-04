import cv2
import numpy as np
import math

def generate_tag(cell_size=50, tag_id=0):
    """
    Generate an AR tag image with the specified ID.
    """
    # Initialize an 8x8 black grid (0 = black)
    # The 2-cell outer border is already black by default
    grid = np.zeros((8, 8), dtype=np.uint8)
    
    # Define the internal 4x4 grid (Indices 2 to 5)
    # Row 2
    grid[2, 2] = 0
    grid[2, 3] = 255
    grid[2, 4] = 255
    grid[2, 5] = 0
    
    # Row 3
    grid[3, 2] = 255
    grid[3, 3] = 255  # ID Bit 1
    grid[3, 4] = 0  # ID Bit 2
    grid[3, 5] = 255
    
    # Row 4
    grid[4, 2] = 255
    grid[4, 3] = 255  # ID Bit 4
    grid[4, 4] = 255  # ID Bit 3
    grid[4, 5] = 255
    
    # Row 5
    grid[5, 2] = 255
    grid[5, 3] = 255
    grid[5, 4] = 255
    grid[5, 5] = 0

    # Scale the 8x8 grid to a visible image size
    tag_image = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    
    cv2.imwrite(f"Tag{tag_id}.png", tag_image)

    return tag_image

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame.

    Args:
        img: The current video frame.
        obj: The loaded OBJ model.
        projection: The 3D projection matrix.
        model: The reference image representing the surface to be augmented.
        color: Whether to render in color. Defaults to False.
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img

import numpy as np

def draw_line(image, start, end, color=(0, 255, 0)):
    """
    Draws a line on a numpy array image from start to end.
    Uses a vectorized Bresenham-like approach for efficiency.
    """
    x1, y1 = start
    x2, y2 = end
    
    # Calculate distance to determine number of points
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0: return

    # Generate coordinate indices along the line
    x_indices = np.linspace(x1, x2, length).astype(int)
    y_indices = np.linspace(y1, y2, length).astype(int)

    # Boundary checks to prevent crashing at edges
    h, w = image.shape[:2]
    valid = (x_indices >= 0) & (x_indices < w) & (y_indices >= 0) & (y_indices < h)
    
    # Apply color to those pixels
    image[y_indices[valid], x_indices[valid]] = color


def detect_edges_binary(binary_image):
    """
    Detects edges in a binary image using Morphological Gradient.
    Method: Edge = Original - Eroded
    Speed: Vectorized NumPy (Extremely Fast)
    
    Args:
        binary_image: 2D NumPy array with values 0 (Black) and 255 (White).
    
    Returns:
        edge_image: Binary image where edges are 255 and background is 0.
    """
    # 1. Convert to Boolean (0/1) for fast bitwise operations
    # Assumes object is White (255) and Background is Black (0)
    bool_img = (binary_image == 255)
    
    # 2. Perform Erosion (Shrink the white shapes by 1 pixel)
    # A pixel survives erosion only if ALL its 4-neighbors are also White.
    
    # Pad with False (Black) to handle borders automatically
    padded = np.pad(bool_img, pad_width=1, mode='constant', constant_values=False)
    
    # Shift padded image to align neighbors
    north = padded[0:-2, 1:-1]
    south = padded[2:,   1:-1]
    west  = padded[1:-1, 0:-2]
    east  = padded[1:-1, 2:]
    
    # Vectorized AND: Pixel is TRUE only if it and all neighbors are TRUE
    eroded = north & south & west & east & bool_img
    
    # 3. Subtract Eroded from Original
    # The difference are the pixels that were White but had a Black neighbor.
    # These are exactly the edge pixels.
    # logical_xor works like subtraction here because eroded is a subset of bool_img
    edges_bool = bool_img ^ eroded 
    
    # 4. Convert back to Uint8 (0-255)
    edge_image = (edges_bool.astype(np.uint8)) * 255
    
    return edge_image

def extract_and_draw_final(frame, resizing_factor=1):
    print(f"--- Processing Frame (Scale: 1/{resizing_factor}) ---")
    
    # A. PREPROCESSING
    # Downscale for speed (Processing 1080p is unnecessary for detection)
    small_frame = fast_scale(frame, scale_factor=resizing_factor)
    grey = to_grayscale(small_frame)
    
    # Gaussian Blur (or use fast_box_blur if you implemented it)
    blurred = gaussian_blur(grey, kernel_size=9, sigma=2.0)

    # B. BINARIZATION
    # Using your optimized global/Otsu thresholding
    binary = binarization(blurred)
    
    gradient =  detect_edges_binary(binary)
    # C. EXTRACT CONTOURS (Optimized Flat Version)
    # The function you defined now uses 'visited_map' and candidate selection.
    # It returns a simple list of arrays, so we don't need a hierarchy stack.
    print("DEBUG: Extracting contours...")
    candidate_contours = extract_contours_from_gradient(gradient)
    print(f"DEBUG: Found {len(candidate_contours)} total contours.")

    # return gradient
    # return binary    
    # D. GEOMETRIC FILTERING
    # Adjust thresholds based on resizing. 
    # If we want to detect the small data bits inside the tags, min_area must be small.
    min_area_thresh = 50 if resizing_factor == 1 else 10
    
    # (Assuming you have a filter_contours function, or use list comprehension)
    # valid_tags = [c for c in candidate_contours if cv2.contourArea(c) > min_area_thresh]
    
    # For now, using your placeholder:
    valid_tags = optimize_contours(candidate_contours, min_area=min_area_thresh)
    # valid_tags = candidate_contours
    print(f"DEBUG: Filtered down to {len(valid_tags)} valid shapes.")

    # E. DRAWING (RED FILL)
    output_frame = frame.copy()
    red_color = (0, 0, 255) # BGR
    
    for tag in valid_tags:
        # 'tag' is a numpy array of coordinates
        
        # 1. Upscale coordinates to match original 1080p frame
        upscaled_tag = tag * resizing_factor
        
        # 2. Convert to standard OpenCV format: (N, 1, 2)
        # FLIP [row, col] -> [x, y] for OpenCV drawing functions
        pts = upscaled_tag[:, ::-1].astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # 3. Draw the shape
        cv2.polylines(output_frame, [pts], isClosed=True, color=red_color, thickness=5)
            
    return output_frame


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur using 1D separable convolution with vectorization.
    Speedup: ~100x compared to pixel-wise loops.
    """
    # 1. Create 1D Gaussian Kernel
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    kernel_1d = np.exp(-(ax**2) / (2. * sigma**2))
    kernel_1d /= kernel_1d.sum()
    
    # 2. Prepare for Vectorized Convolution
    h, w = image.shape
    pad = kernel_size // 2
    
    # We work with floats for precision
    img_float = image.astype(np.float32)
    
    # --- Horizontal Pass ---
    # Pad Left/Right only
    padded_h = np.pad(img_float, ((0, 0), (pad, pad)), mode='edge')
    blurred_h = np.zeros_like(img_float)
    
    # Vectorized "Shift & Add"
    # Instead of looping pixels, we loop through the kernel elements.
    # We take slices of the padded image corresponding to the shift.
    for k in range(kernel_size):
        # Slice the padded image to get a shifted view of size (H, W)
        # If k=0, we take the leftmost slice. If k=kernel_size-1, the rightmost.
        shifted_view = padded_h[:, k : k + w]
        
        # Accumulate: Result += Image_Slice * Kernel_Weight
        blurred_h += shifted_view * kernel_1d[k]

    # --- Vertical Pass ---
    # Pad Top/Bottom only (on the result of the horizontal pass)
    padded_v = np.pad(blurred_h, ((pad, pad), (0, 0)), mode='edge')
    blurred_v = np.zeros_like(blurred_h)
    
    for k in range(kernel_size):
        # Slice vertically
        shifted_view = padded_v[k : k + h, :]
        blurred_v += shifted_view * kernel_1d[k]
        
    return blurred_v.astype(np.uint8)



def binarization(gray_image):
    """
    Otsu's Binarization (Automatic Global Thresholding).
    Calculates the optimal threshold by maximizing inter-class variance.
    
    Args:
        gray_image: Grayscale input image (2D numpy array).
        resizing_factor, window_size, t: Ignored (kept for compatibility).
    """
    h, w = gray_image.shape
    total_pixels = h * w
    
    # 1. Compute Histogram (Frequency of each pixel value 0-255)
    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
    
    # Normalize histogram to get probabilities
    # Use float32 to prevent overflow and enable precise division
    hist_norm = hist.astype(np.float32) / total_pixels
    
    # 2. Compute Cumulative Sums (Weight background w0) and Means
    # w0: Probability that a pixel is <= threshold k
    w0 = np.cumsum(hist_norm)
    
    # w1: Probability that a pixel is > threshold k
    w1 = 1.0 - w0
    
    # Cumulative sum of intensities (weighted sum)
    # used to calculate means mu0 and mu1
    bin_centers = np.arange(256, dtype=np.float32)
    cumulative_intensity_sum = np.cumsum(bin_centers * hist_norm)
    
    # Global mean intensity of the whole image
    global_mean = cumulative_intensity_sum[-1]
    
    # 3. Compute Between-Class Variance for every possible threshold
    # Formula: sigma^2 = w0 * w1 * (mu0 - mu1)^2
    # Simplified vectorized form: (global_mean * w0 - cumulative_intensity_sum)^2 / (w0 * w1)
    
    numerator = (global_mean * w0 - cumulative_intensity_sum) ** 2
    denominator = w0 * w1
    
    # Handle division by zero (where w0 or w1 is 0, usually at the ends)
    denominator[denominator == 0] = 1e-10
    
    variance = numerator / denominator
    
    # 4. Find the Optimal Threshold (Maximize Variance)
    optimal_threshold = np.argmax(variance)
    
    # 5. Apply Threshold
    # Target Bright Objects (Paper) -> White (255), Background -> Black (0)
    binary_image = np.zeros((h, w), dtype=np.uint8)
    binary_image[gray_image > optimal_threshold] = 255

    close_gaps(binary_image, iterations=2)

    return binary_image



def to_grayscale(image):
    """
    Convert a BGR image to Grayscale using standard luminosity weights.
    
    Args:
        image: Input image array of shape (H, W, 3) in BGR format.
    
    Returns:
        gray_image: Grayscale image of shape (H, W) type uint8.
    """
    # 1. Extract Channels
    # OpenCV loads images as BGR
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    # 2. Apply Weighted Sum (Luminosity Method)
    # Weights match human perception (Green is brightest, Blue is darkest)
    # We use floating point math for precision, then cast back to uint8
    gray_float = (0.114 * blue_channel) + (0.587 * green_channel) + (0.299 * red_channel)
    
    # 3. Cast to uint8
    # This truncates the float values to integers (0-255)
    gray_image = gray_float.astype(np.uint8)
    
    return gray_image

def fast_scale(image, scale_factor):
    """
    Downscales an image by skipping pixels (Nearest Neighbor).
    Extremely fast because it uses memory slicing instead of math.
    
    Args:
        image: Input image (2D or 3D numpy array).
        scale_factor: Integer (e.g., 2 for half size, 4 for quarter size).
        
    Returns:
        scaled_image: The downsized image.
    """
    # Slicing syntax: [start:stop:step]
    # We take every n-th pixel in both rows and columns.
    scaled_image = image[::scale_factor, ::scale_factor]
    
    return scaled_image


def minimum_filter(image):
    """
    Expands Black regions (0) to connect broken lines.
    Equivalent to Morphological EROSION on White background.
    """
    # Pad image with White (255) so edges don't become artificially black
    padded = np.pad(image, 1, mode='constant', constant_values=255)
    
    # Create views for neighbors (Up, Down, Left, Right)
    center = padded[1:-1, 1:-1]
    top    = padded[0:-2, 1:-1]
    bottom = padded[2:,   1:-1]
    left   = padded[1:-1, 0:-2]
    right  = padded[1:-1, 2:]
    
    # The output is 0 if ANY neighbor is 0.
    # We chain np.minimum to find the darkest value in the cross pattern.
    min_img = np.minimum(center, top)
    min_img = np.minimum(min_img, bottom)
    min_img = np.minimum(min_img, left)
    min_img = np.minimum(min_img, right)
    
    return min_img

def maximum_filter(image):
    """
    Expands White regions (255) to shave off noise.
    Equivalent to Morphological DILATION on White background.
    """
    # Pad with Black (0) so edges don't become artificially white
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    
    center = padded[1:-1, 1:-1]
    top    = padded[0:-2, 1:-1]
    bottom = padded[2:,   1:-1]
    left   = padded[1:-1, 0:-2]
    right  = padded[1:-1, 2:]
    
    # The output is 255 if ANY neighbor is 255.
    max_img = np.maximum(center, top)
    max_img = np.maximum(max_img, bottom)
    max_img = np.maximum(max_img, left)
    max_img = np.maximum(max_img, right)
    
    return max_img

def close_gaps(binary_image, iterations=2):
    """
    Stronger closing.
    Args:
        iterations: How many times to expand. 
                    Increase this to 3 or 4 if gaps persist.
    """
    temp_image = binary_image.copy()
    
    # 1. Expand Black multiple times (Bridge the gaps)
    for _ in range(iterations):
        temp_image = minimum_filter(temp_image)
        
    # 2. Shrink Black back (Restore shape)
    # We run this fewer times or the same amount. 
    # Usually, we want to keep it slightly "fat" to ensure connectivity.
    for _ in range(iterations):
        temp_image = maximum_filter(temp_image)
        
    return temp_image

def morphological_opening(binary_image, iterations=1):
    """
    Removes small noise (Opening = Erosion -> Dilation).
    Assumes Background is White (255) and Objects are Black (0).
    """
    img = binary_image.copy()
    
    # 1. Erode Black (Expand White) - Kills small black specks
    # We use 'maximum_filter' because max(0, 255) = 255 (White wins)
    for _ in range(iterations):
        img = maximum_filter(img)
        
    # 2. Dilate Black (Expand Black) - Restores shape of surviving objects
    # We use 'minimum_filter' because min(0, 255) = 0 (Black wins)
    for _ in range(iterations):
        img = minimum_filter(img)
        
    return img

# --- Helpers needed if you don't have them ---
def maximum_filter(image):
    # Expands White (255) -> Erodes Black
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    # Checks 5-point cross pattern
    p0 = padded[1:-1, 1:-1]
    p1 = padded[0:-2, 1:-1] # Top
    p2 = padded[2:,   1:-1] # Bottom
    p3 = padded[1:-1, 0:-2] # Left
    p4 = padded[1:-1, 2:]   # Right
    return np.maximum.reduce([p0, p1, p2, p3, p4])

def minimum_filter(image):
    # Expands Black (0) -> Dilates Black
    padded = np.pad(image, 1, mode='constant', constant_values=255)
    p0 = padded[1:-1, 1:-1]
    p1 = padded[0:-2, 1:-1]
    p2 = padded[2:,   1:-1]
    p3 = padded[1:-1, 0:-2]
    p4 = padded[1:-1, 2:] 
    return np.minimum.reduce([p0, p1, p2, p3, p4])

def extract_contours_from_gradient(edge_image):
    contours = []
    h, w = edge_image.shape
    visited_mask = np.zeros((h, w), dtype=bool)
    
    # Get all white pixels
    candidate_points = np.argwhere(edge_image == 255)

    for start_point in candidate_points:
        y, x = start_point
        if visited_mask[y, x]: continue
            
        # Trace line
        contour = trace_line_directional(edge_image, tuple(start_point), visited_mask)
        
        if len(contour) > 10:
            contours.append(contour)
            
    return contours

def trace_line_directional(edge_image, start_point, visited_mask):
    """
    Traces a 1-pixel thick line by prioritizing neighbors that continue 
    the current direction, preventing diagonal short-circuits.
    """
    h, w = edge_image.shape
    contour = []
    curr_y, curr_x = start_point
    
    contour.append([curr_y, curr_x])
    visited_mask[curr_y, curr_x] = True
    
    # Initial arbitrary direction (doesn't matter for first step)
    # 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    current_dir = 0 
    
    # Standard 8-connected offsets
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    max_steps = 10000
    steps = 0
    
    while True:
        steps += 1
        if steps > max_steps: break
            
        found_next = False
        
        # SEARCH STRATEGY: 
        # Start looking from the "Forward-Left" neighbor relative to current direction.
        # This acts like a "Left-Hand Rule" wall follower.
        # It ensures we follow the outer edge of the line.
        
        # If moving North (0), we check NW(7), N(0), NE(1)...
        start_search_idx = (current_dir + 6) % 8 
        
        for i in range(8):
            # Check neighbors in Clockwise order
            idx = (start_search_idx + i) % 8
            dy, dx = offsets[idx]
            ny, nx = curr_y + dy, curr_x + dx
            
            if 0 <= ny < h and 0 <= nx < w:
                # Is it part of the line?
                if edge_image[ny, nx] == 255:
                    
                    # 1. New Pixel?
                    if not visited_mask[ny, nx]:
                        curr_y, curr_x = ny, nx
                        contour.append([curr_y, curr_x])
                        visited_mask[curr_y, curr_x] = True
                        current_dir = idx # Update direction
                        found_next = True
                        break
                    
                    # 2. Closed Loop?
                    # Only close if we hit start point AND line is long enough
                    elif ny == start_point[0] and nx == start_point[1] and len(contour) > 10:
                        return np.array(contour)
        
        if not found_next:
            break
            
    return np.array(contour)


def optimize_contours(contours, min_area=100):
    """
    Filters and simplifies a list of contours.
    1. Removes small noise (Area Check).
    2. Simplifies shapes to stable 4-corner quadrilaterals (Fixes Flickering).
    
    Args:
        contours: List of numpy arrays (contours).
        min_area: Minimum area required to be considered a valid tag.
        
    Returns:
        valid_quads: List of simplified (4, 2) numpy arrays.
    """
    valid_quads = []
    
    for cnt in contours:
        # 1. Area Filter
        # Calculate area (Shoelace formula is fast, or use cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # 2. Simplification (Robust Quad Extraction)
        # Instead of RDP (which flickers), we find the best-fit 4 corners on the Convex Hull.
        try:
            # Helper function from previous step
            hull = get_convex_hull(cnt) 
            quad = find_largest_quad_on_hull(hull)
            
            # 3. Final sanity check: Is the result actually 4 points?
            if len(quad) == 4:
                valid_quads.append(quad)
                
        except Exception as e:
            # If a contour is degenerate (e.g. a straight line), hull might fail.
            # Safe to skip it.
            continue
            
    return valid_quads


def simplify_to_quad(contour):
    """
    Robustly extracts exactly 4 corners from a contour for AR tags.
    Fixes flickering by avoiding recursive splitting.
    
    Args:
        contour: (N, 2) numpy array of (y, x) coordinates.
        
    Returns:
        quad: (4, 2) numpy array of corners sorted (TL, TR, BR, BL).
    """
    # 1. Get Convex Hull
    # This wraps the noisy contour in a rubber band, smoothing out small dents.
    # Note: If you don't have scipy, you can implement Monotone Chain algo, 
    # but for simplicity, let's assume valid input or use a simple extents method.
    hull = get_convex_hull(contour) 
    
    # 2. Heuristic: Find the "Extreme" points
    # For a rotated square, the corners are usually the points that maximize:
    # (x + y), (x - y), (-x + y), (-x - y)
    
    # Convert to x, y for easier reasoning (assuming input is y, x)
    # y = hull[:, 0], x = hull[:, 1]
    pts = hull
    
    # Sum and Diff
    s = pts.sum(axis=1) # y + x
    d = np.diff(pts, axis=1).flatten() # x - y (approx)
    
    # Top-Left: Minimal sum (closest to 0,0)
    tl = pts[np.argmin(s)]
    # Bottom-Right: Maximal sum
    br = pts[np.argmax(s)]
    
    # Top-Right: Minimal difference (y is small, x is big -> y-x is small negative)
    # Bottom-Left: Maximal difference (y is big, x is small -> y-x is big positive)
    # Note: This heuristic works well for rectangles not rotated 45 degrees.
    # A more robust generic way is closest distance to frame corners or rotated calipers.
    
    # --- BETTER GENERIC METHOD (Rotated Calipers approx) ---
    # We want the 4 points that maximize area or distance.
    # Simplified approach for AR: Approximate the polygon to 4 sides.
    
    # Calculate center of mass
    center = contour.mean(axis=0)
    
    # Calculate distance of every point from center
    dists = np.linalg.norm(contour - center, axis=1)
    
    # The corners are local maxima of distance from center.
    # However, noise creates many local maxima.
    # We essentially want to cluster the points into 4 groups (quadrants) 
    # and pick the furthest point in each group.
    
    corners = []
    
    # Classify points into 4 quadrants relative to center
    # Q1: Top-Left (y < cy, x < cx)
    # Q2: Top-Right (y < cy, x > cx) ... etc
    # (Note: This assumes the tag isn't rotated 45 degrees relative to camera)
    
    # ROBUST FALLBACK:
    # If the tag rotates, quadrant logic fails.
    # Instead, just return the 4 points from the convex hull 
    # that form the largest quadrilateral area.
    
    # But since you asked for a replacement for 'simplify_contour',
    # here is the standard OpenCV-style approximation implemented manually:
    
    return find_largest_quad_on_hull(hull)

def get_convex_hull(points):
    """
    Computes Convex Hull using Monotone Chain algorithm.
    Sorts points lexicographically first.
    """
    # Sort by y, then x
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)
        
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)
        
    return np.array(lower[:-1] + upper[:-1])

def find_largest_quad_on_hull(hull):
    """
    Finds the 4 vertices on the hull that form the quadrilateral with max area.
    This is stable because it relies on the global shape, not local edges.
    """
    # If hull is small, just return it or pad
    if len(hull) <= 4:
        if len(hull) < 4: return np.pad(hull, ((0, 4-len(hull)), (0,0)), 'edge')
        return hull

    # Simplification:
    # The furthest two points on the hull are likely diagonals.
    # We can iterate to find the pair with max distance.
    # Then find the point furthest from that line on both sides.
    
    best_dist = 0
    p1_idx, p2_idx = 0, 0
    
    # 1. Find Diagonal (Longest distance)
    # Optimization: Only check points with high stride or just brute force (len is small)
    # Convex hulls are usually small (<50 points). Brute force is fine.
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = np.linalg.norm(hull[i] - hull[j])
            if d > best_dist:
                best_dist = d
                p1_idx, p2_idx = i, j
                
    p1 = hull[p1_idx]
    p2 = hull[p2_idx]
    
    # 2. Find the points furthest from this diagonal line (on each side)
    # Line vector
    line_vec = p2 - p1
    
    # Cross products for all points relative to p1
    vecs = hull - p1
    cross_prods = np.cross(line_vec, vecs)
    
    # One point will have max positive cross product, one will have max negative (min)
    p3_idx = np.argmax(cross_prods)
    p4_idx = np.argmin(cross_prods)
    
    corners = np.array([p1, hull[p3_idx], p2, hull[p4_idx]])
    
    # 3. Sort Corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    # Calculate centroid
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 0] - center[0], corners[:, 1] - center[1])
    
    # Sort by angle
    sort_order = np.argsort(angles)
    sorted_corners = corners[sort_order]
    
    return sorted_corners

