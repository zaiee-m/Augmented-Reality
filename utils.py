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


def fill_polygon(image, vertices, color=(0, 0, 255)):
    """
    Fills a convex polygon defined by vertices on a numpy array image.
    Uses a basic scanline algorithm.
    
    Args:
        image: The image to draw on (H, W, C).
        vertices: Numpy array of shape (N, 2) containing [row, col] coordinates.
        color: Tuple (B, G, R) for the fill color.
    """
    vertices = vertices.astype(int)
    n = len(vertices)
    if n < 3: return

    h, w = image.shape[:2]

    # 1. Find bounding box to limit search range
    min_y = np.max([0, np.min(vertices[:, 0])])
    max_y = np.min([h - 1, np.max(vertices[:, 0])])

    # 2. Iterate through each scanline (row) within the bounding box
    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]

            y1, x1 = v1[0], v1[1]
            y2, x2 = v2[0], v2[1]

            # Check if current scanline 'y' intersects the edge line segment (v1, v2)
            # We use < on one side and >= on the other to handle vertices exactly on scanlines correctly.
            if (y1 <= y < y2) or (y2 <= y < y1):
                # Calculate X intersection point using linear interpolation formula
                # Avoid division by zero for horizontal lines (y2 - y1 = 0), though the if condition usually prevents this.
                if y2 != y1:
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(x)

        # 3. Sort intersections by X coordinate
        intersections.sort()

        # 4. Fill pixels between pairs of intersections (even-odd rule)
        # For a convex polygon, a scanline usually enters once and leaves once (2 intersections).
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                # Round and clip X coordinates to image bounds
                x_start = int(np.round(intersections[i]))
                x_end = int(np.round(intersections[i+1]))
                
                x_start = np.max([0, x_start])
                # Add 1 for inclusive slicing behavior if desired, or keep as is for standard filling
                x_end = np.min([w, x_end + 1]) 

                if x_start < x_end:
                    # Fill the horizontal segment
                    image[y, x_start:x_end] = color

def extract_and_draw_final(frame, resizing_factor=4):
    print(f"--- Processing Frame (Scale: 1/{resizing_factor}) ---")
    
    # A. PREPROCESSING
    # If factor is 1, this just returns a copy of the frame
    small_frame = fast_scale(frame, scale_factor=resizing_factor)
    grey = to_grayscale(small_frame)
    

    blurred = gaussian_blur(grey, kernel_size=9, sigma=2.0)
    
    # B. BINARIZATION (Otsu + Auto-Invert)
    # Separates Grey Floor from White Paper
    binary = binarization(blurred)
    
    raw_hierarchy = extract_contours_hierarchical(binary)

    return binary

    print(f"DEBUG: Found {len(raw_hierarchy)} top-level shapes.")

    # 2. FLATTEN HIERARCHY
    # We want ALL contours: The outer box, the inner box, and the symbols inside.
    candidate_contours = []
    
    # Initialize stack with the top-level contours
    stack = list(raw_hierarchy)
    
    while stack:
        # Pop one shape to process
        current_shape = stack.pop()
        
        # A. Add this shape's coordinates to our final list
        candidate_contours.append(current_shape['contour'])
        
        # B. If it has children, add them to the stack so we process them next
        if current_shape['children']:
            stack.extend(current_shape['children'])

    print(f"DEBUG: Flattened hierarchy contains {len(candidate_contours)} total contours.")

    # E. GEOMETRIC FILTERING (Size & Squareness)
    # Since we are now including small inner symbols (grandchildren), 
    # we must be careful not to filter them out with a high min_area.
    
    # If looking for the big AR tag frames only: keep area high (e.g. 1000)
    # If looking for the tiny inner data bits: reduce area (e.g. 10 or 20)
    min_area_thresh = 50 if resizing_factor == 1 else 10
    
    # We pass 'candidate_contours' (which is now a clean list of arrays)
    # valid_tags = filter_contours(candidate_contours, min_area=min_area_thresh)
    
    print(f"DEBUG: Filtered down to {len(candidate_contours)} valid tags.")

    # F. DRAWING (RED FILL)
    output_frame = frame.copy()
    red_color = (0, 0, 255) # BGR
    
    for tag in candidate_contours:
        # 'tag' is now just the numpy array of coordinates
        
        # Upscale coordinates (if resizing_factor > 1)
        upscaled_tag = tag * resizing_factor
        
        # Convert to standard OpenCV format: (N, 1, 2)
        # FLIP [row, col] -> [x, y] for OpenCV drawing
        pts = upscaled_tag[:, ::-1].astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw
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

def extract_contours_hierarchical(binary_image, level=0):
    """
    Recursive function to find contours and their children (holes).
    
    Args:
        binary_image: 0 (Black Object) and 255 (White Background).
        level: Depth of recursion (0=Outermost, 1=Hole, 2=Inside Hole, etc.)
    
    Returns:
        List of tuples: (contour_coords, children_list, hierarchy_level)
    """
    
    # Base case: Image is too small or empty
    if binary_image.shape[0] < 3 or binary_image.shape[1] < 3:
        return []

    hierarchy = []
    h, w = binary_image.shape
    
    # Work copy to erase shapes as we find them
    # Crucial: We only erase from THIS level's search, not the original data
    work_image = binary_image.copy()

    # --- OPTIMIZATION: Candidate Selection (from previous turn) ---
    is_black = (work_image == 0)
    is_white = (work_image == 255)
    w_pad = np.pad(is_white, pad_width=1, mode='constant', constant_values=False)
    has_white_neighbor = (w_pad[0:-2, 1:-1] | w_pad[2:, 1:-1] | 
                          w_pad[1:-1, 0:-2] | w_pad[1:-1, 2:])
    boundary_mask = is_black & has_white_neighbor
    boundary_points = np.argwhere(boundary_mask)
    # -------------------------------------------------------------

    for start_point in boundary_points:
        y, x = start_point
        
        # Check if already processed/erased
        if work_image[y, x] == 255: continue

        # 1. Trace the Outer Contour
        contour = moore_neighbour_trace(work_image, start_point)
        
        if contour is None or len(contour) < 3:
            work_image[y, x] = 255 # clear noise
            continue

        # 2. Create a Mask for this specific shape
        # We need a blank canvas to paint ONLY this shape
        shape_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use fillPoly to create a solid mask of the contour
        # This fills the "Interior" of the contour with 255
        # Note: cv2.fillPoly expects (x,y), numpy uses (y,x). Flip cols for drawing.
        cv2.fillPoly(shape_mask, [np.fliplr(contour)], 255)
        
        # 3. Extract the "Interior" from the Original Image
        # logical_and gets us the original pixels, but ONLY where the mask is white
        # This captures the "holes" inside the shape.
        interior_view = np.full((h, w), 255, dtype=np.uint8) # Default white background
        
        # Where the mask is valid, copy data from the ORIGINAL binary_image
        # But we need to INVERT it. 
        # Why? Because 'extract_contours' looks for BLACK objects.
        # The 'holes' inside our current black object are WHITE.
        # To find them, we must turn them BLACK.
        
        mask_bool = (shape_mask == 255)
        
        # If original was 255 (hole), it becomes 0 (target). 
        # If original was 0 (solid part), it becomes 255 (background).
        interior_view[mask_bool] = 255 - binary_image[mask_bool]

        # 4. Erase this contour from the CURRENT level (so we don't find it again)
        # We set the pixels of this shape to 255 (White/Background) in work_image
        work_image[mask_bool] = 255

        # 5. RECURSION: Look for contours inside this shape
        # Pass the inverted interior view.
        children = extract_contours_hierarchical(interior_view, level + 1)
        
        hierarchy.append({
            'contour': contour,
            'children': children,
            'level': level
        })
        
    return hierarchy

def moore_neighbour_trace(binary_image, start_point):
    """
    Guaranteed to exit. Includes a hard safety limit to prevent freezing.
    """
    # Pad with White (255)
    padded_image = np.pad(binary_image, pad_width=1, mode="constant", constant_values=255)
    
    offsets = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
    
    contour = []
    
    # Shift start point to Padded World
    current_point = start_point.copy() + 1
    start_padded = current_point.copy()
    
    backtrack_idx = 0
    
    # SAFETY: Stop after this many pixels. 
    # (A 480p image perimeter is ~2000 pixels. 10k is plenty.)
    max_steps = 10000 
    steps = 0

    while True:
        steps += 1
        if steps > max_steps:
            # FORCE BREAK: Prevent infinite loop
            break

        # Save Real World Coordinate
        contour.append(current_point - 1)
        
        # Search neighbors
        search_indices = (np.arange(8) + backtrack_idx) % 8
        neighbor_coords = current_point + offsets[search_indices]
        neighbor_vals = padded_image[neighbor_coords[:, 0], neighbor_coords[:, 1]]
        
        is_black = (neighbor_vals == 0)
        
        # If isolated pixel (no neighbors), stop.
        if not np.any(is_black):
            break
            
        # Move to first black neighbor
        found_local = np.argmax(is_black)
        found_global = search_indices[found_local]
        
        current_point = current_point + offsets[found_global]
        backtrack_idx = (found_global + 5) % 8
        
        # STOPPING CONDITION
        # If we are back at the start coordinates
        if (current_point[0] == start_padded[0] and 
            current_point[1] == start_padded[1]):
            break
            
    return np.array(contour)

def fill_contour(binary_image, contour):
    """
    Manually fills the inside of a contour with White (255).
    This effectively 'deletes' the object from the binary image.
    
    Assumption: The shape is roughly convex (like a square/AR tag).
    """

    # Get all unique Y-coordinates (rows) occupied by the contour
    min_y = np.min(contour[:, 0])
    max_y = np.max(contour[:, 0])

    # Scanline Fill: For every row, find the Left and Right edges.
    for y in range(min_y, max_y + 1):
        # Find all points in the contour that are on this specific row
        indices = np.where(contour[:, 0] == y)[0]

        if len(indices) > 0:
            # Get the X coordinates for this row
            x_coords = contour[indices, 1]

            # Find the start (min) and end (max) of the shape on this row
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)

            # Fill the span with White (255)
            # We add +1 to max_x because numpy slicing is exclusive at the end
            binary_image[y, min_x : max_x + 1] = 255


def extract_contours(binary_image):
    """
    Extracts all the contours within a binary_image as a list of coordinates.
    Optimized: Only iterates over boundary pixels (black pixels with white neighbors).
    
    :param binary_image: NumPy 2D array of 0 (blacks) and 255 (White).
    """
    contours = []
    h, w = binary_image.shape
    
    # Work on a copy to allow filling/erasing shapes
    work_image = binary_image.copy()

    # --- OPTIMIZATION START ---
    # We want to find pixels that are:
    # 1. BLACK (0)
    # 2. Have at least one WHITE (255) neighbor
    
    # Boolean masks
    is_black = (work_image == 0)
    is_white = (work_image == 255)
    
    # Check 4-neighbors (Up, Down, Left, Right) using slicing/padding
    # Pad the white mask with False (treat border as non-white/black boundary)
    # or True (treat border as white/background). 
    # Usually, for contouring, treating the border as background ensures objects touching the edge are detected.
    w_pad = np.pad(is_white, pad_width=1, mode='constant', constant_values=False)
    
    # Shift padded array to check neighbors
    # If a pixel's North neighbor is White, w_pad[y, x+1] (shifted down relative to center) is True.
    has_white_neighbor = (
        w_pad[0:-2, 1:-1] |  # North neighbor is White
        w_pad[2:,   1:-1] |  # South neighbor is White
        w_pad[1:-1, 0:-2] |  # West neighbor is White
        w_pad[1:-1, 2:]      # East neighbor is White
    )
    
    # The Candidates: Must be Black AND have a White Neighbor
    boundary_mask = is_black & has_white_neighbor
    
    # Get coordinates of all True values in boundary_mask
    # np.argwhere returns a list of [y, x] coordinates
    boundary_points = np.argwhere(boundary_mask)
    # --- OPTIMIZATION END ---

    for start_point in boundary_points:
        y, x = start_point
        
        # Is this pixel still black?
        # Even though we pre-calculated boundary_points, 'fill_contour' might have
        # erased this pixel if it belonged to a shape we processed in a previous iteration.
        if work_image[y, x] == 255: 
            continue

        contour = moore_neighbour_trace(work_image, start_point)

        if contour is not None and len(contour) > 2:
            contours.append(contour)
            fill_contour(work_image, contour)
        else:
            # Error handling: single pixel noise
            work_image[y, x] = 255
            
    return contours

def filter_contours(contours, min_area=50):
    """
    Filters a list of raw contours to find potential AR tags.
    
    Strategy:
    1. Filter by Area (Fast): Discard tiny noise specks.
    2. Filter by Shape (Slow): Run Douglas-Peucker to check for 4 corners.
    
    Args:
        contours: List of numpy arrays (the raw output from extract_contours).
        min_area: Minimum pixel area to consider. 
                  (Keep this small if working on the downscaled image!)
                  
    Returns:
        valid_tags: List of simplified 4-corner numpy arrays.
    """

    # return contours

    valid_tags = []
    
    for contour in contours:
        corners = get_corners(contour)
        
        if len(corners) > 1:
                    start_pt = corners[0]
                    end_pt = corners[-1]
                    
                    # Calculate distance between First and Last point
                    dist = np.linalg.norm(start_pt - end_pt)
                    
                    # If they are closer than 10 pixels, it's the same corner. Drop the last one.
                    if dist < 10:
                        corners = corners[:-1]
        
        if len(corners) == 4:
            valid_tags.append(corners)

    return valid_tags


def get_corners(contour):
    """
    Wrapper to simplify a contour into exactly 4 corners if possible.
    """
    # Calculate Perimeter (Approximation)
    # The tolerance (epsilon) is usually relative to the size of the shape.
    # A standard heuristic is 1% to 5% of the arc length.
    perimeter = np.sum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    epsilon = 0.05 * perimeter  # Start with 2% tolerance
    
    # Run RDP
    approx_corners = simplify_contour(contour, epsilon)
    
    return approx_corners


    
def simplify_contour(points, epsilon):
    """
    Recursively simplifies a contour using the Ramer-Douglas-Peucker algorithm.
    
    Args:
        points: (N, 2) numpy array of coordinates.
        epsilon: Max distance allowed from the straight line (The 'tolerance').
        
    Returns:
        simplified: (M, 2) numpy array of the key corners.
    """

    def get_perpendicular_distances(points, start_point, end_point):
        # Vector representing the Line (Start -> End)
        line_vec = end_point - start_point
        
        # Vectors representing Start -> Points (Broadcasting)
        # Subtracts start_point from EVERY row in points instantly
        point_vecs = points - start_point
        
        # Calculate Magnitude of the Cross Product (The "Area")
        cross_product = np.cross(line_vec, point_vecs)
        
        # Calculate Length of the Line Vector (The "Base")
        line_len = np.linalg.norm(line_vec)
        
        # Safety: Avoid division by zero if start == end
        if line_len == 0:
            return np.linalg.norm(point_vecs, axis=1)
            
        # Distance = Area / Base
        return np.abs(cross_product) / line_len


    # Base Case: If only 2 points are left, we cannot simplify further.
    if len(points) < 3:
        return points

    # Setup the Line (Start -> End)
    start_point = points[0]
    end_point = points[-1]
    
    # Find the point with the Maximum Distance from this line
    # (Using the helper function from the previous step)
    dists = get_perpendicular_distances(points, start_point, end_point)
    
    max_dist = np.max(dists)
    index = np.argmax(dists)
    
    # The Decision
    if max_dist > epsilon:
        # Split the curve into two halves at the index
        # We include the corner point in BOTH halves so they connect.
        left_curve = points[:index + 1]
        right_curve = points[index:]
        
        # Recursively simplify both halves
        # These function calls will drill down until they hit straight lines
        simplified_left = simplify_contour(left_curve, epsilon)
        simplified_right = simplify_contour(right_curve, epsilon)
        
        # Merge the results
        return np.vstack((simplified_left[:-1], simplified_right))
        
    else:
        # Just return the Start and End.
        return np.array([start_point, end_point])
    

