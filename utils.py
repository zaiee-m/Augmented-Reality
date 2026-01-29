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

def prepreocess_frame(frame):
    """
    Take a BGR frame and returns a binarized processed frame.
    
    :param frame: BGR image frame
    :return: Binarized image (0 and 255 values)
    """

    # Set the scale_factor.
    resizing_factor = 4

    # To reduce computation overhead, rescale image before processing it.
    frame = fast_scale(frame, scale_factor = resizing_factor)

    grey = to_grayscale(frame)
    blurred = gaussian_blur(grey, kernel_size=5, sigma=1.0)
    binary_frame = binarization(blurred, resizing_factor = resizing_factor)
    return binary_frame

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


def binarization(gray_image, resizing_factor, window_size=31, t=0.025):
    """
    Manually implements Bradley-Roth Adaptive Thresholding using Integral Images.
    
    Args:
        gray_image: Grayscale input image (2D numpy array).
        window_size: The size of the neighborhood to check (local average).
        t: Threshold percentage (0.15 means pixel must be 15% darker than mean).
        
    Returns:
        binary_image: 0 for background, 255 for foreground (tag border).
    """
    scaled_window = max(3, window_size // resizing_factor)

    h,w = gray_image.shape

    # ...
    integral = np.zeros((h+1,w+1),dtype=np.float64)

    # ...
    integral[1:,1:] = np.cumsum(np.cumsum(gray_image.astype(np.float64),axis=0),axis=1)

    r = scaled_window // 2

    rows, cols = np.indices((h,w))

    # Calculate the Top-Left (r1, c1) and Bottom-Right (r2, c2) of the windows.
    # In essence the four boundaries of the window centered at each pixel.
    # We add 1 is added to r2/c2 because these correspond to coordinates within 
    # the padded integral image.
    r1 = np.maximum(rows - r ,0)    
    c1 = np.maximum(cols - r, 0)
    r2 = np.minimum(rows + r + 1, h)
    c2 = np.minimum(cols + r + 1, w)

    # ...
    # The (r1, c1) and (r2, c2) are not part of the window but circumvent it.
    local_window_sum = (
        integral[r2, c2] 
        - integral[r1, c2] 
        - integral[r2, c1] 
        + integral[r1, c1]
    )

    # We must calculate the exact area because windows at the borders are smaller.
    area = (r2 - r1) * (c2 - c1)
    local_mean = local_window_sum / area

    binary_image = np.full((h, w), 255, dtype=np.uint8)

    # Condition: Pixel < (Mean * (1-t))
    # If true, set to Black (0)
    binary_image[gray_image < (local_mean * (1 - t))] = 0

    clean_binary = close_gaps(binary_image)

    return clean_binary




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


def moore_neighbour_trace(binary_image):
    """
        Traces the outer contour of the first black object found in a binary image.
        Uses Moore-Neighbor algorithm (Clockwise).
        
        Args:
            binary_image: 2D numpy array (0=Black/Tag, 255=White/Background).
            
        Returns:
            contour: Numpy array of shape (N, 2) containing (y, x) coordinates.
    """
    # Pad with 255 (White) so we never crash at the edges
    padded_image = np.pad(binary_image, pad_width = 1, mode = "constant", constant_values=255)

    # Returns a list of all black pixels: [[y1, x1], [y2, x2], ...]
    # We just take the first one [0].
    black_pixels = np.argwhere(padded_image == 255)

    if len(black_pixels) == 0:
        return np.array([]) # No object found

    start_point = black_pixels[0]

    # Moore Neighborhood (Clockwise starting from UP)
    # (dy, dx)
    offsets = np.array(
        [
            [-1,  0], # 0: Up
            [-1,  1], # 1: Up-Right
            [ 0,  1], # 2: Right
            [ 1,  1], # 3: Down-Right
            [ 1,  0], # 4: Down
            [ 1, -1], # 5: Down-Left
            [ 0, -1], # 6: Left
            [-1, -1]  # 7: Up-Left
        ]
    )

    contour = []
    current_point = start_point.copy()

    backtrack_idx = 0

    while True: 
        # Save current coordinate (Subtract 1 to un-pad)
        contour.append(current_point - 1)

        # If backtrack is 2, indices are [2, 3, 4, 5, 6, 7, 0, 1]
        search_indices = (np.arange(8) + backtrack_idx) % 8

        # Get Coordinates of all 8 neighbors in search order (Vectorized)
        neighbor_coords = current_point + offsets[search_indices]

        # Extract Values from Image (Vectorized Lookup)
        neighbor_values = padded_image[neighbor_coords[:, 0], neighbor_coords[:, 1]]

        is_black = (neighbor_values == 0)
        
        if not np.any(is_black):
            break # Isolated pixel (singularity), stop.
        
        # logical_not turns 0 (Black) to True. argmax finds first True.
        found_local_idx = np.argmax(is_black)

        # Retrieve the actual global direction index (0-7)
        # This is used to calculate the new backtrack_idx.
        found_global_idx = search_indices[found_local_idx]

        # Move to the new pixel
        current_point = current_point + offsets[found_global_idx]

        # We start the NEXT search from the neighbor *after* the one we just came from.
        # Logic: If we moved East (idx 2), we entered the new pixel from the West (idx 6).
        # We start scanning clockwise from West+1 (Up-Left, idx 7).
        # Formula: (found_idx + 4 + 1) % 8
        backtrack_idx = (found_global_idx + 5) % 8

        # Stop if we return to Start AND we are searching from the same direction.
        if (current_point[0] == start_point[0] and 
            current_point[1] == start_point[1] and 
            backtrack_idx == 0): # Assuming we started with backtrack 0
            break
            
    return np.array(contour)

        

    




