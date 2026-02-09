import cv2
import numpy as np
import math
import customCV

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

def extract_and_draw_final(frame, resizing_factor=3):
    
    # --- A. PREPROCESSING ---
    # 1. Downscale for speed
    small_frame = frame[::resizing_factor, ::resizing_factor]

    # 2. Grayscale
    grey = to_grayscale(small_frame)
    
    # 3. Blur (Gaussian)
    blurred = np.empty_like(grey)
    customCV.gaussian_blur(grey, blurred, 5, 2.0)
    
    # 4. Binarization (Otsu's Method recommended for AR tags)
    # Using your existing binarization function
    binary = binarization(blurred)
    
    # return binary
    
    # 5. Edge Detection
    gradient = detect_edges_binary(binary)

    # --- B. EXTRACT CONTOURS ---
    candidate_contours = extract_contours_from_gradient(gradient)
    
    # --- C. GEOMETRIC FILTERING ---
    # Filter small noise based on resizing factor
    min_area_thresh = 100 if resizing_factor == 1 else 100/resizing_factor
    
    # Filter for Quadrilaterals (using your custom function)
    quads = customCV.find_quads(candidate_contours, min_area_thresh)
    
    # ISOLATE TAGS (The Hierarchy Logic)
    # This keeps valid tags and removes paper borders/data noise
    valid_tags = isolate_multiple_tags(quads)
    # valid_tags = quads

    # --- D. DECODING & DRAWING ---
    output_frame = frame.copy()
    
    for tag in valid_tags:
        # 'tag' is a numpy array of [row, col] (y, x) in DOWNSCALED coordinates.
        
        # 1. DECODE ID
        # We use the small binary image and small coordinates for decoding.
        # We must flip [y, x] -> [x, y] for the Homography logic.
        tag_full_xy = (tag * resizing_factor).astype(np.float32)
        # Flip [row, col] -> [x, y] for homography
        tag_full_xy = tag_full_xy[:, ::-1] 
        
        # dst_pts = np.array([[144, 246], [171, 303], [240, 264], [210, 210]])

        # Pass the small binary image to read the bits
        tag_id, angle = decode_tag_id(frame, tag_full_xy)
    
        # 2. PREPARE DRAWING COORDINATES
        # Upscale: Multiply by resizing factor to map back to 1080p
        # Flip: Ensure we have [x, y] for OpenCV drawing functions
        upscaled_tag = (tag * resizing_factor).astype(np.int32)
        
        # Reshape to standard OpenCV format: (N, 1, 2)
        # We slice [:, ::-1] to flip Y,X to X,Y
        draw_pts = upscaled_tag[:, ::-1].reshape((-1, 1, 2))
        
        # 3. DRAW THE GREEN BOX
        cv2.polylines(output_frame, [draw_pts], isClosed=True, color=(0, 255, 0), thickness=5)

        dest_corners = draw_pts.reshape(4, 2).astype(np.float32)
        template_img = cv2.imread('assets/iitd_logo_template.jpg')
        # OVERLAY IMAGE
        # Only overlay if you successfully decoded the orientation
        if tag_id is not None:
             output_frame = superimpose_image(output_frame, dest_corners, template_img, angle)
        
        # 4. DRAW THE ID TEXT
        # Calculate center of the tag for text placement
        # M = cv2.moments(draw_pts)
        # if M["m00"] != 0:
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        
        #     text = f"ID: {tag_id}"
            
        #     # Draw black outline for text readability
        #     cv2.putText(output_frame, text, (cX - 40, cY), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        #     # Draw red text
        #     cv2.putText(output_frame, text, (cX - 40, cY), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Optional: Draw Orientation Corner (Blue Dot at Top-Right)
            # You might need to rotate this point based on 'angle' to show true "Up"
            # corner_pt = tuple(draw_pts[1][0]) # Assuming index 1 is TR
            # cv2.circle(output_frame, corner_pt, 10, (255, 0, 0), -1)

    return output_frame

def superimpose_image(frame, tag_corners, template_image, orientation_angle):
    """
    Overlays a template image onto the detected AR tag, respecting orientation.
    """
    if template_image is None:
        return frame
        
    # 1. Prepare Source Points (The Template Image Corners)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    h_temp, w_temp = template_image.shape[:2]
    src_pts = np.array([
        [0, 0], 
        [w_temp - 1, 0], 
        [w_temp - 1, h_temp - 1], 
        [0, h_temp - 1]
    ], dtype=np.float32)

    # 2. Prepare Destination Points (The Tag on Screen)
    # We must order the detected corners to match the template order (TL, TR, BR, BL).
    # First, get them in the standard geometric order.
    dst_pts = order_points(tag_corners)
    
    # 3. Adjust for Orientation
    # The 'orientation_angle' tells us how much the tag is rotated.
    # We shift the destination corners list so that index 0 is always the "True Top-Left".
    
    # Logic:
    # 0 deg   -> Anchor is TR. No shift needed.
    # 90 deg  -> Anchor is TL. Shift 1.
    # 180 deg -> Anchor is BL. Shift 2.
    # 270 deg -> Anchor is BR. Shift 3.
    
    shift_amount = 0
    if orientation_angle == 90:
        shift_amount = 1
    elif orientation_angle == 180:
        shift_amount = 2
    elif orientation_angle == 270:
        shift_amount = 3
        
    # Use numpy.roll to shift the array elements
    # We shift 'dst_pts' so the correct physical corner aligns with 'src_pts' [0,0]
    dst_pts = np.roll(dst_pts, shift_amount, axis=0)

    # 4. Calculate Homography
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # 5. Warp the Template Image
    # This creates an image of the same size as 'frame', but with the template
    # warped into the correct position. Black everywhere else.
    h_frame, w_frame = frame.shape[:2]
    warped_img = cv2.warpPerspective(template_image, H, (w_frame, h_frame))

    # 6. Create a Mask to composite
    # We need to cut a hole in the frame where the tag is, and fill it with the warped image.
    
    # Create a mask of the warped image (White where the image is, Black elsewhere)
    # Grayscale -> Threshold
    mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    
    # Invert mask (Black hole where tag is, White elsewhere)
    mask_inv = cv2.bitwise_not(mask)
    
    # 7. Blend
    # Black out the area of the tag in the original frame
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Take only the warped image region
    img_fg = cv2.bitwise_and(warped_img, warped_img, mask=mask)
    
    # Add them together
    final_frame = cv2.add(frame_bg, img_fg)
    
    return final_frame

def decode_tag_id(frame, corners):
    """
    Robust decoding with Center Sampling and Adaptive Thresholding.
    """
    # 1. Perspective Transform
    # Use a fixed size (e.g., 200px)
    size = 200
    cell_size = size // 8
    
    dst_pts = np.array([
        [0, 0], [size-1, 0], [size-1, size-1], [0, size-1]
    ], dtype=np.float32)
    
    rect = order_points(corners)
    M = get_perspective_transform(src=rect, dst=dst_pts)
    
    # Warp the image
    warped = warp_perspective_manual(frame, M, (size, size))
    
    if len(warped.shape) == 3 and warped.shape[2] == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 2. CRITICAL FIX: Re-Threshold the Warped Tag
    # Sometimes lighting varies across the tag. We enforce strict Black/White
    # on the warped image itself. Otsu is safest here.
    # Note: binary_image should be uint8.
    _, warped_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # DEBUG: Un-comment this to see exactly what the decoder sees!
    # If this looks like a white blob, your corners are detecting the Paper, not the Tag.
    # cv2.imshow("Debug Warped Tag", warped_bin) 

    # 3. Read the Grid using CENTER SAMPLING
    grid = np.zeros((8, 8), dtype=int)
    
    # Define a safe margin (e.g., 30%) to ignore the edges of each cell
    margin = int(cell_size * 0.3)
    
    for y in range(8):
        for x in range(8):
            # Coordinates of the cell
            x_start = x * cell_size
            y_start = y * cell_size
            x_end = (x + 1) * cell_size
            y_end = (y + 1) * cell_size
            
            # Extract only the CENTER region of the cell (ignoring borders)
            # This makes it robust against bad homography/rotation
            cell_center = warped_bin[y_start+margin : y_end-margin, 
                                     x_start+margin : x_end-margin]
            
            # If the center is mostly White, it's a 1.
            if cv2.mean(cell_center)[0] > 127:
                grid[y, x] = 1
            else:
                grid[y, x] = 0

    # 4. Determine Orientation (Rotate until White Anchor is at Top-Right)
    # The Inner 4x4 grid is indices 2 to 5.
    # Top-Right of inner grid is at index [2, 5].
    
    orientation = 0
    found = False
    
    for angle in [0, 90, 180, 270]:
        # Check if the Anchor (2,5) is White (1)
        # AND check if the other corners are Black (0) to reduce false positives
        # Corners: TL(2,2), TR(2,5), BR(5,5), BL(5,2)
        is_anchor_white = (grid[2, 5] == 1)
        
        # Optional: strictly check other corners are black (if your tag design follows that)
        # For now, we trust the single white anchor.
        if is_anchor_white:
            orientation = angle
            found = True
            break
        else:
            grid = np.rot90(grid) # Rotate grid counter-clockwise
            
    # If we spun 360 and didn't find the anchor, the ID will likely be garbage (or 15).
    if not found:
        return None, 0 # Return None to indicate read failure
    
    # 5. Decode ID from Central 2x2
    # Grid: (3,3), (3,4), (4,3), (4,4)
    bit1 = grid[3, 3]
    bit2 = grid[3, 4]
    bit3 = grid[4, 3]
    bit4 = grid[4, 4]
    
    tag_id = (bit1 << 3) | (bit2 << 2) | (bit3 << 1) | bit4
    
    return tag_id, orientation

def warp_perspective_manual(img, M, dsize):
    """
    Manually applies a perspective transform (Homography) to an image.
    Uses Nearest Neighbor interpolation for simplicity.
    
    Args:
        img: Input image (H, W, C) or (H, W).
        M: 3x3 Homography Matrix.
        dsize: Tuple (width, height) of output image.
        
    Returns:
        warped: The transformed image.
    """
    dst_w, dst_h = dsize
    src_h, src_w = img.shape[:2]
    
    # 1. Create a Grid of Destination Coordinates (x', y')
    # np.indices creates two grids: one for Y coordinates, one for X
    # shape: (2, dst_h, dst_w)
    y_grid, x_grid = np.indices((dst_h, dst_w))
    
    # 2. Flatten and Homogenize
    # We need coordinates in the form [x', y', 1] for matrix multiplication
    # shape: (3, N) where N is total pixels
    ones = np.ones_like(x_grid)
    dst_coords = np.stack([x_grid, y_grid, ones]).reshape(3, -1)
    
    # 3. Invert the Matrix
    # We transform Dest -> Source, so we need inverse of M
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print("Error: Matrix is not invertible.")
        return np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    # 4. Map Coordinates (Matrix Multiplication)
    # [x, y, w] = M_inv * [x', y', 1]
    src_homo_coords = M_inv @ dst_coords
    
    # 5. Normalize Homogeneous Coordinates
    # x = x_homo / w
    # y = y_homo / w
    w_vec = src_homo_coords[2, :]
    
    # Avoid division by zero
    w_vec[w_vec == 0] = 1e-5
    
    src_x = src_homo_coords[0, :] / w_vec
    src_y = src_homo_coords[1, :] / w_vec
    
    # 6. Nearest Neighbor Sampling
    # Round float coordinates to nearest integer pixel
    src_x = np.round(src_x).astype(int)
    src_y = np.round(src_y).astype(int)
    
    # 7. Boundary Checks (Mask pixels that fall outside original image)
    valid_mask = (
        (src_x >= 0) & (src_x < src_w) & 
        (src_y >= 0) & (src_y < src_h)
    )
    
    # 8. Create Output Image
    # Initialize black image
    if len(img.shape) == 3:
        warped = np.zeros((dst_h, dst_w, img.shape[2]), dtype=img.dtype)
    else:
        warped = np.zeros((dst_h, dst_w), dtype=img.dtype)
        
    # Only map valid pixels
    # Reshape indices back to image dimensions (H, W) for assignment
    # We assign values only where valid_mask is True
    
    # Flatten output to linear array for easy assignment
    warped_flat = warped.reshape(-1, warped.shape[-1] if warped.ndim==3 else 1)
    
    # Linear indices of valid pixels
    flat_indices = np.where(valid_mask)[0]
    
    # Sample from Source
    # format: img[y, x]
    if len(img.shape) == 3:
        values = img[src_y[valid_mask], src_x[valid_mask]]
        warped_flat[flat_indices] = values
        warped = warped_flat.reshape(dst_h, dst_w, 3)
    else:
        values = img[src_y[valid_mask], src_x[valid_mask]]
        warped_flat[flat_indices, 0] = values
        warped = warped_flat.reshape(dst_h, dst_w)

    return warped

def get_perspective_transform(src, dst):
    """
    Calculates the 3x3 Perspective Transform Matrix (Homography) 
    from 4 pairs of corresponding points.
    
    Args:
        src: (4, 2) numpy array of source points [x, y]
        dst: (4, 2) numpy array of destination points [x, y]
        
    Returns:
        M: (3, 3) numpy array representing the transform matrix.
    """
    # Ensure inputs are float32 for precision
    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)
    
    # We need to solve for 8 unknowns in the matrix:
    # [ h00, h01, h02 ]
    # [ h10, h11, h12 ]
    # [ h20, h21,   1 ]  <-- We fix h22 = 1
    
    # The system is A * h = B
    # A will be (8, 8), h will be (8,), B is (8,)
    A = []
    B = []
    
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        
        # Equation 1 (for x-coordinate 'u'):
        # h00*x + h01*y + h02 - h20*x*u - h21*y*u = u
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u])
        B.append(u)
        
        # Equation 2 (for y-coordinate 'v'):
        # h10*x + h11*y + h12 - h20*x*v - h21*y*v = v
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v])
        B.append(v)
        
    # Convert to numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    # Solve the linear system: A * h = B
    # We use np.linalg.solve (Gaussian elimination / LU decomposition)
    try:
        h = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular. Points might be collinear.")
        return None

    # Construct the final 3x3 matrix
    # Append the fixed h22 = 1 at the end
    M = np.append(h, 1).reshape((3, 3))
    
    return M

def order_points(pts):
    """
    Orders coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)
    
    # Sum: TL has min sum, BR has max sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Diff: TR has min diff (x-y), BL has max diff (x-y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def isolate_multiple_tags(quads, min_ratio=0.04, max_ratio=0.12):
    """
    Detects MULTIPLE AR tags by finding all 'Significant Children'.
    
    Args:
        quads: List of simplified contours (4 corners).
        min_ratio: Minimum size of Child relative to Parent (filters out Data bits).
        max_ratio: Maximum size (filters out duplicate borders/noise).
        
    Returns:
        valid_tags: List of contours, where each contour is an AR Tag.
    """
    if not quads:
        return []
    
    # print(len(quads))
    
    candidates = []
    
    # 1. Identify Parent-Child Relationships
    # We look for pairs where Inner is inside Outer
    for i, inner in enumerate(quads):

        outer_list = []
        for j, outer in enumerate(quads):
            if i == j: continue
            
            # Check Nesting
            if is_quad_inside(inner, outer):

                flag = False
                for k, inner_inner in enumerate(quads):
                    if(k == j or k == i): continue

                    if is_quad_inside(inner_inner, inner):
                        flag = True
                        break
                
                if(not flag):
                    outer_list.append(outer)
        
        # Append only the outer with max area
        if outer_list:
            max_outer = min(outer_list, key=cv2.contourArea)
            candidates.append(max_outer)

    
    # 2. Clean Up Duplicates (Optional but recommended)
    # Sometimes a tag might be detected twice (very close concentric lines).
    # We perform a simple Non-Max Suppression.
    # return candidates
    return remove_overlapping_quads(candidates)

def remove_overlapping_quads(candidates, iou_threshold=0.8):
    """
    Removes duplicate detections of the same tag.
    Keeps the largest one (Outer Border).
    """
    if not candidates: return []
    
    # Sort by Area (Descending) -> We prefer the larger "Outer" border
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
    
    keep = []
    for current in candidates:
        is_duplicate = False
        for kept in keep:
            if is_quad_inside(current, kept) or is_quad_inside(kept, current):
                 # Check how similar they are
                 a1 = cv2.contourArea(current)
                 a2 = cv2.contourArea(kept)
                 # If sizes are within 20% of each other, it's the same object
                 if min(a1, a2) / max(a1, a2) > iou_threshold:
                     is_duplicate = True
                     break
        
        if not is_duplicate:
            keep.append(current)
            
    return keep

def point_in_polygon(point, polygon):
    """
    Custom implementation of point-in-polygon test using ray casting algorithm.
    Returns True if point is inside polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def is_quad_inside(inner, outer):
    """ Strictly inside check using custom point-in-polygon test """
    # Convert outer to list of tuples for the custom function
    outer_polygon = [(float(pt[0]), float(pt[1])) for pt in outer]
    
    for pt in inner:
        if not point_in_polygon((float(pt[0]), float(pt[1])), outer_polygon):
            return False
    return True

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

    close_gaps(binary_image, iterations=1)

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
