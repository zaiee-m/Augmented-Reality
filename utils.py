import cv2
import numpy as np
import math
import customcv

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

def render(img, obj, projection, model, orientation=0, color=False):
    """
    Args:
        orientation: The angle of the tag (0, 90, 180, 270)
    """
    h, w = model.shape[:2]

    # --- 1. SETUP ROTATION MATRICES ---
    
    # A. Tag Orientation (Dynamic): Rotates model around Z-axis to match tag
    # We negate the angle because OpenCV's y-axis is inverted relative to standard math
    angle_rad = math.radians(-orientation) 
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    # Rotation around Z-axis
    R_tag = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    # B. Model Correction (Static): Fixes "Sideways" or "Lying Down" models
    # Try changing 'rx' to 0 or 180 if the wolf is still sideways!
    # Common OBJ fix: Rotate 90 degrees around X to stand it up.
    rx = math.radians(90) 
    
    R_model_fix = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])

    # --- 2. AUTO-SCALE ---
    vertices = np.array(obj.vertices)
    # Center the model vertices around (0,0,0) first
    object_center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2.0
    vertices = vertices - object_center
    
    # Scale to fit inside the 100x100 world
    max_dim = np.max(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    scale_factor = (w * 0.8) / (max_dim if max_dim > 0 else 1)
    scale_matrix = np.eye(3) * scale_factor

    # --- 3. PROCESSING LOOP ---
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        
        # TRANSFORMATION CHAIN:
        # 1. Scale the model
        points = np.dot(points, scale_matrix)
        
        # 2. Fix the Model's native orientation (Stand it up)
        points = np.dot(points, R_model_fix)
        
        # 3. Apply Tag Orientation (Rotate with the physical paper)
        points = np.dot(points, R_tag)
        
        # 4. Move to Center of the Tag (World Translation)
        # We assume the tag is at Z=0. We lift the model up by half its height so it stands ON the tag.
        points = np.array([[p[0] + w/2, p[1] + h/2, p[2]] for p in points])

        # --- 4. PROJECTION ---
        points_homo = np.hstack((points, np.ones((len(points), 1))))
        projected = np.dot(projection, points_homo.T)
        
        z = projected[2, :]
        # Clip points behind camera to avoid "Dark Frame" glitches
        if np.any(z <= 0.1): continue
            
        u = projected[0, :] / z
        v = projected[1, :] / z
        imgpts = np.vstack((u, v)).T.astype(np.int32)

        # --- 5. DRAW ---
        if color:
             try:
                # If face[-1] is a hex string (standard)
                if isinstance(face[-1], str):
                    c = hex_to_rgb(face[-1])[::-1]
                    cv2.fillConvexPoly(img, imgpts, c)
                else:
                    # Fallback for generic OBJ
                    cv2.fillConvexPoly(img, imgpts, (100, 200, 100)) 
             except:
                cv2.fillConvexPoly(img, imgpts, (0, 0, 255))
        else:
            cv2.fillConvexPoly(img, imgpts, (0, 0, 0))

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

def extract_and_draw_ar_tags(frame, resizing_factor=1):
    """
    Extracts ar tags from an RGB frame.
    
    Args:
        frame: RGB frame in np(N,1,3) format.
        resizing_factor: factor to be used to scale image for resizing.

    Returns:
        frame, tags tuple
        tags: all the extracts tags as a list of np(N,2) arrays in xy format.
        frame: video frame with tag boundaries drawn for visualisation.
    """

        # --- A. PREPROCESSING ---
    # 1. Downscale for speed
    small_frame = frame[::resizing_factor, ::resizing_factor]

    # 2. Grayscale
    grey = to_grayscale(small_frame)
    
    # 3. Blur (Gaussian)
    blurred = np.empty_like(grey)
    customcv.gaussian_blur(grey, blurred, 5, 2.0)
    
    # 4. Binarization (Otsu's Method recommended for AR tags)
    # Using your existing binarization function
    binary = binarization(blurred)
    # return binary
    # 5. Edge Detection
    gradient = detect_edges_binary(binary)
    # return gradient
    
    # --- B. EXTRACT CONTOURS ---
    contours, hierarchy = customcv.find_contours(gradient)

    # The shapes that follow the inside a rectangle,
    # corners >= 4, and shares center with parent criteria.
    tag_candidates = extract_tags(contours,hierarchy)
    
    valid_tags = []

    # Validtae candidate tags by checking the 2x2 boundary.
    for tag in tag_candidates:
        _, _, grid = decode_tag_id(frame, tag)
        
        if grid is None:
            continue

        is_boundary_valid = (
            np.all(grid[0:2, :] == 0) and  # Top 2 rows are black
            np.all(grid[-2:, :] == 0) and  # Bottom 2 rows are black
            np.all(grid[:, 0:2] == 0) and  # Left 2 columns are black
            np.all(grid[:, -2:] == 0)      # Right 2 columns are black
        )

        if is_boundary_valid:
            valid_tags.append(tag)
        

    corrected_contours = []

    for cnt in valid_tags:
        # Reshape to (N, 1, 2) and ensure it's int32.
        cnt_formatted = cnt.reshape((-1, 1, 2)).astype(np.int32)
        
        corrected_contours.append(cnt_formatted)

    cv2.drawContours(frame,corrected_contours,-1,(0,255,0),3)

    return frame, valid_tags

def detect_tags_in_image(frame, resizing_factor=1):
    """
    Docstring for detect_and_draw_tags
    
    :param frame: Description
    :param resizing_factor: Description
    """
    # Extract and draw ar tags.
    frame_with_contours_drawn, tags = extract_and_draw_ar_tags(frame, resizing_factor=resizing_factor)

    # Filter small noise based on resizing factor
    min_area_thresh = 100 if resizing_factor == 1 else 100/resizing_factor

    output_frame = frame_with_contours_drawn

    for tag in tags:
        # Extact tag_id and orientation of the corner cell.
        # For e.g. angle=90 means the tag is rotated 90 degrees
        # clockwise from the correct orientation (white cell as bottom right).
        tag_id, angle, _ = decode_tag_id(frame, tag)

        # For tag_id visualization.
        cX, cY = get_polygon_centroid(tag)
    
        text = f"ID: {tag_id}"
        
        # Draw black outline for text readability
        cv2.putText(output_frame, text, (cX - 40, cY), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        # Draw red text
        cv2.putText(output_frame, text, (cX - 40, cY), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw Orientation Corner (Blue Dot at Bottom-Right)
        # Rotate this point based on 'angle' to show true "Up".

        # Visualizing the Orientation (Blue Dot on the Tag's Top-Right Corner)
        # Points in tag are in TL, BL, BR, TR order.
        if angle == 0:
            tr_index = 2
        elif angle == 90:
            tr_index = 3
        elif angle == 180:
            tr_index = 0  # Physical TR is now at Screen Bottom-Left
        else: # angle == 270
            tr_index = 1 # Physical TR is now at Screen Bottom-Right

        # Extract point and ensure it's a tuple of integers
        # draw_pts is shape (4, 2).
        tr_point = tag[tr_index]
        corner_pt = tuple(tr_point) # Assuming index 1 is TR
        cv2.circle(output_frame, corner_pt, 10, (255, 0, 0), -1)

    return output_frame 

def overlay_image(frame, template_path):
    """
    Docstring for overlay_image
    
    :param frame: Description
    :param template: Description
    """

    # Load template image.
    template_img = cv2.imread(template_path)

    if template_img is None:
        raise FileNotFoundError("Could not load image at 'assets/iitd_logo_template.jpg'. Check the path and file integrity.")
    
    frame_with_ar_tag, tags = extract_and_draw_ar_tags(frame)
    output_frame = frame_with_ar_tag

    for tag in tags:
        # Determine current tag orientation.
        _, angle, _ = decode_tag_id(frame, tag)

        # Reshape just in case.
        dest_corners = tag.reshape(4, 2).astype(np.float32)

        # Superimpose image onto the tag in angle orientation.
        output_frame = superimpose_image(output_frame, dest_corners, template_img, angle)
    return output_frame

def overlay_object(frame, object_path):
    """
    Overlays a 3d .obj file onto an ar tag in the frame.
    
    Args:
        frame: video frame to be overlaid.
        object_path: path to the object to be overlaid.
    """

    # Extract and draw ar tags in the frame.
    frame_with_ar_tag, tags = extract_and_draw_ar_tags(frame)
    output_frame = frame_with_ar_tag

    # Load the 3d object file.
    obj_3d = OBJ(object_path)
    square_size = 100
    dummy_model_surface = np.zeros((square_size, square_size), dtype=np.uint8)

    mtx, _ = load_calibration()

    dst_points = np.array([
        [0, 0],
        [square_size, 0],
        [square_size, square_size],
        [0, square_size]
    ], dtype=np.float32)

    for tag in tags:
        tag_id, angle, _ = decode_tag_id(frame, tag)

        # Convert detected corners to float32
        dest_corners = tag.reshape(4, 2).astype(np.float32)

        if tag_id is not None:
            
    
            # We map World (dst_points) -> Image (dest_corners)
            H = get_perspective_transform(dst_points, dest_corners)
            
            # This will now return a valid Projection Matrix
            P = get_projection_matrix(H=H, K=mtx)
            
            # Pass the dummy surface so the object centers at (50, 50)
            # instead of (1, 2)
            output_frame = render(
                img=output_frame, 
                obj=obj_3d, 
                projection=P, 
                model=dummy_model_surface, 
                color=False,
                orientation=angle
            )

    return output_frame

def get_polygon_centroid(contour):
    """
    Calculates the centroid using Green's Theorem (the "Shoelace Formula").
    
    Args:
        contour: Points representing the contour in np.array(N,1,2) or np.array(N,2) format.

    Returns:
        centroid: tuple of numbers representing the centroid in xy
    """
    #  Flatten and ensure float type for precision
    pts = contour.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    
    # Needs at least a triangle
    if n < 3:
        return np.mean(pts, axis=0).astype(int) # Fallback

    #  Vectorized Shift (Roll)
    # x represents x_i, x_next represents x_{i+1}
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Shift arrays by -1 (move index 1 to 0, wrap last to first)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    
    # 3. Compute Cross Product Term (Vectorized)
    # (x_i * y_{i+1} - x_{i+1} * y_i)
    cross_term = x * y_next - x_next * y
    
    #  Compute Sums
    # Note: This 'area_sum' is actually 2 * SignedArea
    area_sum = np.sum(cross_term)
    
    # Avoid division by zero
    if abs(area_sum) < 1e-6:
        return np.mean(pts, axis=0).astype(int)
    
    # Compute Centroids
    cx = np.sum((x + x_next) * cross_term)
    cy = np.sum((y + y_next) * cross_term)
    
    # Formula: C = Sum / (6 * SignedArea)
    # Since area_sum = 2 * SignedArea, we divide by 3 * area_sum
    cx = cx / (3.0 * area_sum)
    cy = cy / (3.0 * area_sum)
    
    return int(cx), int(cy)


def extract_tags(contours, hierarchy):
    """
    Docstring for extract_tags
    
    :param contours: Description
    :param hierarchy: Description
    """
    
    all_tag_corners = []
    processed_parents = set()

    # hierarchy shape is usually (1, N, 4) -> [Next, Previous, First_Child, Parent]
    if hierarchy is None or len(hierarchy) == 0:
        return []

    for i in range(len(contours)):
        # Filter 1: Look for contours with a parent (child/hole)
        parent_id = hierarchy[0][i][3]
        
        # Only process if it has a parent and we haven't already validated this tag
        if parent_id != -1 and parent_id not in processed_parents:

            area_parent = cv2.contourArea(contours[parent_id])
            if area_parent < 500: # Minimum recognizable size
                continue

            # Filter 2: Centroid Check
            c1 = np.array(get_polygon_centroid(contours[i]))
            c2 = np.array(get_polygon_centroid(contours[parent_id]))
                
            # Verify centers within < 10% of parent's diagonal
            diagonal_approx = np.sqrt(area_parent)
            if np.linalg.norm(c1 - c2) < (0.10 * diagonal_approx):

                # Filter 3: Check inner portion complexity
                p1 = arc_length(contours[i], True)
                approx1 = customcv.approx_poly_dp(contours[i], 0.02 * p1, True)
                
                if len(approx1) > 4:

                    # Filter 4: Look for quadrilateral parents (the tag border)
                    p2 = arc_length(contours[parent_id], True)
                    approx2 = customcv.approx_poly_dp(contours[parent_id], 0.02 * p2, True)

                    if len(approx2) == 4:
                        # Order the 4 corners of the parent (the tag)
                        ordered_corners = order_points(approx2)
                        all_tag_corners.append(ordered_corners.astype(np.int32))
                        
                        # Mark parent as processed to avoid duplicate detection
                        processed_parents.add(parent_id)

    return all_tag_corners

def superimpose_image(frame, tag_corners, template_image, orientation_angle):
    """
    Docstring for superimpose_image
    
    :param frame: Description
    :param tag_corners: Description
    :param template_image: Description
    :param orientation_angle: Description
    """
        
    # Prepare Source Points (The Template Image Corners)
    # Order: Top-Left, Bottom-Left, Bottom-Right, Top-Right.
    h_temp, w_temp = template_image.shape[:2]
    src_pts = np.array([
        [0, 0],                     # Index 0: Top-Left
        [w_temp - 1, 0],            # Index 1: Top-Right
        [w_temp - 1, h_temp - 1],   # Index 2: Bottom-Right
        [0, h_temp - 1]             # Index 3: Bottom-Left
    ], dtype=np.float32)

    # Prepare Destination Points (The Tag on Screen)
    # We must order the detected corners to match the template order (TL, TR, BR, BL).
    # First, get them in the standard geometric order.
    dst_pts = order_points(tag_corners)
    
    # Adjust for Orientation
    # The 'orientation_angle' tells us how much the tag is rotated.
    # We shift the destination corners list so that index 0 is always the "True Top-Left".
    
    shift_amount = 0
    if orientation_angle == 90:
        shift_amount = -1  # or 3
    elif orientation_angle == 180:
        shift_amount = -2  # or 2
    elif orientation_angle == 270:
        shift_amount = -3  # or 1
        
    # Use numpy.roll to shift the array elements
    # We shift 'dst_pts' so the correct physical corner aligns with 'src_pts' [0,0]
    dst_pts = np.roll(dst_pts, shift_amount, axis=0)

    # Calculate Homography
    H = get_perspective_transform(src=src_pts, dst=dst_pts)

    # Warp the Template Image
    # This creates an image of the same size as 'frame', but with the template
    # warped into the correct position. Black everywhere else.
    h_frame, w_frame = frame.shape[:2]
    warped_img = warp_perspective_manual(template_image, H, (w_frame, h_frame))

    def create_polygon_mask(h, w, pts):
        """
        Creates a binary mask for a convex polygon using pure NumPy (Edge Functions).
        """
        # Get Bounding Box.
        # We only compute math for pixels inside the rectangle enclosing the tag
        # rather than the entire 1080p frame.
        pts = pts.astype(int)
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])

        # Clip to image boundaries
        min_x, max_x = max(0, min_x), min(w, max_x)
        min_y, max_y = max(0, min_y), min(h, max_y)

        # Create Grid of Coordinates (Broadcasting)
        # ogrid is faster than meshgrid for this purpose
        Y, X = np.ogrid[min_y:max_y, min_x:max_x]

        # Compute Edge Functions
        # For a point to be inside a convex polygon, it must be on the same side
        # of all lines defined by the edges.
        
        # We essentially compute the Cross Product (Determinant) for every edge
        # Value = (Px - Ax) * (By - Ay) - (Py - Ay) * (Bx - Ax)
        
        inside_mask = np.ones((max_y - min_y, max_x - min_x), dtype=bool)
        
        num_pts = len(pts)
        cross_products = []
        
        for i in range(num_pts):
            p1 = pts[i]
            p2 = pts[(i + 1) % num_pts] # Wrap around to first point
            
            # Vector for Edge (P1 -> P2)
            edge_dx = p2[0] - p1[0]
            edge_dy = p2[1] - p1[1]
            
            # Vector from P1 to every pixel (X, Y)
            # Note: X and Y are 2D grids broadcasted against scalars p1[0], p1[1]
            pixel_dx = X - p1[0]
            pixel_dy = Y - p1[1]
            
            # Cross Product (2D)
            cp = pixel_dx * edge_dy - pixel_dy * edge_dx
            cross_products.append(cp)

            # Check Winding Order
            # If points are Clockwise, all CP <= 0. If Counter-Clockwise, all CP >= 0.
            # We stack them to check all conditions at once.
            cp_stack = np.array(cross_products)
            
            # A point is inside if ALL cross products are positive OR ALL are negative
            is_inside = np.all(cp_stack >= 0, axis=0) | np.all(cp_stack <= 0, axis=0)

            # Place the small mask into the full-size mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[min_y:max_y, min_x:max_x] = is_inside.astype(np.uint8) * 255
            
        return full_mask
        
    mask = create_polygon_mask(h_frame, w_frame, dst_pts)

    # Convert mask to a boolean for easy indexing
    # We create a 3D boolean mask so it applies to all 3 color channels (BGR)
    bool_mask = (mask > 0)

    # Perform the Blend
    # Create a copy of the frame to avoid modifying the original in-place
    final_frame = frame.copy()

    # Replace pixels where the mask is True with the foreground pixels
    final_frame[bool_mask] = warped_img[bool_mask]
    return final_frame
        

def decode_tag_id(frame, corners):
    """
    Docstring for decode_tag_id
    
    Returns:
        tag_id:
        orientation:
        grid:
    """

    def gray_image_mean(img, mask=None):
        """
        Calculates the mean of a grayscale image, optionally using a mask.
        Mimics cv2.mean() behavior for single-channel images.
        
        Args:
            img: 2D numpy array (grayscale image).
            mask: Optional 2D numpy array (binary mask).
            
        Returns:
            float: The mean value.
        """
        # Case 1: No Mask
        if mask is None:
            return np.mean(img)
        
        # Case 2: With Mask
        # We only care about pixels where mask is NOT zero
        valid_pixels = img[mask != 0]
        
        # Edge Case: If mask is empty (all zeros), OpenCV returns 0.0
        if valid_pixels.size == 0:
            return 0.0
            
        return np.mean(valid_pixels)
    
    # Perspective Transform
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
        warped = to_grayscale(warped)

    warped_bin = binarization(warped)

    # Read the Grid using CENTER SAMPLING
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
            if gray_image_mean(cell_center) > 127:
                grid[y, x] = 1
            else:
                grid[y, x] = 0

    # Determine Orientation (Rotate until White Anchor is at Top-Right)
    # The Inner 4x4 grid is indices 2 to 5.
    # Top-Right of inner grid is at index [2, 5].
    
    orientation = 0
    found = False
    
    for angle in [0, 90, 180, 270]:
        # Check if the Anchor (2,5) is White (1)
        # AND check if the other corners are Black (0) to reduce false positives
        # Corners: TL(2,2), TR(2,5), BR(5,5), BL(5,2)
        is_anchor_white = (grid[5, 5] == 1)
        
        if is_anchor_white:
            orientation = angle
            found = True 
            break
        else:
            grid = np.rot90(grid) # Rotate grid counter-clockwise
            
    # If we spun 360 and didn't find the anchor, the ID will likely be garbage (or 15).
    if not found:
        return None, 0, None # Return None to indicate read failure
    
    # 5. Decode ID from Central 2x2
    # Grid: (3,3), (3,4), (4,3), (4,4)
    bit1 = grid[3, 3]
    bit2 = grid[3, 4]
    bit3 = grid[4, 4]
    bit4 = grid[4, 3]
    
    tag_id = (bit4 << 3) | (bit3 << 2) | (bit2 << 1) | bit1

    return tag_id, orientation, grid

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
    Orders points in TL, TR, BR, BL order.
    
    Args:
        pts: np.array(4,2) of points.
    
    Returns:
        pts: np.array(4,2, dtype="float32") in correct order.
    """
    # Sort the points based on their x-coordinates
    # Reshape to ensure (4, 2)
    pts = pts.reshape(4, 2)
    
    # Sort by X to separate left-side points from right-side points
    xSorted = pts[pts[:, 0].argsort()]

    # Grab the left-most and right-most points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Sort the left-most coordinates according to their y-coordinates
    # Smallest Y is Top-Left, Largest Y is Bottom-Left
    leftMost = leftMost[leftMost[:, 1].argsort()]
    (tl, bl) = leftMost

    # Sort the right-most coordinates according to their y-coordinates
    # Smallest Y is Top-Right, Largest Y is Bottom-Right
    rightMost = rightMost[rightMost[:, 1].argsort()]
    (tr, br) = rightMost

    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    return np.array([tl, tr, br, bl], dtype="float32")

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

    # close_gaps(binary_image, iterations=1)

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

def perpendicular_distance(point, line_start, line_end):
    """Calculates the distance from a point to a line segment."""
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    # Standard formula for distance from point to line (x1,y1) to (x2,y2)
    return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

        
def arc_length(contour, closed=True):
    """
    Calculates the perimeter or curve length of a contour.
    
    Args:
        contour: (N, 1, 2) or (N, 2) numpy array of coordinates.
        closed: Boolean flag to close the loop (add distance from end to start).
    """
    # Reshape to (N, 2) for easier coordinate access
    pts = contour.reshape(-1, 2)
    num_pts = len(pts)
    
    if num_pts < 2:
        return 0.0

    length = 0.0
    
    # Iterate through each pair of consecutive points
    for i in range(num_pts - 1):
        # Euclidean distance formula
        dist = np.sqrt((pts[i+1, 0] - pts[i, 0])**2 + 
                       (pts[i+1, 1] - pts[i, 1])**2)
        length += dist

    # If the curve is a closed loop, add the final segment
    if closed:
        dist_to_start = np.sqrt((pts[0, 0] - pts[-1, 0])**2 + 
                                (pts[0, 1] - pts[-1, 1])**2)
        length += dist_to_start

    return length

def get_projection_matrix(H, K):
    """
    Extracts the Rotation (R) and Translation (t) from Homography and Intrinsic K.
    Returns the 3x4 Projection Matrix P = K * [R|t].

    Args:
        H:
        K:

    Returns:
        P: 
    """
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    
    K_inv = np.linalg.inv(K)
    
    # Recover scale factor lambda
    # We use the average of the norms of the first two columns to be more robust
    lambda_1 = 1 / np.linalg.norm(np.dot(K_inv, h1))
    lambda_2 = 1 / np.linalg.norm(np.dot(K_inv, h2))
    lambda_val = (lambda_1 + lambda_2) / 2
    
    # Recover R and t
    r1 = lambda_val * np.dot(K_inv, h1)
    r2 = lambda_val * np.dot(K_inv, h2)
    t  = lambda_val * np.dot(K_inv, h3)
    
    # Recover r3 (orthogonality: r3 = r1 x r2)
    r3 = np.cross(r1, r2)
    
    # Assemble Rotation Matrix
    R = np.column_stack((r1, r2, r3))
    
    # Combine to form Extrinsics [R|t]
    extrinsics = np.column_stack((R, t))
    
    # Compute Projection Matrix P = K * [R|t]
    P = np.dot(K, extrinsics)
    return P

def manual_project_points(P, vertices):
    """
    Projects 3D vertices to 2D using matrix P.
    Replaces cv2.projectPoints.
    """
    # 1. Add homogeneous coordinate (w=1) to 3D points
    # vertices shape: (N, 3) -> (N, 4)
    ones = np.ones((vertices.shape[0], 1))
    vertices_homo = np.hstack([vertices, ones])
    
    # 2. Project: x = P * X
    # (3, 4) dot (4, N) -> (3, N) -> Transpose to (N, 3)
    projected_homo = np.dot(P, vertices_homo.T).T
    
    # 3. Normalize by Z (perspective division)
    # Avoid division by zero
    z = projected_homo[:, 2]
    z[z == 0] = 1e-10 
    
    u = projected_homo[:, 0] / z
    v = projected_homo[:, 1] / z
    
    return np.column_stack((u, v))

def load_calibration(filepath="camera_calibration.npz"):
    """
    Loads camera matrix and distortion coefficients from a file.
    
    Args:
        filepath: path of the intrinsics file.

    Returns:
        mtx: the camera matrix K.
        dist: the distortion coefficients.
    """

    
    with np.load(filepath) as data:
        mtx = data['mtx']
        dist = data['dist']
    return mtx, dist