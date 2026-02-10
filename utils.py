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

def extract_and_draw_final(frame, resizing_factor=1):
    
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
    
    # 5. Edge Detection
    gradient = detect_edges_binary(binary)
    # gradient = cv2.Canny(blurred,100,200)
    
    # --- B. EXTRACT CONTOURS ---
    contours, hierarchy = customCV.find_contours(gradient)

    tags = extract_tags(contours,hierarchy)


    corrected_contours = []

    for cnt in tags:
        # 1. Swap columns from [y, x] to [x, y]
        # [:, ::-1] is a fast way to reverse the last dimension
        # cnt_xy = cnt[:, ::-1]*resizing_factor
        
        # 2. Reshape to (N, 1, 2) and ensure it's int32
        cnt_formatted = cnt.reshape((-1, 1, 2)).astype(np.int32)
        
        corrected_contours.append(cnt_formatted)

    # l = []
    # corners = []

    # # Loop through the hierarchy array (shape (1, N, 4))
    # for i in range(len(contours)):

    #     # Filter 1: Look for contours with a parent (Parent index is at hierarchy[0][i][3])
    #     # Suzuki-Abe parent index != -1 means it is a child/hole
    #     parent_id = hierarchy[0][i][3]
        
    #     if parent_id != -1:

    #         # Filter 2: Centroid Check
    #         # Ensure the child and parent share a similar center of mass
    #         M1 = cv2.moments(contours[i])
    #         M2 = cv2.moments(contours[parent_id])
            
    #         if M1["m00"] != 0 and M2["m00"] != 0:
    #             c1 = np.array([int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])])
    #             c2 = np.array([int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])])
                
    #             # Distance threshold to verify alignment (perspective robust)
    #             if np.linalg.norm(c1 - c2) < 15:

    #                 # Filter 3: Check inner portion complexity (10 points specified)
    #                 # approxPolyDP uses Douglas-Peucker to simplify vertices
    #                 p1 = arc_length(contours[i], True)
    #                 approx1 = cv2.approxPolyDP(contours[i], 0.02 * p1, True)
                    
    #                 if len(approx1)>4:

    #                     # Filter 4: Look for quadrilateral parents (the marker border)
    #                     p2 = arc_length(contours[parent_id], True)
    #                     approx2 = cv2.approxPolyDP(contours[parent_id], 0.02 * p2, True)
                        
    #                     if len(approx2) == 4:
    #                         l.append(i)
    #                         l.append(parent_id)
    #                         corners.append(approx1)
    #                         corners.append(approx2)


    # filteredContours = [contours[idx] for idx in l]
    # rescaled_contours = [(cnt * resizing_factor).astype(np.int32) for cnt in filteredContours]
    cv2.drawContours(frame,corrected_contours,-1,(0,255,0),3)

    # rescaled_contours = [(cnt * resizing_factor).astype(np.int32) for cnt in contours]
    # cv2.drawContours(frame, rescaled_contours, -1, (0, 255, 0), 3)
    # return frame

    # --- C. GEOMETRIC FILTERING ---
    # Filter small noise based on resizing factor
    min_area_thresh = 100 if resizing_factor == 1 else 100/resizing_factor
    

    # --- D. DECODING & DRAWING ---
    output_frame = frame.copy()
    for tag in corrected_contours:
        # 'tag' is a numpy array of [row, col] (y, x) in DOWNSCALED coordinates.
        
        # 1. DECODE ID
        # We use the small binary image and small coordinates for decoding.
        # We must flip [y, x] -> [x, y] for the Homography logic.
        # tag_full_xy = (tag * resizing_factor).astype(np.float32)
        # Flip [row, col] -> [x, y] for homography
        # tag_full_xy = tag_full_xy[:, ::-1] 
        
        # tag_full_xy = order_points(tag_full_xy)

        # Pass the small binary image to read the bits
        # return decode_tag_id(frame, tag_full_xy)
        tag_id, angle = decode_tag_id(frame, tag)
    
        # 2. PREPARE DRAWING COORDINATES
        # Upscale: Multiply by resizing factor to map back to 1080p
        # Flip: Ensure we have [x, y] for OpenCV drawing functions
        # upscaled_tag = (tag * resizing_factor).astype(np.int32)
        
        # Reshape to standard OpenCV format: (N, 1, 2)
        # We slice [:, ::-1] to flip Y,X to X,Y
        # draw_pts = upscaled_tag[:, ::-1].reshape((-1, 1, 2))
        
        # 3. DRAW THE GREEN BOX
        # cv2.polylines(output_frame, [upscaled_tag], isClosed=True, color=(0, 255, 0), thickness=5)

        dest_corners = tag.reshape(4, 2).astype(np.float32)
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
            
        #     # Optional: Draw Orientation Corner (Blue Dot at Top-Right)
        #     # You might need to rotate this point based on 'angle' to show true "Up"

        #     # Visualizing the Orientation (Blue Dot on the Tag's Top-Right Corner)
        #     if angle == 0:
        #         tr_index = 3
        #     elif angle == 90:
        #         tr_index = 2
        #     elif angle == 180:
        #         tr_index = 1  # Physical TR is now at Screen Bottom-Left
        #     else: # angle == 270
        #         tr_index = 0 # Physical TR is now at Screen Bottom-Right

        #     # Extract point and ensure it's a tuple of integers
        #     # draw_pts is shape (4, 1, 2), so we access [index][0]
        #     tr_point = draw_pts[tr_index][0]
        #     corner_pt = tuple(tr_point) # Assuming index 1 is TR
        #     cv2.circle(output_frame, corner_pt, 10, (255, 0, 0), -1)

    return output_frame

def extract_tags(contours, hierarchy):
    """
    Analyzes contours and hierarchy to find nested tags and return ordered corners.
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

            # Filter 2: Centroid Check
            M1 = cv2.moments(contours[i])
            M2 = cv2.moments(contours[parent_id])
            
            if M1["m00"] != 0 and M2["m00"] != 0:
                c1 = np.array([M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]])
                c2 = np.array([M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]])
                
                # Verify centers are close (within 15 pixels)
                if np.linalg.norm(c1 - c2) < 15:

                    # Filter 3: Check inner portion complexity
                    p1 = cv2.arcLength(contours[i], True)
                    approx1 = cv2.approxPolyDP(contours[i], 0.02 * p1, True)
                    
                    if len(approx1) > 4:

                        # Filter 4: Look for quadrilateral parents (the tag border)
                        p2 = cv2.arcLength(contours[parent_id], True)
                        approx2 = cv2.approxPolyDP(contours[parent_id], 0.02 * p2, True)
                        
                        if len(approx2) == 4:
                            # Order the 4 corners of the parent (the tag)
                            ordered_corners = order_points(approx2)
                            all_tag_corners.append(ordered_corners)
                            
                            # Mark parent as processed to avoid duplicate detection
                            processed_parents.add(parent_id)

    return all_tag_corners

def superimpose_image(frame, tag_corners, template_image, orientation_angle):
    """
    Overlays a template image onto the detected AR tag, respecting orientation.
    """
    if template_image is None:
        return frame
        
    # 1. Prepare Source Points (The Template Image Corners)
    # Order: Top-Left, Bottom-Left, Bottom-Right, Top-Right.
    h_temp, w_temp = template_image.shape[:2]
    src_pts = np.array([
        [0, 0],                     # Index 0: Top-Left
        [w_temp - 1, 0],            # Index 1: Top-Right
        [w_temp - 1, h_temp - 1],   # Index 2: Bottom-Right
        [0, h_temp - 1]             # Index 3: Bottom-Left
    ], dtype=np.float32)

    # 2. Prepare Destination Points (The Tag on Screen)
    # We must order the detected corners to match the template order (TL, BL, BR, TR).
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

    # Logic:
    # 0 deg   -> Anchor is TR. No shift needed.
    # 90 deg  -> Anchor is BR. Shift 1.
    # 180 deg -> Anchor is BL. Shift 2.
    # 270 deg -> Anchor is TL. Shift 3.
    
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

    # return warped

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
            print(angle)
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

    # M = cv2.moments(dst_pts)
    # if M["m00"] != 0:
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    
    #     text = f"A: {angle}"
        
    #     # Draw black outline for text readability
    #     cv2.putText(warped_bin, text, (cX - 40, cY), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    #     # Draw red text
    #     cv2.putText(warped_bin, text, (cX - 40, cY), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # return warped_bin
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
    Orders coordinates: Top-Left, Bottom-Left, Bottom-Right, Top-Right.

    Expects inputs in xy format and returns in xy aswell.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)
    
    # Sum: TL has min sum, BR has max sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Diff: BL has min diff (x-y), TR has max diff (x-y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

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

def perpendicular_distance(point, line_start, line_end):
    """Calculates the distance from a point to a line segment."""
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    # Standard formula for distance from point to line (x1,y1) to (x2,y2)
    return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

def approx_poly_dp(points, epsilon):
    """
    Simplifies a list of points using the Douglas-Peucker algorithm.
    points: Nx2 numpy array
    epsilon: distance threshold for simplification
    """
    if len(points) < 3:
        return points

    # Find the point with the maximum distance from the line connecting start and end
    dmax = 0
    index = 0
    start_point = points[0]
    end_point = points[-1]

    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], start_point, end_point)
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive calls
        left_side = approx_poly_dp(points[:index+1], epsilon)
        right_side = approx_poly_dp(points[index:], epsilon)

        # Build the result list (avoid duplicating the middle point)
        return np.vstack((left_side[:-1], right_side))
    else:
        # All points are close enough; just return the endpoints
        return np.array([start_point, end_point])
        
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