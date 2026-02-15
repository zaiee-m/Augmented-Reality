#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>  // <--- THIS IS MISSING OR NOT FOUND
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

// --- Helper: Generate Kernel ---
std::vector<float> create_kernel(int ksize, float sigma) {
    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    int r = ksize / 2;
    for (int i = 0; i < ksize; ++i) {
        int x = i - r;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; ++i) kernel[i] /= sum;
    return kernel;
}

static PyObject* customcv_gaussian_blur(PyObject* self, PyObject* args) {
    PyObject *in_obj, *out_obj;
    int ksize = 5;
    float sigma = 1.0f;

    if (!PyArg_ParseTuple(args, "OO|if", &in_obj, &out_obj, &ksize, &sigma)) return NULL;

    Py_buffer view_in, view_out;
    int flags = PyBUF_STRIDES | PyBUF_FORMAT;
    if (PyObject_GetBuffer(in_obj, &view_in, flags) < 0) return NULL;
    if (PyObject_GetBuffer(out_obj, &view_out, flags) < 0) {
        PyBuffer_Release(&view_in);
        return NULL;
    }

    if (view_in.itemsize != 1 || view_out.itemsize != 1 || view_in.ndim != 2 || 
        view_in.shape[0] != view_out.shape[0] || view_in.shape[1] != view_out.shape[1]) {
        PyErr_SetString(PyExc_ValueError, "Inputs must be 2D uint8 arrays of same shape.");
        PyBuffer_Release(&view_in);
        PyBuffer_Release(&view_out);
        return NULL;
    }

    Py_ssize_t h = view_in.shape[0];
    Py_ssize_t w = view_in.shape[1];
    Py_ssize_t stride_row = view_in.strides[0];
    
    unsigned char* src = (unsigned char*)view_in.buf;
    unsigned char* dst = (unsigned char*)view_out.buf;

    // Allocate Temp Buffer
    float* temp = (float*)malloc(h * w * sizeof(float));
    if (!temp) {
        PyBuffer_Release(&view_in);
        PyBuffer_Release(&view_out);
        return PyErr_NoMemory();
    }

    // --- HIGH PERFORMANCE SECTION ---
    Py_BEGIN_ALLOW_THREADS

    std::vector<float> kernel = create_kernel(ksize, sigma);
    int r = ksize / 2;
    float* k_dat = kernel.data();

    // --- PASS 1: HORIZONTAL (Branchless Center) ---
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        unsigned char* row_ptr = src + (y * stride_row);
        float* temp_ptr = temp + (y * w);

        // 1. Left Edge (Bounds Check Needed)
        for (int x = 0; x < r; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                int sx = x + k - r;
                if (sx < 0) sx = 0; 
                // Right bound check not needed here for usual sizes
                sum += row_ptr[sx] * k_dat[k];
            }
            temp_ptr[x] = sum;
        }

        // 2. CENTER (FAST PATH - NO IF STATEMENTS)
        // Compiler can SIMD this loop (process 8 pixels at once)
        for (int x = r; x < w - r; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                // Direct pointer arithmetic: Safe because we are away from edges
                sum += row_ptr[x + k - r] * k_dat[k];
            }
            temp_ptr[x] = sum;
        }

        // 3. Right Edge (Bounds Check Needed)
        for (int x = w - r; x < w; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                int sx = x + k - r;
                if (sx >= w) sx = w - 1;
                sum += row_ptr[sx] * k_dat[k];
            }
            temp_ptr[x] = sum;
        }
    }

    // --- PASS 2: VERTICAL (Tiled + Branchless Inner) ---
    int tile_w = 64; 
    #pragma omp parallel for schedule(dynamic)
    for (int x_start = 0; x_start < w; x_start += tile_w) {
        int x_end = std::min((int)(x_start + tile_w), (int)w);

        for (int y = 0; y < h; ++y) {
            unsigned char* dst_row = dst + (y * stride_row);
            
            // Note: Vertical pass is harder to make fully branchless 
            // without complex logic, but we can optimize the checks.
            
            // Standard loop with clamped logic moved out? 
            // For vertical, the 'k' loop iterates VERTICALLY.
            // The boundaries depend on 'y', not 'x'. 
            // So we can determine safe 'y' range outside.

            bool y_safe = (y >= r && y < h - r);

            if (y_safe) {
                // FAST PATH: y is safe, so 'y + k - r' is always valid
                for (int x = x_start; x < x_end; ++x) {
                    float sum = 0.0f;
                    for (int k = 0; k < ksize; ++k) {
                        int sy = y + k - r;
                        sum += temp[sy * w + x] * k_dat[k];
                    }
                    dst_row[x] = (unsigned char)std::min(std::max(sum, 0.0f), 255.0f);
                }
            } else {
                // SLOW PATH: Edge cases
                for (int x = x_start; x < x_end; ++x) {
                    float sum = 0.0f;
                    for (int k = 0; k < ksize; ++k) {
                        int sy = y + k - r;
                        if (sy < 0) sy = 0; else if (sy >= h) sy = h - 1;
                        sum += temp[sy * w + x] * k_dat[k];
                    }
                    dst_row[x] = (unsigned char)std::min(std::max(sum, 0.0f), 255.0f);
                }
            }
        }
    }

    free(temp);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&view_in);
    PyBuffer_Release(&view_out);
    Py_RETURN_NONE;
}

// --- Helper: Find Neighbor Clockwise/Counter-Clockwise ---
// Uses 1-based indexing for 'padded' image logic
static bool find_neighbor(int* img, int h, int w, int cy, int cx, int& ny, int& nx, 
                         int sy, int sx, bool clockwise, const int offsets[8][2]) {
    int dy = sy - cy;
    int dx = sx - cx;
    int start_idx = -1;

    for (int i = 0; i < 8; ++i) {
        if (offsets[i][0] == dy && offsets[i][1] == dx) {
            start_idx = i;
            break;
        }
    }
    if (start_idx == -1) return false;

    for (int i = 1; i <= 8; ++i) {
        int idx = (start_idx + (clockwise ? i : -i) + 8) % 8;
        int ty = cy + offsets[idx][0];
        int tx = cx + offsets[idx][1];
        if (img[ty * w + tx] != 0) {
            ny = ty; nx = tx;
            return true;
        }
    }
    return false;
}

struct BorderInfo {
    int parent_idx = -1;
    int first_child = -1;
    int next_sibling = -1;
    int prev_sibling = -1;
};

static PyObject* customcv_find_contours(PyObject* self, PyObject* args) {
    PyArrayObject* in_arr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_arr)) return NULL;

    int h = (int)PyArray_DIM(in_arr, 0);
    int w = (int)PyArray_DIM(in_arr, 1);
    int ph = h + 2, pw = w + 2;
    std::vector<int> padded(ph * pw, 0);
    unsigned char* src_data = (unsigned char*)PyArray_DATA(in_arr);

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if (src_data[i * w + j] > 0) padded[(i + 1) * pw + (j + 1)] = 1;
        }
    }

    int nbd = 1; // nbd=1 is background
    const int offsets[8][2] = {{-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}};
    
    PyObject* py_contours = PyList_New(0);
    std::vector<BorderInfo> hierarchy_tree;
    hierarchy_tree.push_back({}); // Placeholder for background (ID 1)

    for (int i = 1; i < ph - 1; ++i) {
        int lnbd = 1;
        for (int j = 1; j < pw - 1; ++j) {
            int& f_ij = padded[i * pw + j];
            if (f_ij == 0) continue;

            bool is_outer = (f_ij == 1 && padded[i * pw + (j - 1)] == 0);
            bool is_hole  = (f_ij >= 1 && padded[i * pw + (j + 1)] == 0);

            if (is_outer || is_hole) {
                nbd++;
                int current_parent = -1;

                // --- Hierarchy Logic ---
                if (is_outer) {
                    // Outer border: parent is LNBD if LNBD is a hole, else parent of LNBD
                    // In a simplified tree, LNBD is the container
                    current_parent = lnbd;
                } else {
                    // Hole border: parent is the border we are currently on
                    current_parent = (f_ij > 1) ? f_ij : lnbd;
                }

                // Update sibling/child pointers for hierarchy
                BorderInfo info;
                info.parent_idx = current_parent;
                int nbd_idx = nbd - 1; // 0-based index for hierarchy array
                int parent_idx_0 = current_parent - 1;

                if (hierarchy_tree[parent_idx_0].first_child == -1) {
                    hierarchy_tree[parent_idx_0].first_child = nbd_idx;
                } else {
                    int sibling = hierarchy_tree[parent_idx_0].first_child;
                    while (hierarchy_tree[sibling].next_sibling != -1) {
                        sibling = hierarchy_tree[sibling].next_sibling;
                    }
                    hierarchy_tree[sibling].next_sibling = nbd_idx;
                    info.prev_sibling = sibling;
                }
                hierarchy_tree.push_back(info);

                // --- Trace Border ---
                int ny, nx, sy = i, sx = (is_outer ? j - 1 : j + 1);
                std::vector<int> pts;
                if (!find_neighbor(padded.data(), ph, pw, i, j, ny, nx, sy, sx, true, offsets)) {
                    padded[i * pw + j] = -nbd;
                    pts.push_back(j - 1); pts.push_back(i - 1);
                } else {
                    int p2y = ny, p2x = nx, p3y = i, p3x = j, fny = ny, fnx = nx;
                    while (true) {
                        int p4y, p4x;
                        find_neighbor(padded.data(), ph, pw, p3y, p3x, p4y, p4x, p2y, p2x, false, offsets);
                        if (padded[p3y * pw + (p3x + 1)] == 0) padded[p3y * pw + p3x] = -nbd;
                        else if (padded[p3y * pw + p3x] == 1) padded[p3y * pw + p3x] = nbd;
                        pts.push_back(p3x - 1); pts.push_back(p3y - 1);
                        if (p4y == i && p4x == j && p3y == fny && p3x == fnx) break;
                        p2y = p3y; p2x = p3x; p3y = p4y; p3x = p4x;
                    }
                }

                npy_intp out_dims[3] = {(npy_intp)(pts.size() / 2), 1, 2};
                PyObject* contour_arr = PyArray_SimpleNew(3, out_dims, NPY_INT32);
                memcpy(PyArray_DATA((PyArrayObject*)contour_arr), pts.data(), pts.size() * sizeof(int));
                PyList_Append(py_contours, contour_arr);
                Py_DECREF(contour_arr);
            }
            if (f_ij != 1) lnbd = std::abs(f_ij);
        }
    }

    // Convert hierarchy to (1, N, 4) array (skipping background ID 1)
    int num_found = nbd - 1;
    npy_intp h_dims[3] = {1, num_found, 4};
    PyObject* py_hierarchy = PyArray_SimpleNew(3, h_dims, NPY_INT32);
    int* h_data = (int*)PyArray_DATA((PyArrayObject*)py_hierarchy);

for (int k = 0; k < num_found; ++k) {
    BorderInfo& b = hierarchy_tree[k + 1];
    
    // Only subtract 1 if the index is valid (>= 0). 
    // If it's -1, keep it -1.
    h_data[k * 4 + 0] = (b.next_sibling >= 0) ? b.next_sibling - 1 : -1;
    h_data[k * 4 + 1] = (b.prev_sibling >= 0) ? b.prev_sibling - 1 : -1;
    h_data[k * 4 + 2] = (b.first_child >= 0)  ? b.first_child - 1  : -1;
    
    // For Parent: Since parent_idx stores raw NBD values (2, 3, 4...)
    // and background is 1, a root contour (parent 1) should result in -1.
    h_data[k * 4 + 3] = (b.parent_idx > 1) ? b.parent_idx - 2 : -1;
}

    PyObject* result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, py_contours);
    PyTuple_SetItem(result, 1, py_hierarchy);
    return result;
}

struct Point {
    int x, y;
};

// --- Helper: Squared Euclidean Distance ---
double distSq(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx*dx + dy*dy;
}

// --- Helper: Perpendicular Distance Squared ---
// Returns distance from point P to line segment AB
double getPointLineDistSq(Point p, Point a, Point b) {
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    
    if (dx == 0 && dy == 0) {
        return distSq(p, a);
    }

    double numerator = (double)((dy * p.x) - (dx * p.y) + (b.x * a.y) - (b.y * a.x));
    return (numerator * numerator) / (dx*dx + dy*dy);
}

// --- Internal RDP Logic (Open Chain) ---
void rdp_open(const std::vector<Point>& points, double epsilon, std::vector<Point>& out_pts) {
    int m = points.size();
    if (m < 3) {
        out_pts = points;
        return;
    }

    std::vector<bool> keep(m, false);
    keep[0] = true;
    keep[m - 1] = true;

    // Iterative Stack-based RDP (avoids recursion depth issues)
    std::stack<std::pair<int, int>> stack;
    stack.push({0, m - 1});
    
    double epsilon_sq = epsilon * epsilon;

    while (!stack.empty()) {
        std::pair<int, int> range = stack.top();
        stack.pop();

        int start = range.first;
        int end = range.second;

        if (end <= start + 1) continue;

        double max_dist_sq = 0.0;
        int max_idx = 0;

        // Find point farthest from line segment
        for (int i = start + 1; i < end; ++i) {
            double d = getPointLineDistSq(points[i], points[start], points[end]);
            if (d > max_dist_sq) {
                max_dist_sq = d;
                max_idx = i;
            }
        }

        if (max_dist_sq > epsilon_sq) {
            keep[max_idx] = true;
            stack.push({max_idx, end});
            stack.push({start, max_idx});
        }
    }

    // Collect results
    out_pts.reserve(m);
    for (int i = 0; i < m; ++i) {
        if (keep[i]) out_pts.push_back(points[i]);
    }
}

static PyObject* customcv_approx_poly_dp(PyObject* self, PyObject* args) {
    PyObject* in_obj;
    double epsilon;
    int closed = 0;

    if (!PyArg_ParseTuple(args, "Odi", &in_obj, &epsilon, &closed)) return NULL;

    PyArrayObject* in_arr = (PyArrayObject*)PyArray_FROM_OTF(in_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (!in_arr) return NULL;

    int num_points = (int)PyArray_DIM(in_arr, 0);
    int* data = (int*)PyArray_DATA(in_arr);

    // Safety check for tiny contours
    if (num_points < 3) {
        Py_INCREF(in_arr);
        return (PyObject*)in_arr;
    }

    std::vector<Point> pts(num_points);
    long long sum_x = 0, sum_y = 0;

    // 1. Load Data & Calculate Centroid
    for (int i = 0; i < num_points; ++i) {
        pts[i].x = data[i * 2];
        pts[i].y = data[i * 2 + 1];
        if (closed) {
            sum_x += pts[i].x;
            sum_y += pts[i].y;
        }
    }

    // --- CLOSED LOOP STABILIZATION (Your "Secret Sauce") ---
    if (closed) {
        // A. Filter redundant end point if input is already closed
        if (pts.size() > 1 && pts.front().x == pts.back().x && pts.front().y == pts.back().y) {
            pts.pop_back();
            num_points--;
            // Recalculate sums if strictly needed, but usually negligible for centroid
        }

        // B. Find Centroid
        Point center;
        center.x = (int)(sum_x / num_points);
        center.y = (int)(sum_y / num_points);

        // C. Find index farthest from centroid
        double max_d2 = -1.0;
        int best_idx = 0;
        for (int i = 0; i < num_points; ++i) {
            double d2 = distSq(pts[i], center);
            if (d2 > max_d2) {
                max_d2 = d2;
                best_idx = i;
            }
        }

        // D. Rotate vector so best_idx is at index 0
        std::rotate(pts.begin(), pts.begin() + best_idx, pts.end());

        // E. Make explicit closed loop (append first point to end)
        pts.push_back(pts[0]);
    }

    // --- Run RDP ---
    std::vector<Point> simp;
    rdp_open(pts, epsilon, simp);

    // --- Cleanup Closed Loop ---
    if (closed && simp.size() > 1) {
        Point first = simp.front();
        Point last = simp.back();
        
        // If start and end are effectively the same, remove the duplicate
        if (distSq(first, last) <= epsilon * epsilon) {
            simp.pop_back();
        }
    }

    // --- Output to NumPy ---
    npy_intp out_dims[3] = {(npy_intp)(simp.size()), 1, 2};
    PyObject* out_arr = PyArray_SimpleNew(3, out_dims, NPY_INT32);
    int* out_data = (int*)PyArray_DATA((PyArrayObject*)out_arr);

    for (size_t i = 0; i < simp.size(); ++i) {
        out_data[i * 2 + 0] = simp[i].x;
        out_data[i * 2 + 1] = simp[i].y;
    }

    Py_DECREF(in_arr);
    return out_arr;
}


// Module boilerplate
static PyMethodDef customcvMethods[] = {
    {"gaussian_blur", (PyCFunction)customcv_gaussian_blur, METH_VARARGS, "Optimized Gaussian Blur"},
    {"find_contours", (PyCFunction)customcv_find_contours, METH_VARARGS, "Suzuki-Abe Contour Detection"},
    {"approx_poly_dp", (PyCFunction)customcv_approx_poly_dp, METH_VARARGS, "Douglas-Peucker Polygon Approximation"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef customcv_module = { PyModuleDef_HEAD_INIT, "customcv", NULL, 0, customcvMethods };
extern "C" {
    PyMODINIT_FUNC PyInit_customcv(void) {
        // CRITICAL: Initialize NumPy API table
        // Without this, any PyArray_* call crashes the program.
        import_array(); 
        
        return PyModuleDef_Init(&customcv_module);
    }
}