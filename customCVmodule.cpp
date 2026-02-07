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

static PyObject* customCV_gaussian_blur(PyObject* self, PyObject* args) {
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

struct Point {
    int x, y;
};

// --- Helper: Cross Product of (b-a) and (c-a) ---
long long cross_product(Point a, Point b, Point c) {
    return (long long)(b.x - a.x) * (c.y - a.y) - (long long)(b.y - a.y) * (c.x - a.x);
}

// --- Helper: Distance Squared ---
long long dist_sq(Point a, Point b) {
    long long dx = a.x - b.x;
    long long dy = a.y - b.y;
    return dx*dx + dy*dy;
}

// --- Algorithm: Monotone Chain Convex Hull ---
std::vector<Point> get_convex_hull(const std::vector<Point>& points) {
    int n = points.size();
    if (n <= 2) return points;

    std::vector<Point> sorted_pts = points;
    // Sort by X, then Y
    std::sort(sorted_pts.begin(), sorted_pts.end(), [](Point a, Point b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });

    std::vector<Point> hull;

    // Lower hull
    for (const auto& p : sorted_pts) {
        while (hull.size() >= 2 && cross_product(hull[hull.size()-2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // Upper hull
    size_t lower_size = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        const auto& p = sorted_pts[i];
        while (hull.size() > lower_size && cross_product(hull[hull.size()-2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // Remove duplicate start point
    if (hull.size() > 1) hull.pop_back();
    
    return hull;
}

// --- Algorithm: Largest Quadrilateral on Hull ---
std::vector<Point> find_largest_quad(const std::vector<Point>& hull) {
    int n = hull.size();
    if (n < 4) return hull; // Should handle strictly in wrapper, but safety first

    // 1. Find the two points with max distance (The Main Diagonal)
    int p1_idx = 0, p2_idx = 0;
    long long max_d2 = 0;

    // Brute force is fine for Hull (N usually < 50)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            long long d2 = dist_sq(hull[i], hull[j]);
            if (d2 > max_d2) {
                max_d2 = d2;
                p1_idx = i;
                p2_idx = j;
            }
        }
    }

    Point p1 = hull[p1_idx];
    Point p2 = hull[p2_idx];

    // 2. Find points furthest from this line on both sides
    // Logic: Maximize Cross Product (Area of triangle formed by diagonal + point)
    long long max_pos_cross = -1;
    long long max_neg_cross = 1; 
    int p3_idx = -1;
    int p4_idx = -1;

    for (int i = 0; i < n; i++) {
        long long cp = cross_product(p1, p2, hull[i]);
        if (cp > max_pos_cross) {
            max_pos_cross = cp;
            p3_idx = i;
        }
        if (cp < max_neg_cross) {
            max_neg_cross = cp;
            p4_idx = i;
        }
    }

    // If points are collinear or something failed, fallback to indices
    if (p3_idx == -1) p3_idx = (p1_idx + 1) % n;
    if (p4_idx == -1) p4_idx = (p2_idx + 1) % n;

    std::vector<Point> quad = {p1, hull[p3_idx], p2, hull[p4_idx]};

    // 3. Sort Corners (TL, TR, BR, BL)
    // Calc Center
    double cx = 0, cy = 0;
    for(auto& p : quad) { cx += p.x; cy += p.y; }
    cx /= 4.0; cy /= 4.0;

    std::sort(quad.begin(), quad.end(), [cx, cy](Point a, Point b) {
        return std::atan2(a.y - cy, a.x - cx) < std::atan2(b.y - cy, b.x - cx);
    });

    return quad;
}

// --- Python Wrapper: find_quads(contours, min_area) ---
// ... imports and helper functions (get_convex_hull, etc.) remain the same ...

static PyObject* customCV_find_quads(PyObject* self, PyObject* args) {
    PyObject *contours_list;
    float min_area = 100.0f;

    if (!PyArg_ParseTuple(args, "O|f", &contours_list, &min_area)) return NULL;
    if (!PyList_Check(contours_list)) {
        PyErr_SetString(PyExc_TypeError, "contours must be a list");
        return NULL;
    }

    Py_ssize_t num_contours = PyList_Size(contours_list);
    PyObject* result_list = PyList_New(0);
    
    printf("\n--- C++ DEBUG: Processing %zd contours ---\n", num_contours);

    for (Py_ssize_t i = 0; i < num_contours; i++) {
        PyObject* cnt_obj = PyList_GetItem(contours_list, i);
        
        // 1. Force conversion to (N, 2) int32 array
        PyArrayObject* cnt_arr = (PyArrayObject*)PyArray_FROM_OTF(cnt_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
        
        if (!cnt_arr) {
            printf("Contour %zd: INVALID (Not an array)\n", i);
            PyErr_Clear();
            continue;
        }

        // 2. Check Dimensions
        int ndim = PyArray_NDIM(cnt_arr);
        npy_intp* dims = PyArray_DIMS(cnt_arr);
        int n_pts = (int)dims[0]; // Assuming dim 0 is length

        // Allow (N, 2) OR (N, 1, 2)
        bool valid_shape = false;
        if (ndim == 2 && dims[1] == 2) valid_shape = true;
        else if (ndim == 3 && dims[1] == 1 && dims[2] == 2) valid_shape = true;

        if (!valid_shape) {
            printf("Contour %zd: REJECTED (Bad Shape: %d dims)\n", i, ndim);
            Py_DECREF(cnt_arr);
            continue;
        }

        int* data = (int*)PyArray_DATA(cnt_arr);

        // 3. Calc Area
        std::vector<Point> pts(n_pts);
        double area = 0.0;
        
        for (int j = 0; j < n_pts; j++) {
            pts[j] = {data[j*2], data[j*2+1]};
            if (j > 0) area += (double)(pts[j-1].x * pts[j].y - pts[j].x * pts[j-1].y);
        }
        area += (double)(pts[n_pts-1].x * pts[0].y - pts[0].x * pts[n_pts-1].y);
        area = std::abs(area) * 0.5;

        // 4. Filter
        if (area < min_area) {
            // Uncomment to see small noise
            // printf("Contour %zd: SKIPPED (Area %.1f < %.1f)\n", i, area, min_area);
            Py_DECREF(cnt_arr);
            continue;
        }

        // 5. Convex Hull
        std::vector<Point> hull = get_convex_hull(pts);

        if (hull.size() < 4) {
            printf("Contour %zd: REJECTED (Hull has %zd points, need 4+)\n", i, hull.size());
            Py_DECREF(cnt_arr);
            continue;
        }

        // 6. Success!
        printf("Contour %zd: ACCEPTED! (Area: %.1f, Hull: %zd pts)\n", i, area, hull.size());

        std::vector<Point> quad = find_largest_quad(hull);
        
        npy_intp out_dims[2] = {4, 2};
        PyObject* out_arr = PyArray_SimpleNew(2, out_dims, NPY_INT32);
        int* out_data = (int*)PyArray_DATA((PyArrayObject*)out_arr);

        for(int k=0; k<4; k++) {
            out_data[k*2] = quad[k].x;
            out_data[k*2+1] = quad[k].y;
        }
        PyList_Append(result_list, out_arr);
        Py_DECREF(out_arr);
        Py_DECREF(cnt_arr);
    }
    
    printf("--- End Debug ---\n");
    return result_list;
}

// Module boilerplate
static PyMethodDef customcvMethods[] = {
    {"gaussian_blur", (PyCFunction)customCV_gaussian_blur, METH_VARARGS, "Optimized Gaussian Blur"},
    {"find_quads", (PyCFunction)customCV_find_quads, METH_VARARGS, "Filter contours and extract 4 corners"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef customCV_module = { PyModuleDef_HEAD_INIT, "customCV", NULL, 0, customcvMethods };
extern "C" {
    PyMODINIT_FUNC PyInit_customCV(void) {
        // CRITICAL: Initialize NumPy API table
        // Without this, any PyArray_* call crashes the program.
        import_array(); 
        
        return PyModuleDef_Init(&customCV_module);
    }
}