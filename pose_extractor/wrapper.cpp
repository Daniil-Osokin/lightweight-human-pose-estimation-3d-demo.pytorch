#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <opencv2/core/core.hpp>

#include "extract_poses.hpp"

static std::vector<cv::Mat> wrap_feature_maps(PyArrayObject* py_feature_maps) {
    int num_channels = static_cast<int>(PyArray_SHAPE(py_feature_maps)[0]);
    int h = static_cast<int>(PyArray_SHAPE(py_feature_maps)[1]);
    int w = static_cast<int>(PyArray_SHAPE(py_feature_maps)[2]);
    float* data = static_cast<float*>(PyArray_DATA(py_feature_maps));
    std::vector<cv::Mat> feature_maps(num_channels);
    for (long c_id = 0; c_id < num_channels; c_id++)
    {
        feature_maps[c_id] = cv::Mat(h, w, CV_32FC1,
                                     data + c_id * PyArray_STRIDE(py_feature_maps, 0) / sizeof(float),
                                     PyArray_STRIDE(py_feature_maps, 1));
    }
    return feature_maps;
}

static PyObject* extract_poses(PyObject* self, PyObject* args) {
    PyArrayObject* py_heatmaps;
    PyArrayObject* py_pafs;
    int ratio;
    if (!PyArg_ParseTuple(args, "OOi", &py_heatmaps, &py_pafs, &ratio))
    {
        throw std::runtime_error("passed non-numpy array as argument");
    }
    std::vector<cv::Mat> heatmaps = wrap_feature_maps(py_heatmaps);
    std::vector<cv::Mat> pafs = wrap_feature_maps(py_pafs);

    std::vector<human_pose_estimation::HumanPose> poses = human_pose_estimation::extractPoses(
                heatmaps, pafs, ratio);

    size_t num_persons = poses.size();
    size_t num_keypoints = 0;
    if (num_persons > 0) {
        num_keypoints = poses[0].keypoints.size();
    }
    float* out_data = new float[num_persons * (num_keypoints * 3 + 1)];
    for (size_t person_id = 0; person_id < num_persons; person_id++)
    {
        for (size_t kpt_id = 0; kpt_id < num_keypoints * 3; kpt_id += 3)
        {
            out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 0] = poses[person_id].keypoints[kpt_id / 3].x;
            out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 1] = poses[person_id].keypoints[kpt_id / 3].y;
            out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 2] = poses[person_id].keypoints[kpt_id / 3].z;
        }
        out_data[person_id * (num_keypoints * 3 + 1) + num_keypoints * 3] = poses[person_id].score;
    }
    npy_intp dims[] = {static_cast<npy_intp>(num_persons),
                       static_cast<npy_intp>(num_keypoints * 3 + 1)};
    PyObject *pArray = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, reinterpret_cast<void*>(out_data));
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(pArray), NPY_ARRAY_OWNDATA);
    return Py_BuildValue("(N)", pArray);
}

PyMethodDef method_table[] = {
    {"extract_poses", static_cast<PyCFunction>(extract_poses), METH_VARARGS,
     "Extracts 2d poses from provided heatmaps and pafs"},
    {NULL, NULL, 0, NULL}
};

PyModuleDef pose_extractor_module = {
    PyModuleDef_HEAD_INIT,
    "pose_extractor",
    "Module for fast 2d pose extraction",
    -1,
    method_table
};

PyMODINIT_FUNC PyInit_pose_extractor(void) {
    PyObject* module = PyModule_Create(&pose_extractor_module);
    if (module == nullptr)
    {
        return nullptr;
    }
    import_array();
    if (PyErr_Occurred())
    {
        return nullptr;
    }

    return module;
}
