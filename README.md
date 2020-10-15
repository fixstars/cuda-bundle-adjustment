# cuda-bundle-adjustment
A CUDA implementation of Bundle Adjustment

## Description
This project implements a Bundle Adjustment algorithm with CUDA.
It optimizes camera poses and landmarks (3D points) represented by a graph.

The reference CPU implementation is [RainerKuemmerle/g2o](https://github.com/RainerKuemmerle/g2o).
This project is designed to provide following g2o features, which are commonly used in Visual SLAM.

- `g2o::BlockSolver_6_3`
- `g2o::OptimizationAlgorithmLevenberg`
- `g2o::VertexSE3Expmap`
- `g2o::VertexSBAPointXYZ`
- `g2o::EdgeSE3ProjectXYZ`
- `g2o::EdgeStereoSE3ProjectXYZ`

For example, see [Use cuda-bundle-adjustment in ORB-SLAM2]().

## Performance

The performance obtained from `sample/sample_comparison_with_g2o` is as follows.

#### `Settings`

Key|Value
---|---
CPU / implementation|Core-i7 6700K(4.00 GHz) / g2o
GPU / implementation|GeForce GTX 1080 / cuda-bundle-adjustment
number of iterations for optimization|10

#### `Results`
input filename|P|L|E|CPU[sec]|GPU[sec]
---|---|---|---|---|---
ba_kitti_07.json|248|26127|95037|1.8|0.23
ba_kitti_00.json|1332|133383|561116|11.9|1.23

**P**: number of poses, **L**: number of landmarks, **E**: number of edges

## Limitations
Some features supported in g2o are currently simplified or not implemented.

- Information matrix is represented by a scalar
- Camera parameters are the same in all edges (assumes obserbations are done by a single camera)
- Robust kernel is not implemented
- Level optimization is not implemented

## Requirement
- CMake
- CUDA (with compute capability >= 6.0)
- Eigen
- OpenCV (for sample)
- g2o (for sample, optional)

## How to build
```
$ git clone https://github.com/fixstars/cuda-bundle-adjustment.git
$ cd cuda-bundle-adjustment
$ mkdir build
$ cd build
$ cmake ..
$ make
```

With `WITH_G2O` option, you can run `sample/sample_comparison_with_g2o`.
[g2o](https://github.com/RainerKuemmerle/g2o) needs to be installed beforehand.

```
cmake -DWITH_G2O=ON ..
```

With `USE_FLOAT32` option, 32bit float is used in internal floating-point operations (default is 64bit float).
Currently there is no significant speedup by this option.

```
cmake -DUSE_FLOAT32=ON ..
```

## How to run samples

First, extract input graph files.

```
$ cd cuda-bundle-adjustment/samples
$ 7za x ba_input.7z
```

input filename|description
---|---
ba_kitti_07.json|graph components sampled from `KITTI sequences/07` using [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
ba_kitti_00.json|graph components sampled from `KITTI sequences/00` using [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)

Then, pass to the sample code.

```
$ cd cuda-bundle-adjustment/build
$ ./samples/sample_ba_from_file/sample_ba_from_file ../samples/ba_input/ba_kitti_00.json
```

<details>
<summary>output example of sample_ba_from_file</summary>

```
$ ./samples/sample_ba_from_file/sample_ba_from_file ../samples/ba_input/ba_kitti_00.json

Reading Graph... Done.

=== Graph size :
num poses      : 1322
num landmarks  : 133383
num edges      : 561116

Running BA... Done.

=== Processing time :
BA total : 1.22[sec]

0: Initialize Optimizer        :     67.9[msec]
1: Build Structure             :     69.1[msec]
2: Compute Error               :     11.0[msec]
3: Build System                :     50.4[msec]
4: Schur Complement            :    106.2[msec]
5: Symbolic Decomposition      :    353.8[msec]
6: Numerical Decomposition     :    554.5[msec]
7: Update Solution             :      1.2[msec]

=== Objective function value :
iter:  1, chi2: 334210.0
iter:  2, chi2: 331822.8
iter:  3, chi2: 329700.4
iter:  4, chi2: 327743.4
iter:  5, chi2: 326123.2
iter:  6, chi2: 324876.6
iter:  7, chi2: 323698.5
iter:  8, chi2: 322572.7
iter:  9, chi2: 321410.3
iter: 10, chi2: 320086.4
```
</details>

<details>
<summary>output example of sample_comparison_with_g2o</summary>

```
$ ./samples/sample_comparison_with_g2o/sample_comparison_with_g2o ../samples/ba_input/ba_kitti_00.json

Reading Graph... Done.

=== Graph size :
num poses      : 1322
num landmarks  : 133383
num edges      : 561116

Running BA with CPU... Done.

Running BA with GPU... Done.

=== Processing time :
CPU :   11.93 [sec]
GPU :    1.23 [sec]

=== Objective function value :
 iteration|  chi2 CPU|  chi2 GPU
         1|  334210.0|  334210.0
         2|  331822.8|  331822.8
         3|  329700.4|  329700.4
         4|  327743.4|  327743.4
         5|  326123.2|  326123.2
         6|  324876.6|  324876.6
         7|  323698.5|  323698.5
         8|  322572.7|  322572.7
         9|  321410.3|  321410.3
        10|  320086.4|  320086.4

=== RMSE between CPU estimates and GPU estimates :
Rotation    : 7.63e-16
Translation : 4.50e-13
Landmark    : 4.50e-13
```
</details>

## Author
The "adaskit Team"  

The adaskit is an open-source project in [Fixstars Corporation](https://www.fixstars.com/) and its subsidiary companies including [Fixstars Autonomous Technologies](https://at.fixstars.com/), aimed at contributing to the ADAS industry by developing high-performance implementations for algorithms with high computational cost.

## License
Apache License 2.0
