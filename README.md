# Reducing Drift of Lidar Odometry by Incorporating OpenStreetMap Building Data
Read our paper [here](https://www.techrxiv.org/users/812931/articles/1214384-reducing-drift-of-lidar-odometry-by-incorporating-openstreetmap-building-data)

Disclaimer: 
- This repository is currently under construction and is missing many quality of life features for installation and execution.
- This version currently only supports the KITTI odometry file format. Other dataloader will follow.
- This version currently does not contain the script for building custom maps. We have included pre-built maps for the KITTI odometry data set

# Usage:
Execute our method with
```bash
python3 src/Pipeline.py <path/to/data_dir> --map <path/to/map.bin> --out_path <path/to/result.json>
```
Use `--help` for help.
The alignment against the global map is skipped if the argument `--map` in not provided.
You can use the `--visualize` argument for a real-time visualization of our method using RVIZ2.
For this, execute RVIZ2 and load the config from `./res/pipeline.rviz`,
then execute our method with the `--visualize` flag.

## Examples:
### Running our method on the KITTI odometry data set:

The following code executes our method on sequence 00 of the kitti odometry data set, using the pre-built map from `./res/map_kitti_00.bin` and saving the results to `./out/results_00.json`
```bash
python3 src/Pipeline.py /path/to/kitti_odometry/sequences/00/velodyne/ --map ./res/map_kitti_00.bin --out_path ./out/results_00.json
```

### Evaluating our method on the KITTI odometry data set
Execute
```bash
python3 src/eval.py ./out/results_00.json /path/to/kitti_odometry/poses/00.txt
```
to evaluate the result of our method on sequence 00 of the KITTI odometry data set against the ground truth.
Use the argument `--plot` to see an interactive plot of the estimated and ground truth trajectories.


# Installation
## Installation of the core icp_cpp module
Install the pip package icp_cpp with `pip install src/icp_cpp`.
Third party C++ dependencies are installed automatically:
- Eigen
- Sophus
- Pybind11
- oneTBB
## Installation of python dependencies
- Numpy
- Scipy
- pykitti
- matplotlib
- and probably others


