# Reducing Drift of Lidar Odometry by Incorporating OpenStreetMap Building Data
**Disclaimer: This repository is currently under construction, more features and data will be added in the coming weeks.**

Read our paper [here](https://www.techrxiv.org/users/812931/articles/1214384-reducing-drift-of-lidar-odometry-by-incorporating-openstreetmap-building-data)


# Quickstart:
First, install the `icp_cpp` package using
```bash
pip install ./src/icp_cpp
```
(may require you to download some pip packages manually, sorry :()
and then run the following commands to
- create a map for a sequence of the KITTI odometry data set
- execute our method on it
- and evaluate the results and show a plot of the trajectory
```bash
seq=00
kitti_path=/path/to/kitti
python3 src/create_map.py --dataloader kitti --seq $seq --output_file ./res/map_kitti_$seq.bin
python3 src/Pipeline.py $kitti_path/sequences/$seq/velodyne/ --map ./res/map_kitti_$seq.bin --out_path ./out/results_kitti_$seq.json
python3 src/eval.py ./out/results_kitti_$seq.json $kitti_path/poses/$seq.txt --plot
```

# Usage:
## Execute our method with Pipeline.py
Execute our method with
```bash
python3 src/Pipeline.py <path/to/data_dir> --map <path/to/map.bin> --out_path <path/to/result.json>
```
Use `--help` for help.
The alignment against the global map is skipped if the argument `--map` in not provided.
Read "Creating a global map" for more information about creating such a global map.
You can use the `--visualize` argument for a real-time visualization of our method using RVIZ2.
For this, execute RVIZ2 and load the config from `./res/pipeline.rviz`,
then execute our method with the `--visualize` flag.


## Create a map with create_map.py
To generate a global map from OpenStreetMap data, we prove the script `create_map.py`.
For convenience, we provide a shortcut for creating maps for the KITTI, Boreas and our custom data sets.
To create the global map for sequence 00 of the KITTI odometry data set, you can simply use
```bash
python3 src/create_map.py --dataloader kitti --seq 00 --output_file map_kitti_00.bin
```

To create a map of an arbitrary point, use the `--init_pose` argument. The `--init_pose` argument takes a flattened 3x4 transformation matrix (row-major ordering) as input.
Here, the position is assumed to be in geodesic coordinates and the rotation relative to the north direction at that point.
Fer example, use
```bash
python3 src/create_map.py --init_pose 1 0 0 48.98 0 1 0 8.39 0 0 0 1 --radius 1 --output_file my_map.bin
```
to create a global map with radius of 1 degrees around the geodesic coordinate (48.98,8.39), centered at the geodesic coordinate (48.98,8.39), facing north. Use `--help` form more information.

# Installation
## Installation of the core icp_cpp module
Install the pip package icp_cpp with `pip install src/icp_cpp`.
Third party C++ dependencies are installed automatically and same pip packages may have to be installed manually (sorry for that :/)

