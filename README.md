<h1 align="center">SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using Large Language Models</h1>

<div align="center" margin-bottom="1em">
<a href="https://nzantout.github.io">Nader Zantout<sup>✶</sup></a>,
<a href="https://HaochenZ11.github.io">Haochen Zhang<sup>✶</sup></a>,
<a href="https://sites.google.com/view/pujith-kachana/">Pujith Kachana</a>,
<a href="https://www.jinkaiq.com/">Jinkai Qiu</a>,
<a href="https://gfchen01.cc/">Guofei Chen</a>,
<a href="https://frc.ri.cmu.edu/~zhangji/">Ji Zhang</a>,
<a href="http://www.wangwenshan.com/">Wenshan Wang</a>
<br>
<sup>* </sup>Equal contribution<br>
</div>
&nbsp;
<div align="center" margin-bottom="1em">
    <a href="https://arxiv.org/abs/2504.18684" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://youtu.be/Jhd_ThwBSGo" target="_blank">
    <img src="https://img.shields.io/badge/Video-YouTube-9966ff" alt="Video"></a>
</div>
&nbsp;

We propose **SORT3D**, an LLM-based object-centric grounding and indoor navigation system employing a spatial reasoning toolbox and state-of-the-art 2D VLMs for perception. The toolbox is capable of interpreting both direct and indirect statements about spatial relations, using an LLM for high-level reasoning and guiding the autonomous robot to navigate through the environment. It has demonstrated the best zero-shot performance on spatial reasoning benchmarks. To the best of our knowledge, this is the first implementation of a general spatial relation toolbox for autonomous vision-language navigation that is fully integrated into real-robot systems.

&nbsp;

<div align="center"><img src="media/diagram.png" alt="SORT3D Diagram" width="99%"></div>

&nbsp;

https://github.com/user-attachments/assets/20865dc0-1ffc-4d72-9975-508687dbbe76



This repository is set up to run both grounding evaluation on the [ReferIt3D](https://referit3d.github.io) and [VLA-3D](https://github.com/HaochenZ11/VLA-3D) benchmarks and online navigation, on both real robots and provided simulated environments. We also provide a [dataset](#dataset) of Scannet object crops and captions generated using our pipeline.

## Updates

- [2025-06] SORT3D is accepted to IROS 2025!
- [2025-03] We release SORT3D for offline grounding and online object-centric navigation. 

-----

### Table of Contents


- [Repository Structure](#repository-structure)
- [Data](#data)
  - [Dataset For SORT3D-Bench](#dataset-for-sort3d-bench)
  - [ROS Bag Files for SORT3D-Nav](#ros-bag-files-for-sort3d-nav)
- [System Requirements](#system-requirements)
  - [Hardware Requirements](#hardware-requirements)
  - [Operating System](#operating-system)
- [SORT3D-Bench: Setup](#sort3d-bench-setup)
  - [1) Conda Environment](#1-conda-environment)
  - [2) Dataset Setup](#2-dataset-setup)
- [SORT3D-Bench: Usage](#sort3d-bench-usage)
- [SORT3D-Nav: Setup](#sort3d-nav-setup)
  - [0) Cloning Repo and Recommended Installation Method](#0-cloning-repo-and-recommended-installation-method)
  - [1) Docker Installation (Recommended)](#1-docker-installation-recommended)
  - [2) Pulling and Preparing Docker Image](#2-pulling-and-preparing-docker-image)
  - [3a) Building ROS Humble System with Wheelchair Simulator](#3a-building-ros-humble-system-with-wheelchair-simulator)
  - [3b) Building ROS Noetic System with Wheelchair Simulator (Ubuntu 22.04)](#3b-building-ros-noetic-system-with-wheelchair-simulator-ubuntu-2204)
  - [3c) Building ROS Humble System with Mecanum Simulator](#3c-building-ros-humble-system-with-mecanum-simulator)
  - [(Optional) Installing ROS Humble System Dependencies Without Docker](#optional-installing-ros-humble-system-dependencies-without-docker)
  - [(Optional) Installing ROS Noetic System Dependencies Without Docker](#optional-installing-ros-noetic-system-dependencies-without-docker)
- [SORT3D-Nav: Usage](#sort3d-nav-usage)
  - [Simulation with Ground Truth Semantics](#simulation-with-ground-truth-semantics)
  - [Simulation with Semantic Mapping Module](#simulation-with-semantic-mapping-module)
  - [ROS Bag](#ros-bag)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Repository Structure

SORT3D has two major versions:

1. **SORT3D-Bench**: The version of SORT3D used to run the [ReferIt3D](https://referit3d.github.io) and the [IRef-VLA](https://github.com/HaochenZ11/IRef-VLA) benchmarks.
2. **SORT3D-Nav**: The version of SORT3D used to run navigation on our robot platforms, built on top of our base autonomy stack. SORT3D is deployed on two research platforms:
    1. [Our wheelchair-base robot (**wheelchair**)](https://github.com/jizhang-cmu/cmu_vla_challenge_unity), for which we have both **ROS Noetic** and **ROS Humble** versions.
    2. [Our mecanum-wheeled robot (**mecanum**)](https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform), for which we have a **ROS Humble** version.    
&nbsp;
<p align="center">
  <img src="media/mecanum_wheel.jpg" height="300" />
  <img src="media/wheelchair.jpg" height="300" />
</p>
&nbsp;
This repository contains a separate branch for each platform and each ROS version SORT3D-Nav is deployed on. The SORT3D-Bench script is included in the `humble-wheelchair` branch. Each version of SORT3D-Nav is accompanied with a unity-based simulator and a ROS bag recording of the office areas the live demonstrations were recorded in. Additionally, we provide launch scripts of SORT3D-Nav using both ground truth semantic segmentations and our live semantic mapping module. The table below summarizes the currently available systems and their respective branches:

| Platform | ROS Version | Branch | Simulation Available | Live Demo Available (Using ROS Bag) | Ground Truth Semantics Available | Semantic Mapping Module Available |
|---|---|---|---|---|---|---|
| Benchmark  | - | `humble-wheelchair` | ☑️ | - | ☑️ | - |
| Wheelchair | Noetic | `humble-wheelchair` | ☑️ | ☑️ | ☑️ | ☑️ |
| Wheelchair | Humble | `noetic-wheelchair` | ☑️ | ☑️ | ☑️ | ☑️ |
| Mecanum | Humble | `humble-mecanum` | ☑️ | ☑️ | ☑️ | ☑️ |

## Data

### Dataset For SORT3D-Bench

To run SORT3D-Bench, ensure the following three datasets are downloaded and unzipped:

1. **Object Captions Dataset**: For our benchmark, we have pregenerated 2D object crops and captions using our captioning system and [Qwen2.5-VL](#https://github.com/QwenLM/Qwen2.5-VL). To download, first install minio and tqdm:

    ```bash
    pip install minio tqdm
    ```

    Then run

    ```bash
    python data/download_crops_dataset.py --download_path data
    ```

    The data will be downloaded as a zip file in `data/`. Unzip the file directly into `data`, the path to the unzipped folder should be `data/captions`.

2. **IRef-VLA Scannet**: We use the processed pointclouds in [IRef-VLA](https://github.com/HaochenZ11/IRef-VLA) for our benchmark. Follow the [instructions in the repo](https://github.com/HaochenZ11/IRef-VLA/tree/main?tab=readme-ov-file#dataset-download) and download only the Scannet subset of the data:
    ```bash
    python download_dataset.py --download_path data/IRef-VLA --subset scannet
    ```

    Afterwards, unzip Scannet.zip into `data/IRef-VLA`. The folder structure should be `data/IRef-VLA/Scannet`.

3. **ReferIt3D**: We provide the subsets of [ReferIt3D](https://referit3d.github.io/) used for the benchmark in `data/referit3d`.

Extract the IRef-VLA and the captions data into the same folder. The final folder structure should look like so:

data/<br>
&nbsp;&nbsp;&nbsp;&nbsp;IRef-VLA/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scannet/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scene0000_00<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;instance_crops<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scene0000_00_free_space_pc_result.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scene0000_00_...<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scene0000_01<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;instance_crops<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scene0000_00_free_space_pc_result.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scene0000_00_...<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...<br>
&nbsp;&nbsp;&nbsp;&nbsp;referit3d/<br>


### ROS Bag Files for SORT3D-Nav

We provide ROS bag files for both the wheelchair and mecanum platforms. To download, install minio and tqdm:

```bash
pip install minio tqdm
```

Then run

```bash
python data/download_rosbag.py --download_path bagfiles --platform [wheelchair|mecanum]
```

while making sure to pick the correct platform. Each ROS bag will be downloaded as a zip file in `bagfiles/`. Unzip the bag files into your directory of choice before replaying them. The wheelchair bag file is currently available, with the mecanum-wheeled robot bag file upcoming with the release of the mecanum version of SORT3D-Nav.

## System Requirements

### Hardware Requirements

SORT3D-Nav has been deployed on an Nvidia RTX 4090 with 24GB of VRAM to run the live captioning model on the wheelchair, and on an Nvidia RTX 4090 with 16GB of VRAM to run the live captioning model on the mecanum-wheeled robot. The system requires a minimum of:

- 10GB of VRAM to run the semantic mapping module along with live captioning.
- 7GB of VRAM to run using ground truth semantics with live captioning.

If you have more VRAM, you may increase the `captioner_batch_size` in the run scripts to get faster captioning throughput (and vice versa). 

The language planner additionally requires a WiFi connection on the robot to connect to the Mistral servers. This system has been tested in Ubuntu 20.04, 22.04, and 24.04, running in the Ubuntu 22.04 Docker image we provide. 

## SORT3D-Bench: Setup

### 1.1) Conda Environment

**First, make sure you are checked out into `humble-wheelchair`**:
```
git checkout humble-wheelchair
```

We provide a conda environment containing all the the dependencies required for SORT3D-Bench, which does not require ROS. Create the conda environment like so.

```bash
conda env create -f environment.yml -n sort3d
```

A `requirements.txt` is also provided mirroring the pip requirements in the `environment.yml`. The Docker image contains all the requirements for SORT3D-Bench preinstalled as well. You may follow sections 1-2 in [Setup: SORT3D-Nav](#setup-sort3d-nav) to install Docker and set the image up.

### 1.2) Use Docker (Alternatively)

Build the docker: 

```bash
docker build --network=host -t sort3d:latest -f docker/Dockerfile_benchmark .
```

Run the docker: 

```bash
docker run --gpus all -it --rm -v [CODE_PATH]:/home/sort3d/SORT3D sort3d:latest
```

### 2) Dataset Setup

Follow the instructions in [Dataset For SORT3D-Bench](#dataset-for-sort3d-bench) to ensure the dataset is correctly set up.

## SORT3D-Bench: Usage

SORT3D uses [Mistral Large 2](https://mistral.ai/) by default. Create a free research API key, then set the environment variable `MISTRAL_API_KEY`:
```bash
export MISTRAL_API_KEY="YOUR API KEY HERE"
```
You may then run the benchmark on either Nr3D or Sr3D:
```bash
cd ai_module/src/language_planner/language_planner
conda activate sort3d # you can skip this if using a docker
python3 language_planner_benchmark.py --dataset [nr3d|sr3d] --log_dir [LOGFOLDER]
```
Choose `nr3d` or `sr3d` as the `--dataset` argument to run the benchmark on our subsets of Nr3D and Sr3D respectively. The benchmark results are logged in `ai_module/src/language_planner/language_planner/logs/exp###` by default (where ### starts at 000 and is automatically incremented with each run). The script logs all correct answers and LLM reasoning in `correct.json`, and all incorrect answers in `incorrect.json`.

The script takes a set of optional arguments. The fully supported ones for this release are tabulated below:
| Argument | Supported Values | Description |
|---|---|---|
|`--exp_name`| Any string | Give the current experiment an optional name. Default is exp###, where ### is an automatically assigned number. |
|`--model`| `mistral` - `gpt-4o` | Use a different LLM for grounding. Default is Mistral, and we have tested GPT-4o in our paper; other models included in our code may be buggy. For OpenAI, provide the API key in the `OPENAI_API_KEY` environment variable. |

## SORT3D-Nav: Setup

### 0) Cloning Repo and Recommended Installation Method

Begin by cloning the repo with its submodules in your home directory:
```bash
cd ~
git clone https://github.com/nzantout/SORT3D.git --recursive
```

We provide a CUDA-enabled Ubuntu 22.04 Docker image with both ROS Noetic (built from source) and ROS Humble preinstalled. **This is the recommended way to run SORT3D, as ROS and all dependencies are preinstalled in the docker image.** Follow sections 1 through 3 to install Docker on your computer, pull the image, and download simulation files. The user home directory, `/home/$USER`, is mounted as a volume in the Docker image, allowing access to the repo from the Docker image if the repo has been cloned within the home directory. We provide optional instructions to install the system on a base Ubuntu 22.04 system for both [ROS Humble](#optional-installing-ros-humble-system-dependencies-without-docker) and [ROS Noetic](#optional-installing-ros-noetic-system-dependencies-without-docker).

### 1) Docker Installation (Recommended)

Install Docker and grant user permission.
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
sudo usermod -aG docker ${USER}
```
Make sure to **restart the computer**, then install Nvidia Container Toolkit (Nvidia GPU Driver
should be installed already).

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```
sudo apt update && sudo apt install nvidia-container-toolkit
```
Configure Docker runtime and restart Docker daemon.
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Test if the installation is successful. You should see something like below.
```
docker run --gpus all --rm nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```
```
Sat Dec 16 17:27:17 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 24%   50C    P0    40W / 200W |    918MiB /  8192MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

### 2) Pulling and Preparing Docker Image

Allow remote X connections.
```
xhost +
```

Pull the Docker image and build the container:
```bash
cd docker
docker compose -f compose_gpu.yml up --build -d
```

To run without rebuilding:
```bash
docker compose -f compose_gpu.yml up -d
```

You may then access the running container.
```bash
docker exec -it ubuntu22_ros bash
```

### 3a) Building ROS Humble System with Wheelchair Simulator

**Make sure you are checked out into `humble-wheelchair`**:
```
git checkout humble-wheelchair
```

The instructions for building the base system are excerpted from [its original repo](https://github.com/jizhang-cmu/cmu_vla_challenge_unity/tree/foxy-humble). Start by making sure ROS Humble is sourced:

```bash
source /opt/ros/humble/setup.bash
```
Then build the base autonomy system in `simulator/wheelchair_unity`:
```bash
cd simulator/wheelchair_unity
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Download any of our [Unity environment models](https://drive.google.com/drive/folders/1ZDkAsXIBNCG6O6NGx81eW3HVU10oKVfZ?usp=sharing) **(the models are configured for ROS2, not compatible with ROS1)** and unzip the files to the 'src/vehicle_simulator/mesh/unity' folder. The environment model files should look like below. Note that the 'AssetList.csv' file is generated upon start of the system.

mesh/<br>
&nbsp;&nbsp;&nbsp;&nbsp;unity/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;environment/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model_Data/ (multiple files in the folder)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model.x86_64<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UnityPlayer.so<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AssetList.csv (generated at runtime)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dimensions.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Categories.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object_list.txt<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;traversable_area.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;render.jpg<br>

Build SORT3D-Nav in `ai_module`:

```bash
cd ../../ai_module
colcon build --symlink-install
```

Afterwards, install the following dependencies in the [semantic mapping module](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar). If you are using our provided Docker image, all the other dependencies in the repositories are preinstalled, and you only need to install these. Otherwise, if you want to use the module outside the image, follow the instructions in the [repo README](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar?tab=readme-ov-file#repository-setup).

```bash
cd ../semantic_mapper/external
pip install Grounded-SAM-2/grounding_dino
pip install Grounded-SAM-2
pip install byte_track cython_bbox
```

### 3b) Building ROS Noetic System with Wheelchair Simulator (Ubuntu 22.04)

**Make sure you are checked out into `noetic-wheelchair`**:
```bash
git checkout noetic-wheelchair
```

The instructions for building the base system are excerpted from [its original repo](https://github.com/jizhang-cmu/cmu_vla_challenge_unity). Since SORT3D requires Python > 3.9 to work, ROS Noetic cannot be used on its default 20.04, and must be built from source on Ubuntu 22.04. Instructions to build ROS Noetic on Ubuntu 22.04 from source are in [this section](#optional-building-ros-noetic-system-in-base-ubuntu-2204), and ROS Noetic is already prebuilt in the provided Docker image. The base autonomy system requires extra ROS dependencies which we have modified to compile on Ubuntu 22.04, found in `simulator/noetic_ubuntu22_extra_deps`. These dependencies must be built first, then the [workspace overlaid](https://wiki.ros.org/catkin/Tutorials/workspace_overlaying) by sourcing it before building the simulator workspace:

```bash
source /opt/ros/noetic/setup.bash
cd simulator/noetic_ubuntu22_extra_deps
catkin_make
source devel/setup.bash
cd ../wheelchair_unity
catkin_make
```

Download any of our [Unity environment models](https://drive.google.com/drive/folders/1bmxdT6Oxzt0_0tohye2br7gqTnkMaq20?usp=share_link) **(the models are configured for ROS1, not compatible with ROS2)** and unzip the files to the 'src/vehicle_simulator/mesh/unity' folder. The environment model files should look like below. Note that the 'AssetList.csv' file is generated upon start of the system.

mesh/<br>
&nbsp;&nbsp;&nbsp;&nbsp;unity/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;environment/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model_Data/ (multiple files in the folder)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model.x86_64<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UnityPlayer.so<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AssetList.csv (generated at runtime)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dimensions.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Categories.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object_list.txt<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;traversable_area.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;render.jpg<br>

Build SORT3D-Nav in `ai_module`:
```bash
cd ../../ai_module
catkin_make
```

Afterwards, install the following dependencies in the [semantic mapping module](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar/tree/ros1?tab=readme-ov-file). If you are using our provided Docker image, all the other dependencies in the repositories are preinstalled, and you only need to install these. Otherwise, if you want to use the module outside the image, follow the instructions in the [repo README](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar/tree/ros1?tab=readme-ov-file#repository-setup).

```bash
cd ../semantic_mapper/external
pip install Grounded-SAM-2/grounding_dino
pip install Grounded-SAM-2
pip install byte_track cython_bbox
```

### 3c) Building ROS Humble System with Mecanum Simulator

**Make sure you are checked out into `humble-mecanum`**:
```
git checkout humble-mecanum
```

The instructions for building the base system are excerpted from [its original repo](https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform/tree/humble). Start by making sure ROS Humble is sourced:

```bash
source /opt/ros/humble/setup.bash
```

Then build the base autonomy system in `simulator/mecanum_unity`, skipping the SLAM module and Mid-360 lidar driver (the two packages are not needed for simulation):
```bash
cd simulator/mecanum_unity
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2
```
Download a [Unity environment model for the Mecanum wheel platform](https://drive.google.com/drive/folders/1G1JYkccvoSlxyySuTlPfvmrWoJUO8oSs?usp=sharing) and unzip the files to the 'src/base_autonomy/vehicle_simulator/mesh/unity' folder. The environment model files should look like below.

mesh/<br>
&nbsp;&nbsp;&nbsp;&nbsp;unity/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;environment/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model_Data/ (multiple files in the folder)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model.x86_64<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;UnityPlayer.so<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AssetList.csv (generated at runtime)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dimensions.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Categories.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;object_list.txt<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;traversable_area.ply<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;map.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;render.jpg<br>

Build SORT3D-Nav in `ai_module`:

```bash
cd ../../ai_module
colcon build --symlink-install
```

Afterwards, install the following dependencies in the [semantic mapping module](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar). If you are using our provided Docker image, all the other dependencies in the repositories are preinstalled, and you only need to install these. Otherwise, if you want to use the module outside the image, follow the instructions in the [repo README](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar?tab=readme-ov-file#repository-setup).

```bash
cd ../semantic_mapper/external
pip install Grounded-SAM-2/grounding_dino
pip install Grounded-SAM-2
pip install byte_track cython_bbox
```

### (Optional) Installing ROS Humble System Dependencies without Docker

This section contains instructions to install ROS Humble and SORT3D-Nav system dependencies on a base Ubuntu 22.04 system. Please report any issues to the issue tracker.

1. Begin by installing ros-humble-desktop, following the [ROS wiki page](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).
2. Install CUDA Toolkit 12.x following [the instructions on the official website](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu). This system has been tested with CUDA 12.1, but should work with higher CUDA versions.
3. Install ROS Humble dependencies for the base autonomy system:
    ```bash
    sudo apt update
    sudo apt install libusb-dev ros-humble-perception-pcl ros-humble-sensor-msgs-py ros-humble-tf-transformations ros-humble-joy python3-colcon-common-extensions python-is-python3 
    pip install transforms3d pyyaml
    ```
4. Install the pip dependencies for SORT3D-Nav. Make sure you are in this repo's top level directory:
    ```bash
    pip install -r requirements.txt
    ```
5. Follow [Section 3a](#3a-building-ros-humble-system-with-wheelchair-simulator) or [Section 3c](#3c-building-ros-humble-system-with-mecanum-simulator) to set up the system.


### (Optional) Installing ROS Noetic System Dependencies without Docker

This section contains instructions to build ROS Noetic from source and SORT3D-Nav system dependencies on a base Ubuntu 22.04 system. Please report any issues to the issue tracker.

1. As ROS Noetic does not support Ubuntu 22.04, it must be built from source. Follow the instructions in [this Reddit post](https://www.reddit.com/r/ROS/comments/158icpy/compiling_ros1_noetic_from_source_on_ubuntu_2204/), mirrored in [this repository](https://github.com/nzantout/ros-noetic-ubuntu-2204-compile-instructions).
2. Install CUDA Toolkit 12.x following [the instructions on the official website](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu). This system has been tested with CUDA 12.1, but should work with higher CUDA versions.
3. Install ROS Noetic dependencies for the base autonomy system:
    ```
    sudo apt update
    sudo apt install libusb-dev python-yaml python-is-python3
    ```
4. Install the pip dependencies for SORT3D-Nav. Make sure you are in this repo's top level directory:
    ```bash
    pip install -r requirements.txt
    ```
5. Follow [Section 3b](#3b-building-ros-noetic-system-with-wheelchair-simulator-ubuntu-2204) to set up the system.


## SORT3D-Nav: Usage

### Simulation with Ground Truth Semantics

**The instructions for running the simulated system using ground truth semantics are the same regardless of which branch you are using. Check out the branch you wish to run.**

SORT3D uses [Mistral Large 2](https://mistral.ai/) by default. Create a free research API key, then replace the placeholder in [`scripts/run_full_system_gt_semantics.sh`](scripts/run_full_system_gt_semantics.sh) with your API key:
```
export MISTRAL_API_KEY="YOUR API KEY HERE"
```
You may do the same with [`scripts/run_sort3d_navigation_gt_semantics.sh`](scripts/run_sort3d_navigation_gt_semantics.sh) if you want to run SORT3D separately from the base autonomy system. Make sure all the scripts are executable:

```
chmod -R +x scripts 
```

Then, in one terminal, run
```
scripts/run_full_system_gt_semantics.sh
```

Wait until the system starts up. You should see the RViz and Unity windows open:

<img src="media/rviz.png" alt="RViz window" width="63%"> <img src="media/unity_window.png" alt="Unity window" width="36%">

In your terminal, the captioning and language planner nodes will be logging to standard output:

<img src="media/captioning_terminal.png" alt="" class="">

In another terminal, run the query publisher node to take in from standard input:

```bash
scripts/run_query_publisher.sh
```

The output of the query publisher node should look like so:

```
[INFO] [1743652602.832061201] [language_publisher]: LanguagePublisher node has been started. Type your query below.
Enter a query to publish: 
```

You may then type a natural language navigation statement, like "go near the red chair", and watch the system navigate:

https://github.com/user-attachments/assets/a4b88e80-4ae6-4ce1-9215-1cd449275b7f

### Simulation with Semantic Mapping Module

**The instructions for running the simulated system with the semantic mapping module are the same regardless of which branch you are using. Check out the branch you wish to run.**

Create a free research API key for [Mistral Large 2](https://mistral.ai/), then replace the placeholder in [`scripts/run_full_system_semantic_mapping.sh`](scripts/run_full_system_semantic_mapping.sh) with your API key:
```bash
export MISTRAL_API_KEY="YOUR API KEY HERE"
```
You may do the same with [`scripts/run_sort3d_navigation_semantic_mapping.sh`](scripts/run_sort3d_navigation_semantic_mapping.sh) if you want to run SORT3D separately from the base autonomy system. Make sure all the scripts are executable:

```bash
chmod -R +x scripts 
```

Then, in one terminal, run
```bash
scripts/run_full_system_semantic_mapping.sh
```

Wait until the system starts up. You should see the RViz and Unity windows open:

<img src="media/rviz_semantic_mapping.png" alt="RViz window" width="62%"> <img src="media/unity_window_semantic_mapping.png" alt="Unity window" width="37%">

In your terminal, the semantic mapping and language planner nodes will be logging to standard output:

<img src="media/semantic_mapping_terminal.png" alt="">

In another terminal, run the query publisher node to take in from standard input:

```bash
scripts/run_query_publisher.sh
```

The output of the query publisher node should look like so:

```
[INFO] [1743652602.832061201] [language_publisher]: LanguagePublisher node has been started. Type your query below.
Enter a query to publish: 
```

Afterwards, drive the robot around to create a semantic map of the scene scene either using the virtual joystick or by clicking the "Waypoint with Heading" button and supplying waypoints:

https://github.com/user-attachments/assets/8d05dea9-5365-4ea1-9bf8-257a86b33791

You may then type a natural language navigation statement, like "go to the potted plant furthest from you", and watch the system navigate:

https://github.com/user-attachments/assets/2c5efd27-dc2d-46fa-bec8-b674cfed157d

### ROS Bag

We provide ROS bags of various indoor environments to demonstrate SORT3D-Nav in real environments. [Follow the instructions above to download](#ros-bag-files-for-sort3d-nav) a ROS bag for either the mecanum-wheeled robot or the wheelchair-base robot. Again, make sure you have created a free research API key for [Mistral Large 2](https://mistral.ai/), then replace the placeholder in [`scripts/run_sort3d_navigation_semantic_mapping.sh`](scripts/run_sort3d_navigation_semantic_mapping.sh) with your API key:
```
export MISTRAL_API_KEY="YOUR API KEY HERE"
```

Start by running the script for SORT3D-Nav using semantic mapping (script is the same regardless of which branch you are using):
```bash
scripts/run_sort3d_navigation_semantic_mapping.sh
```

Run the Rviz viewer in a second terminal:
```bash
scripts/run_rviz_viewer.sh
```

In a third terminal, play the ROS bag you downloaded. **If you are using ROS 1:**
```bash
rosbag play [ros_bag].bag
```

**If you are using ROS 2:**
```bash
ros2 bag play [ros_bag].db3
```

Follow the instructions on screen to pause/unpause the bag file. Run the bag for a while to generate a map, and you can see it being generated in the Rviz screen:

https://github.com/user-attachments/assets/1781e73e-6250-434f-8bfb-0919f33842c8

To see the target bounding boxes for a query being generated, you may pause the ROS bag, then run the query publisher in a fourth terminal and provide a query:

```bash
scripts/run_query_publisher.sh
```


## Troubleshooting

Please report any issues you face in the issue tracker, and we'll add them here.

## Citation

If you use our work, please cite:

```
@misc{zantout2025sort3dspatialobjectcentricreasoning,
      title={SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using Large Language Models}, 
      author={Nader Zantout and Haochen Zhang and Pujith Kachana and Jinkai Qiu and Ji Zhang and Wenshan Wang},
      year={2025},
      eprint={2504.18684},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.18684}, 
}
```
