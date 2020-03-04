#!/usr/bin/env bash
set -ex

pushd_fail() {
  echo "#######################################################################"
  echo "##                          PUSHD FAILED                             ##"
  echo "#######################################################################"
  exit 127
}

popd_fail() {
  echo "#######################################################################"
  echo "##                           POPD FAILED                             ##"
  echo "#######################################################################"
  exit 127
}

bail_out_early() {
  echo "#######################################################################"
  echo "##                   STOPPING HERE EARLY                             ##"
  echo "#######################################################################"
  exit 255
}

check_your_blaster() {
  sudo update-alternatives --display libblas.so-aarch64-linux-gnu
  sudo update-alternatives --display libblas.so.3-aarch64-linux-gnu
  sudo update-alternatives --display liblapack.so-aarch64-linux-gnu
  sudo update-alternatives --display liblapack.so.3-aarch64-linux-gnu
}

pick_your_blaster() {
  if [ "${1}" == "atlas" ]; then
    sudo update-alternatives --set libblas.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/libblas.so
    sudo update-alternatives --set libblas.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/libblas.so.3
    sudo update-alternatives --set liblapack.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/liblapack.so
    sudo update-alternatives --set liblapack.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/atlas/liblapack.so.3
  fi

  if [ "${1}" == "blas" ]; then
    sudo update-alternatives --set libblas.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/blas/libblas.so
    sudo update-alternatives --set libblas.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/blas/libblas.so.3
    sudo update-alternatives --set liblapack.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/lapack/liblapack.so
    sudo update-alternatives --set liblapack.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/lapack/liblapack.so.3
  fi

  if [ "${1}" == "openblas" ]; then
    sudo update-alternatives --set libblas.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/libblas.so
    sudo update-alternatives --set libblas.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/libblas.so.3
    sudo update-alternatives --set liblapack.so-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/liblapack.so
    sudo update-alternatives --set liblapack.so.3-aarch64-linux-gnu /usr/lib/aarch64-linux-gnu/openblas/liblapack.so.3
  fi
}

NPROC=$(nproc)
IP_ADDR=$(ip addr show eth0 | grep -Eo '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -1)
HOSTNAME=$(hostname -s)
APT_CMD="sudo apt --quiet --assume-yes"

INSTALL_ROOT="/opt"
INSTALL_SRC="${INSTALL_ROOT}/src"
CATKIN_WS="${INSTALL_ROOT}/catkin_ws"
CATKIN_SRC="${CATKIN_WS}/src"
sudo mkdir -p "${INSTALL_SRC}"
sudo chown -R "${USER}:${USER}" "${INSTALL_SRC}"
sudo mkdir -p "${CATKIN_SRC}"
sudo chown -R "${USER}:${USER}" "${CATKIN_WS}"
pushd ${INSTALL_ROOT} || pushd_fail

echo "#########################################################################"
echo " install opencv and as many apt/ python installs as possible            #"
echo "#########################################################################"
${APT_CMD} purge libopencv || true

${APT_CMD} remove update-manager || true

${APT_CMD} install \
  build-essential \
  cmake \
  curl \
  g++ \
  gcc \
  gfortran \
  git-core \
  gnuplot \
  gnuplot-x11 \
  imagemagick \
  ipython \
  libatlas-base-dev \
  libatlas3-base \
  libblas-dev \
  libblas3 \
  libboost-all-dev \
  libedit-dev \
  libfftw3-dev \
  libflann-dev \
  libgflags-dev \
  libgl1-mesa-dri \
  libgoogle-glog-dev \
  libgraphicsmagick1-dev \
  libhdf5-serial-dev \
  libjpeg-dev \
  libjpeg-turbo8-dev \
  liblapack-dev \
  liblapack3 \
  libleveldb-dev \
  liblmdb-dev \
  libopenblas-base \
  libopenni2-dev \
  libpcl-dev \
  libpng-dev \
  libprotobuf-dev \
  libqt4-dev \
  libreadline-dev \
  libsnappy-dev \
  libsox-dev \
  libsox-fmt-all \
  libsuitesparse-dev \
  libtinfo-dev \
  libturbojpeg0-dev \
  libusb-1.0-0-dev \
  libxml2-dev \
  libzmq3-dev \
  ncurses-dev \
  ninja-build \
  ntp \
  pkg-config \
  protobuf-compiler \
  python-pip \
  python-qt4 \
  python3-pip \
  software-properties-common \
  sox \
  unzip \
  zlib1g-dev

${APT_CMD} purge libeigen3-dev || true

echo "#########################################################################"
echo "# install eigen to most updated version                                 #"
echo "#########################################################################"
# https://github.com/eigenteam/eigen-git-mirror/blob/master/INSTALL
mkdir -p ${INSTALL_SRC}/eigen
pushd ${INSTALL_SRC}/eigen || pushd_fail
git clone https://github.com/eigenteam/eigen-git-mirror .
mkdir -p build
pushd build || pushd_fail
cmake ..
sudo make install -j"${NPROC}"
popd || popd_fail
popd || popd_fail

pip install requests numpy pyyaml
pip3 install cython  # this is by itself on purpose (numpy needs it and I
                     # dont trust pip to install them in the right order)
pip3 install numpy decorator attrs pyyaml

echo "#########################################################################"
echo "# install librealsense2                                                 #"
echo "#########################################################################"
# https://github.com/dorodnic/librealsense/blob/jetson_doc/doc/installation_jetson.md
# https://github.com/IntelRealSense/librealsense/issues/4969
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
${APT_CMD} install librealsense2-utils librealsense2-dev

echo "#########################################################################"
echo "# calibration toolkit                                                   #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/calibration_toolkit
pushd ${CATKIN_SRC}/calibration_toolkit || pushd_fail
git clone --branch v0.2 --single-branch https://github.com/iaslab-unipd/calibration_toolkit .
popd || popd_fail

echo "#########################################################################"
echo "# ros realsense                                                         #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/realsense
pushd ${CATKIN_SRC}/realsense || pushd_fail
git clone --branch 2.2.12 --single-branch https://github.com/intel-ros/realsense.git .
popd || popd_fail

echo "#########################################################################"
echo "# install iai_kinect2                                                   #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/iai_kinect2
pushd ${CATKIN_SRC}/iai_kinect2 || pushd_fail
git clone https://github.com/code-iai/iai_kinect2.git .
popd || popd_fail

echo "#########################################################################"
echo "# install freenect2                                                     #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/libfreenect2
pushd ${INSTALL_SRC}/libfreenect2 || pushd_fail
git clone https://github.com/OpenKinect/libfreenect2.git .
# mesa links libGL.so incorrectly on arm v8, properly link libGL
sudo rm -f /usr/lib/aarch64-linux-gnu/libGL.so
sudo ln -sf /usr/lib/aarch64-linux-gnu/libGL.so.1.0.0 /usr/lib/aarch64-linux-gnu/libGL.so
mkdir -p build
pushd build || pushd_fail
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/freenect2
make -j"${NPROC}"
sudo make install -j"${NPROC}"
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# install opencv to most updated version                                #"
echo "#########################################################################"
# https://github.com/eigenteam/eigen-git-mirror/blob/master/INSTALL
mkdir -p ${INSTALL_SRC}/opencv
mkdir -p ${INSTALL_SRC}/opencv_contrib
export OPENCV_CONTRIB_PATH="${INSTALL_SRC}/opencv_contrib"
git clone https://github.com/opencv/opencv.git ${INSTALL_SRC}/opencv
git clone https://github.com/opencv/opencv_contrib.git ${OPENCV_CONTRIB_PATH}
pushd ${INSTALL_SRC}/opencv || pushd_fail
mkdir -p build
pushd build || pushd_fail
# not sure if this will work, but we'll see
cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D INSTALL_C_EXAMPLES=OFF \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D BUILD_EXAMPLES=ON \
  -D PYTHON3_EXECUTABLE="$(command -v python3)" \
  -D PYTHON_INCLUDE_DIR="$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")" \
  -D PYTHON_INCLUDE_DIR2="$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))")" \
  -D PYTHON_LIBRARY="$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))")" \
  -D PYTHON3_NUMPY_INCLUDE_DIRS="$(python3 -c "import numpy; print(numpy.get_include())")" \
  -D PYTHON3_PACKAGES_PATH="$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")" \
  -D PYTHON2_EXECUTABLE="$(command -v python2)" \
  -D PYTHON2_INCLUDE_DIR="$(python2 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")" \
  -D PYTHON2_INCLUDE_DIR2="$(python2 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))")" \
  -D PYTHON_LIBRARY="$(python2 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))")" \
  -D PYTHON2_NUMPY_INCLUDE_DIRS="$(python2 -c "import numpy; print(numpy.get_include())")" \
  -D PYTHON2_PACKAGES_PATH="$(python2 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")" \
  -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_PATH}/modules ..
make -j"${NPROC}"
# sudo make install -j"${NPROC}"
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# install ceres_solver                                                  #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/ceres
pushd ${INSTALL_SRC}/ceres || pushd_fail
git clone https://github.com/ceres-solver/ceres-solver.git .
mkdir -p build
pushd build || pushd_fail
cmake .. -DEIGEN_INCLUDE_DIR=/usr/local/include/eigen3
make -j"${NPROC}"
sudo make install -j"${NPROC}"
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# install TVM and TVM deps                                              #"
echo "#########################################################################"
# https://docs.tvm.ai/install/from_source.html
mkdir -p ${INSTALL_SRC}/tvm
pushd ${INSTALL_SRC}/tvm || pushd_fail
git clone --recursive https://github.com/apache/incubator-tvm .
mkdir -p build
cp cmake/config.cmake build/config.cmake
pushd build || pushd_fail
cmake -GNinja \
 -DUSE_CUDA=ON \
 -DUSE_CUDNN=ON \
 -DUSE_CUBLAS=ON \
 -DUSE_SORT=ON ..
ninja -v
# install the python libraries -- it only works in py3
# and we won't use tvm to ever compile device side, only run side
pushd ${INSTALL_SRC}/tvm/python || pushd_fail
python3 setup.py install --user
popd || popd_fail
pushd ${INSTALL_SRC}/tvm/topi/python
python3 setup.py install --user
popd || popd_fail
popd || popd_fail

#echo "#########################################################################"
#echo "# install pytorch and pytorch deps                                      #"
#echo "#########################################################################"
#mkdir -p ${INSTALL_SRC}/pytorch
#pushd ${INSTALL_SRC}/pytorch || pushd_fail
# https://github.com/pytorch/pytorch#from-source
# https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/
#curl -L https://nvidia.box.com/shared/static/1v2cc4ro6zvsbu0p8h6qcuaqco1qcsif.whl -o torch-1.4.0-cp27-cp27mu-linux_aarch64.whl
#pip install torch-1.4.0-cp27-cp27mu-linux_aarch64.whl

#curl -L https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -o torch-1.4.0-cp36-cp36m-linux_aarch64.whl
#pip3 install torch-1.4.0-cp36-cp36m-linux_aarch64.whl

echo "#########################################################################"
echo "# install pytorch and pytorch deps                                      #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/pytorch
pushd ${INSTALL_SRC}/pytorch || pushd_fail
git clone --recursive https://github.com/pytorch/pytorch .
git submodule sync
git submodule update --init --recursive
# python3 setup.py install
python3 setup.py install
# i think we have to copy the directories to use
popd || popd_fail



echo "#########################################################################"
echo "# openptrack deps and clone necessary deps                               #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/open_ptrack
pushd ${CATKIN_SRC}/open_ptrack || pushd_fail
git clone https://github.com/ammolitor/open_ptrack_v2 .
curl -kL https://pjreddie.com/media/files/yolo.weights -o ${CATKIN_SRC}/open_ptrack/yolo_detector/darknet_opt/coco.weights
sudo cp -r /usr/local/include/eigen3 /usr/include/eigen3
popd || popd_fail

echo "#########################################################################"
echo "# install ROS using jetsonhacks XAVIER as a guide                       #"
echo "#########################################################################"
# https://github.com/jetsonhacks/installROS/blob/master/installROS.sh
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
${APT_CMD} update
${APT_CMD} install ros-melodic-desktop-full
${APT_CMD} install python-rosdep \
  ros-melodic-compressed-depth-image-transport \
  ros-melodic-compressed-image-transport \
  ros-melodic-cv-bridge \
  ros-melodic-rgbd-launch \
  ros-melodic-rqt-common-plugins \
  ros-melodic-rviz
sudo rosdep init
rosdep update
grep -q -F 'source /opt/ros/melodic/setup.bash' "${HOME}"/.bashrc || \
  echo "source /opt/ros/melodic/setup.bash" >> "${HOME}"/.bashrc
# shellcheck source=.bashrc
. "${HOME}"/.bashrc
${APT_CMD} install \
  python-rosinstall \
  python-rosinstall-generator \
  python-wstool

echo "#########################################################################"
echo "# update deps for rosdep                                                #"
echo "#########################################################################"
${APT_CMD} install ros-melodic-rqt-common-plugins \
  ros-melodic-camera-calibration \
  libcanberra-gtk-module



pushd ${CATKIN_WS} || pushd_fail
. /opt/ros/melodic/setup.bash
rosdep install -y -r --from-paths .
export ROS_PARALLEL_JOBS="-j1 -l1"
catkin_make
popd || popd_fail

echo "#########################################################################"
echo " # THE FOLLOWING IS RUNTIME CONFIG                                      #"
echo "#########################################################################"
echo "export ROS_MASTER_URI=http://${IP_ADDR}:11311/
export ROS_IP=${IP_ADDR}
export ROS_PC_NAME=${HOSTNAME}" >> ~/.bashrc

