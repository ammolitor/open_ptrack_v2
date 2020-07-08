#!/usr/bin/env bash
set -ex

sudo /bin/sed -i 's/%sudo	ALL=(ALL:ALL) ALL/%sudo	ALL=(ALL:ALL) NOPASSWD:ALL/' /etc/sudoers

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
  sudo update-alternatives --display libblas.so-x86_64-linux-gnu
  sudo update-alternatives --display libblas.so.3-x86_64-linux-gnu
  sudo update-alternatives --display liblapack.so-x86_64-linux-gnu
  sudo update-alternatives --display liblapack.so.3-x86_64-linux-gnu
}

pick_your_blaster() {
  if [ "${1}" == "atlas" ]; then
    sudo update-alternatives --set libblas.so-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/atlas/libblas.so
    sudo update-alternatives --set libblas.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/atlas/libblas.so.3
    sudo update-alternatives --set liblapack.so-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/atlas/liblapack.so
    sudo update-alternatives --set liblapack.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/atlas/liblapack.so.3
  fi

  if [ "${1}" == "blas" ]; then
    sudo update-alternatives --set libblas.so-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/blas/libblas.so
    sudo update-alternatives --set libblas.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/blas/libblas.so.3
    sudo update-alternatives --set liblapack.so-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/lapack/liblapack.so
    sudo update-alternatives --set liblapack.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3
  fi

  if [ "${1}" == "openblas" ]; then
    sudo update-alternatives --set libblas.so-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/openblas/libblas.so
    sudo update-alternatives --set libblas.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/openblas/libblas.so.3
    sudo update-alternatives --set liblapack.so-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/openblas/liblapack.so
    sudo update-alternatives --set liblapack.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/openblas/liblapack.so.3
  fi

  if [ "${1}" == "auto" ]; then
  sudo update-alternatives --auto libblas.so-x86_64-linux-gnu
  sudo update-alternatives --auto libblas.so.3-x86_64-linux-gnu
  sudo update-alternatives --auto liblapack.so-x86_64-linux-gnu
  sudo update-alternatives --auto liblapack.so.3-x86_64-linux-gnu
  fi
}

NPROC=$(nproc)
IP_ADDR=$(ip addr show | awk '/inet / && !/127/{gsub("/..", "", $2); print $2}')
HOSTNAME=$(hostname -s)
APT_CMD="sudo apt-get --quiet --assume-yes"

INSTALL_ROOT="/opt"
INSTALL_SRC="${INSTALL_ROOT}/src"
CATKIN_WS="${INSTALL_ROOT}/catkin_ws"
CATKIN_SRC="${CATKIN_WS}/src"
sudo mkdir -p "${INSTALL_SRC}"
sudo chown -R "${USER}:${USER}" "${INSTALL_SRC}"
sudo mkdir -p "${CATKIN_SRC}"
sudo chown -R "${USER}:${USER}" "${CATKIN_WS}"
pushd ${INSTALL_ROOT} || pushd_fail

################################################################################
#  nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04
# https://gitlab.com/nvidia/container-images/opengl/-/tree/ubuntu18.04
################################################################################
sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    pkg-config \
    libxau6           libxau6:i386 \
    libxdmcp6         libxdmcp6:i386 \
    libxcb1           libxcb1:i386 \
    libxext6          libxext6:i386 \
    libx11-6          libx11-6:i386 \
    libglvnd0         libglvnd0:i386 \
    libgl1            libgl1:i386 \
    libglx0           libglx0:i386 \
    libegl1           libegl1:i386 \
    libgles2          libgles2:i386 \
    libglvnd-dev      libglvnd-dev:i386 \
    libgl1-mesa-dev   libgl1-mesa-dev:i386 \
    libegl1-mesa-dev  libegl1-mesa-dev:i386 \
    libgles2-mesa-dev libgles2-mesa-dev:i386

sudo tee -a /etc/ld.so.conf.d/nvidia.conf > /dev/null << EOF
/usr/local/nvidia/lib
/usr/local/nvidia/lib64
EOF

sudo tee -a /etc/environment > /dev/null << 'EOF'
LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
EOF

sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json > /dev/null << EOF
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

echo "#########################################################################"
echo " remove bloat                                                           #"
echo "#########################################################################"
sudo sed -i 's/NoDisplay=true/NoDisplay=false/g' /etc/xdg/autostart/*.desktop

sudo systemctl disable cupsd || true
sudo systemctl disable cups-browsed || true
sudo systemctl disable avahi-daemon || true

sudo chmod -x /usr/lib/x86_64-linux-gnu/hud/hud-service || true
sudo mv /usr/lib/evolution-data-server /usr/lib/evolution-data-server-disabled || true
sudo mv /usr/lib/evolution /usr/lib/evolution-disabled || true

${APT_CMD} purge gnome-software || true
${APT_CMD} purge libreoffice\* || true
${APT_CMD} purge update-manager unattended-upgrades || true

${APT_CMD} purge snapd || true
rm -rf snap

${APT_CMD} autoremove || true
${APT_CMD} clean || true

echo "#########################################################################"
echo " increase usb fs buffer for multiple cameras on a single host           #"
echo "#########################################################################"
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=256"/' /etc/default/grub
sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb > /dev/null << EOF
256
EOF
sudo update-grub

echo "#########################################################################"
echo " setup ntp                                                              #"
echo "#########################################################################"
${APT_CMD} install ntp
sudo tee /etc/ntp.conf > /dev/null << EOF
# /etc/ntp.conf, configuration for ntpd; see ntp.conf(5) for help
driftfile /var/lib/ntp/ntp.drift
# Enable this if you want statistics to be logged.
#statsdir /var/log/ntpstats/
statistics loopstats peerstats clockstats
filegen loopstats file loopstats type day enable
filegen peerstats file peerstats type day enable
filegen clockstats file clockstats type day enable
# Specify one or more NTP servers.
server 192.168.1.200 iburst # DOUBLE CHECK THIS IP ADDRESS
disable auth
broadcastclient
EOF

echo "#########################################################################"
echo " remove and install packages                                            #"
echo "#########################################################################"
# https://github.com/OpenPTrack/open_ptrack_v2/blob/1804/docker/open_ptrack-dep/Dockerfile
# https://github.com/OpenPTrack/open_ptrack_v2/blob/1804/docker/open_ptrack/Dockerfile
${APT_CMD} purge libopencv* || true
${APT_CMD} purge opencv* || true
${APT_CMD} autoremove || true
${APT_CMD} update
${APT_CMD} upgrade

${APT_CMD} install \
  ocl-icd-libopencl1 \
  ocl-icd-opencl-dev \

sudo mkdir -p /etc/OpenCL/vendors
sudo tee /etc/OpenCL/vendors/nvidia.icd > /dev/null << EOF
libnvidia-opencl.so.1
EOF

${APT_CMD} install \
  apt-transport-https \
  apt-utils \
  automake \
  build-essential \
  clang \
  clinfo \
  curl \
  doxygen \
  g++ \
  g++-multilib \
  gcc \
  gcc-multilib \
  gfortran \
  git \
  git-core \
  gnuplot \
  gnuplot-x11 \
  imagemagick \
  ipython \
  libboost-all-dev \
  libedit-dev \
  libfftw3-dev \
  libflann-dev \
  libgflags-dev \
  libgl1-mesa-dev \
  libgl1-mesa-dri \
  libglewmx-dev \
  libglfw3-dev \
  libglu1-mesa-dev \
  libgoogle-glog-dev \
  libgraphicsmagick1-dev \
  libgtk2.0-dev \
  libhdf5-serial-dev \
  libjpeg-dev \
  libjpeg-turbo8-dev \
  liblapack-dev \
  liblapack3 \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libopenni2-dev \
  libpcl-dev \
  libpng-dev \
  libprotobuf-dev \
  libqt4-dev \
  libreadline-dev \
  libsnappy-dev \
  libsoundio-dev \
  libsox-dev \
  libsox-fmt-all \
  libssl-dev \
  libsuitesparse-dev \
  libtinfo-dev \
  libtool \
  libturbojpeg \
  libturbojpeg0-dev \
  libudev-dev \
  libusb-1.0-0-dev \
  libvulkan-dev \
  libx11-dev \
  libxcursor-dev \
  libxi-dev \
  libxinerama-dev \
  libxml2-dev \
  libxmu-dev \
  libxrandr-dev \
  libzmq3-dev \
  lsb-release \
  mesa-common-dev \
  mesa-utils \
  nasm \
  ncurses-dev \
  net-tools \
  ninja-build \
  nlohmann-json-dev \
  ntp \
  pkg-config \
  protobuf-compiler \
  python-pip \
  python-qt4 \
  python3.7 \
  python3.7-dev \
  python3-testresources \
  software-properties-common \
  sox \
  ssh \
  tig \
  tmux \
  tree \
  udev \
  unzip \
  uuid-dev \
  vim \
  vino \
  wget \
  xorg-dev \
  zlib1g-dev

${APT_CMD} install \
  libblas3 \
  libblas-dev \
  libopenblas-base \
  libopenblas-dev \
  libatlas3-base \
  libatlas-base-dev

${APT_CMD} purge libeigen3-dev || true
sudo curl -o /usr/include/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
pick_your_blaster "openblas"

echo "#########################################################################"
echo "# install cmake                                                         #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/cmake
pushd ${INSTALL_SRC}/cmake || pushd_fail
curl -kL -O https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz
tar --strip-components=1 -xf cmake-3.17.3.tar.gz --directory ${INSTALL_SRC}/cmake && rm cmake-3.17.3.tar.gz
./bootstrap
make
sudo make install
popd || popd_fail

echo "#########################################################################"
echo "# install cuda 10                                                       #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/nvidia
pushd ${INSTALL_SRC}/nvidia || pushd_fail
curl -kL https://food-safety-infra.s3-us-west-2.amazonaws.com/openptrack/nvidia/NVIDIA-Linux-x86_64-440.64.run -o nvidia.run
chmod +x nvidia.run
sudo ./nvidia.run --silent --no-questions
curl -kL https://food-safety-infra.s3-us-west-2.amazonaws.com/openptrack/nvidia/cuda_10.2.89_440.33.01_linux.run -o cuda.run
chmod +x cuda.run
sudo ./cuda.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.2 --samples --samplespath=${INSTALL_SRC}/nvidia/samples
curl -kL https://food-safety-infra.s3-us-west-2.amazonaws.com/openptrack/nvidia/cudnn-10.2-linux-x64-v7.6.5.32.tgz -o cudnn-10.2.tgz
tar -xf cudnn-10.2.tgz
sudo cp -av cuda/include/* /usr/local/cuda/include
sudo cp -av cuda/lib64/* /usr/local/cuda/lib64/
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
echo "export PATH=${PATH}:/usr/local/cuda/bin" >> ~/.bashrc
# shellcheck source=.bashrc
. "${HOME}"/.bashrc
sudo ldconfig

sudo chown -R ubuntu:ubuntu /home/ubuntu
curl -Ls https://bootstrap.pypa.io/get-pip.py | sudo -H python3.7
sudo -H pip3.7 install cython
pip3.7 install Pillow==7.1.2 attrs awscli decorator gluoncv mxnet-cu102mkl numpy pyyaml xgboost
sudo chown -R ubuntu:ubuntu /home/ubuntu

echo "#########################################################################"
echo "# install eigen to most updated version                                 #"
echo "#########################################################################"
# https://github.com/eigenteam/eigen-git-mirror/blob/master/INSTALL
mkdir -p ${INSTALL_SRC}/eigen
pushd ${INSTALL_SRC}/eigen || pushd_fail
git clone https://github.com/eigenteam/eigen-git-mirror .
git checkout 3.3.7
mkdir -p build
pushd build || pushd_fail
cmake ..
sudo make install
sudo cp -r /usr/local/include/eigen3 /usr/include/eigen3
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# calibration toolkit                                                   #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/calibration_toolkit
pushd ${CATKIN_SRC}/calibration_toolkit || pushd_fail
git clone https://github.com/iaslab-unipd/calibration_toolkit .
git fetch origin --tags
git checkout tags/v0.2
popd || popd_fail

echo "#########################################################################"
echo "# ros realsense                                                         #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/realsense
pushd ${CATKIN_SRC}/realsense || pushd_fail
git clone --branch 2.2.12 --single-branch https://github.com/intel-ros/realsense.git .
popd || popd_fail

echo "#########################################################################"
echo "# zed ros wrapper                                                       #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/zed-ros-wrapper
pushd ${CATKIN_SRC}/zed-ros-wrapper || pushd_fail
git clone --branch v2.8.x https://github.com/stereolabs/zed-ros-wrapper.git .
popd || popd_fail

echo "#########################################################################"
echo "# azure ros driver                                                  #"
echo "#########################################################################"
# The most recent tagged release is pretty far behind, so we check out the
# November 7, 2019 commit. If there is a tagged release after this, we should
# update the checkout hash
mkdir -p ${CATKIN_SRC}/Azure_Kinect_ROS_Driver
pushd ${CATKIN_SRC}/Azure_Kinect_ROS_Driver || pushd_fail
git clone https://github.com/microsoft/Azure_Kinect_ROS_Driver .
git checkout 8c6964fcc30827b476d6c18076e291fc22daa702
popd || popd_fail

# Add the Microsoft repository, get the libk4a package, extract and install libdepthengine .
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
${APT_CMD} update
${APT_CMD} install --download-only libk4a1.3
sudo dpkg -x /var/cache/apt/archives/libk4a*.deb ${INSTALL_SRC}/libk4a
sudo cp ${INSTALL_SRC}/libk4a/usr/lib/x86_64-linux-gnu/libdepthengine.so.2.0 /usr/lib/x86_64-linux-gnu

echo "#########################################################################"
echo "# install iai_kinect2                                                   #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/iai_kinect2
pushd ${CATKIN_SRC}/iai_kinect2 || pushd_fail
git clone --branch 1607 https://github.com/OpenPTrack/iai_kinect2.git .
popd || popd_fail

echo "#########################################################################"
echo "# build and install librealsense2                                       #"
echo "#########################################################################"
export PATH=${PATH}:/usr/local/cuda/bin
mkdir -p ${INSTALL_SRC}/librealsense
pushd ${INSTALL_SRC}/librealsense || pushd_fail
git clone https://github.com/IntelRealSense/librealsense.git .
git checkout v2.34.1
./scripts/setup_udev_rules.sh
mkdir -p build
pushd build || pushd_fail
cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=true
make -j"${NPROC}"
sudo make install

echo "#########################################################################"
echo "# install zed sdk                                                       #"
echo "#########################################################################"
echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
${APT_CMD} update
mkdir -p ${INSTALL_SRC}/zed
pushd ${INSTALL_SRC}/zed || pushd_fail
curl -kLs https://download.stereolabs.com/zedsdk/2.8/cu102/ubuntu18 -o ZED_SDK_Linux_Ubuntu18_cuda102_v2.8.5b.run
chmod +x ./ZED_SDK_Linux_Ubuntu18_cuda102_v2.8.5b.run
sudo ./ZED_SDK_Linux_Ubuntu18_cuda102_v2.8.5b.run --noexec --keep --target install
pushd install || pushd_fail
sudo  ./linux_install_release.sh silent
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# build and install the Azure Kinect Sensor SDK.                        #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/Azure-Kinect-Sensor-SDK
pushd ${INSTALL_SRC}/Azure-Kinect-Sensor-SDK || pushd_fail
git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git .
git checkout v1.3.0
mkdir -p build
pushd build || pushd_fail
cmake .. -GNinja
ninja
sudo ninja install
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# install freenect2                                                     #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/libfreenect2
pushd ${INSTALL_SRC}/libfreenect2 || pushd_fail
git clone --branch 1606 https://github.com/OpenPTrack/libfreenect2.git .
mkdir -p build
pushd build || pushd_fail
sudo ln -sf /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so
cmake .. -DENABLE_CXX11=ON -DCUDA_PROPAGATE_HOST_FLAGS=off
make -j"${NPROC}"
sudo make install
popd || popd_fail
popd || popd_fail

sudo tee /etc/udev/rules.d/90-kinect2.rules > /dev/null << EOF
# ATTR{product}=="Kinect2"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c4", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02d8", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02d9", MODE="0666"
EOF

echo "#########################################################################"
echo "# install ceres_solver                                                  #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/ceres
pushd ${INSTALL_SRC}/ceres || pushd_fail
git clone https://ceres-solver.googlesource.com/ceres-solver .
git fetch --tags
git checkout tags/1.12.0
mkdir -p build
pushd build || pushd_fail
cmake .. -DEIGEN_INCLUDE_DIR=/usr/local/include/eigen3
make
# make test
sudo make install
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# install LLVM                                                          #"
echo "#########################################################################"
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

echo "#########################################################################"
echo "# install torch                                                         #"
echo "#########################################################################"
mkdir -p /opt/src/torch
cd /opt/src/torch
git clone https://github.com/nagadomi/distro.git . --recursive
bash install-deps
./install.sh -b
# shellcheck source=.bashrc
. "${HOME}"/.bashrc
cd install/bin
./luarocks install dpnn
./luarocks install csvigo

echo "#########################################################################"
echo "# install openface                                                      #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/openface
cd ${INSTALL_SRC}/openface
git clone https://github.com/cmusatyalab/openface.git .
sudo python2.7 setup.py install

echo "#########################################################################"
echo "# install dlib                                                          #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/dlib
cd ${INSTALL_SRC}/dlib
git clone  --branch v19.4 https://github.com/davisking/dlib.git .
mkdir python_examples/build
cd python_examples/build
cmake ../../tools/python -DUSE_AVX_INSTRUCTIONS=ON
cmake --build . --config Release
sudo cp dlib.so /usr/local/lib/python2.7/dist-packages

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
sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/' config.cmake
sed -i 's#set(USE_LLVM OFF)#set(USE_LLVM /usr/lib/llvm-10/bin/llvm-config)#' config.cmake
sed -i 's/set(USE_CUDNN OFF)/set(USE_CUDNN ON)/' config.cmake
sed -i 's/set(USE_CUBLAS OFF)/set(USE_CUBLAS ON)/' config.cmake
sed -i 's/set(USE_THRUST OFF)/set(USE_THRUST ON)/' config.cmake
cmake ..
make -j"${NPROC}"
sudo make install
# install the python libraries -- it only works in py3
# and we won't use tvm to ever compile device side, only run side
pushd ${INSTALL_SRC}/tvm/python || pushd_fail
python3.7 setup.py install --user
popd || popd_fail
pushd ${INSTALL_SRC}/tvm/topi/python
python3.7 setup.py install --user
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# install pytorch and pytorch deps                                      #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/pytorch
pushd ${INSTALL_SRC}/pytorch || pushd_fail
git clone --branch v1.4.0 --recursive https://github.com/pytorch/pytorch .
git submodule sync
git submodule update --init --recursive
USE_NCCL=0 USE_DISTRIBUTED=0 python3.7 setup.py install --user
popd || popd_fail

echo "#########################################################################"
echo "# install pytorch vision                                                #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/torchvision
pushd ${INSTALL_SRC}/torchvision || pushd_fail
git clone --branch v0.5.0 --recursive https://github.com/pytorch/vision.git .
python3.7 setup.py install --user
popd || popd_fail

echo "#########################################################################"
echo "# install ROS                                                           #"
echo "#########################################################################"
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
${APT_CMD} update
${APT_CMD} install \
  ros-melodic-desktop-full \
  ros-melodic-ddynamic-reconfigure \
  ros-melodic-rgbd-launch \
  python-rosdep \
  python-rosinstall \
  python-rosinstall-generator \
  python-wstool
sudo rosdep init
rosdep update
grep -q -F '. /opt/ros/melodic/setup.bash' "${HOME}"/.bashrc || \
  echo ". /opt/ros/melodic/setup.bash" >> "${HOME}"/.bashrc
# shellcheck source=.bashrc
. "${HOME}"/.bashrc

echo "#########################################################################"
echo "# openptrack deps and clone necessary deps                               #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/open_ptrack
pushd ${CATKIN_SRC}/open_ptrack || pushd_fail
git clone --branch 1804_m2 https://github.com/ammolitor/open_ptrack_v2.git .
pushd ${CATKIN_SRC}/open_ptrack/rtpose_wrapper || pushd_fail
sed -i -e 's/CUDNN_CROSS_CORRELATION/CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT/g' include/caffe/util/cudnn.hpp
sudo ln -sf /usr/lib/x86_64-linux-gnu/liblapack.so.3 /usr/lib/liblapack.so.3
make all -j"${NPROC}"
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# fetch various model data files                                        #"
echo "#########################################################################"
curl -kL https://food-safety-infra.s3-us-west-2.amazonaws.com/openptrack/models/coco.weights \
  -o ${CATKIN_SRC}/open_ptrack/yolo_detector/darknet_opt/coco.weights
curl -kL https://food-safety-infra.s3-us-west-2.amazonaws.com/openptrack/models/shape_predictor_68_face_landmarks.dat \
  -o ${CATKIN_SRC}/open_ptrack/recognition/data/shape_predictor_68_face_landmarks.dat
curl -kL https://food-safety-infra.s3-us-west-2.amazonaws.com/openptrack/models/nn4.small2.v1.t7 \
  -o ${CATKIN_SRC}/open_ptrack/recognition/data/nn4.small2.v1.t7

echo "#########################################################################"
echo "# use catkin_make to build open_ptrack                                  #"
echo "#########################################################################"
pushd ${CATKIN_WS} || pushd_fail
. /opt/ros/melodic/setup.bash
export LD_LIBRARY_PATH=/root/workspace/ros/devel/lib:/opt/ros/melodic/lib:/opt/ros/melodic/lib/x86_64-linux-gnu:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/cuda/lib64
rosdep install -y -r --from-paths .
# export ROS_PARALLEL_JOBS="-j1 -l1"  # for debugging catkin_make errors
catkin_make -DCMAKE_BUILD_TYPE=Release --cmake-args "-Wno-dev"
popd || popd_fail

echo "#########################################################################"
echo " # THE FOLLOWING IS RUNTIME CONFIG                                      #"
echo "#########################################################################"
gsettings set org.gnome.Vino require-encryption false
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.lockdown disable-lock-screen true
gsettings set org.gnome.desktop.session idle-delay 0

find "${HOME}" -maxdepth 7 -mindepth 7 -type d -empty -exec rmdir '{}' \;
find "${HOME}" -maxdepth 6 -mindepth 6 -type d -empty -exec rmdir '{}' \;
find "${HOME}" -maxdepth 5 -mindepth 5 -type d -empty -exec rmdir '{}' \;
find "${HOME}" -maxdepth 4 -mindepth 4 -type d -empty -exec rmdir '{}' \;
find "${HOME}" -maxdepth 3 -mindepth 3 -type d -empty -exec rmdir '{}' \;
find "${HOME}" -maxdepth 2 -mindepth 2 -type d -empty -exec rmdir '{}' \;
find "${HOME}" -maxdepth 1 -mindepth 1 -type d -empty ! -name Desktop -exec rmdir '{}' \;

touch "${HOME}"/.hushlogin
touch "${HOME}"/.Xauthority

echo "
set -o vi
export EDITOR=vi
alias ls='ls -lh --color=auto'
export LRS_LOG_LEVEL="Fatal"" >> ~/.bashrc

echo "
. ${CATKIN_WS}/devel/setup.bash
export ROS_MASTER_URI=http://${IP_ADDR}:11311/
export ROS_IP=${IP_ADDR}
export ROS_PC_NAME=${HOSTNAME}" >> ~/.bashrc

