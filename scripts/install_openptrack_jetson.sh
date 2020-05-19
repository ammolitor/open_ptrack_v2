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

  if [ "${1}" == "auto" ]; then
  sudo update-alternatives --auto libblas.so-aarch64-linux-gnu
  sudo update-alternatives --auto libblas.so.3-aarch64-linux-gnu
  sudo update-alternatives --auto liblapack.so-aarch64-linux-gnu
  sudo update-alternatives --auto liblapack.so.3-aarch64-linux-gnu
  fi
}

NPROC=$(nproc)
IP_ADDR=$(ip addr show | awk '/inet / && !/127/{gsub("/..", "", $2); print $2}')
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
${APT_CMD} purge libopencv* || true
${APT_CMD} purge opencv* || true
${APT_CMD} autoremove || true
${APT_CMD} update
${APT_CMD} upgrade

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
  net-tools \
  ninja-build \
  nlohmann-json-dev \
  ntp \
  pkg-config \
  protobuf-compiler \
  python-pip \
  python-qt4 \
  python3 \
  python3-numpy \
  python3-opencv \
  python3-pip \
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

pip install requests numpy pyyaml typing
pip3 install cython  # this is by itself on purpose (numpy needs it and I
                     # dont trust pip to install them in the right order)
pip3 install numpy decorator attrs pyyaml awscli

echo "#########################################################################"
echo "# install librealsense2                                                 #"
echo "#########################################################################"
# https://github.com/dorodnic/librealsense/blob/jetson_doc/doc/installation_jetson.md
# https://github.com/IntelRealSense/librealsense/issues/4969
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
${APT_CMD} update
${APT_CMD} install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg

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
echo "# install iai_kinect2                                                   #"
echo "#########################################################################"
mkdir -p ${CATKIN_SRC}/iai_kinect2
pushd ${CATKIN_SRC}/iai_kinect2 || pushd_fail
git clone --branch 1607 https://github.com/OpenPTrack/iai_kinect2.git .
popd || popd_fail

echo "#########################################################################"
echo "# install freenect2                                                     #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/libfreenect2
pushd ${INSTALL_SRC}/libfreenect2 || pushd_fail
git clone --branch 1606 https://github.com/OpenPTrack/libfreenect2.git .
# mesa links libGL.so incorrectly on arm v8, properly link libGL
sudo rm -f /usr/lib/aarch64-linux-gnu/libGL.so
sudo ln -sf /usr/lib/aarch64-linux-gnu/libGL.so.1.0.0 /usr/lib/aarch64-linux-gnu/libGL.so
mkdir -p build
pushd build || pushd_fail
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
sed -i 's/set(USE_CUDNN OFF)/set(USE_CUDNN ON)/' config.cmake
sed -i 's/set(USE_CUBLAS OFF)/set(USE_CUBLAS ON)/' config.cmake
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' config.cmake
cmake -GNinja ..
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

echo "#########################################################################"
echo "# install pytorch and pytorch deps                                      #"
echo "#########################################################################"
mkdir -p ${INSTALL_SRC}/pytorch
pushd ${INSTALL_SRC}/pytorch || pushd_fail
git clone --branch v1.4.0 --recursive https://github.com/pytorch/pytorch .
git submodule sync
git submodule update --init --recursive
USE_NCCL=0 USE_DISTRIBUTED=0 TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2" python setup.py install --user
# USE_NCCL=0 USE_DISTRIBUTED=0 TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2" python3 setup.py install --user
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
git clone https://github.com/ammolitor/open_ptrack_v2 .
pushd ${CATKIN_SRC}/open_ptrack/rtpose_wrapper || pushd_fail
make all -j"${NPROC}"
popd || popd_fail
popd || popd_fail

echo "#########################################################################"
echo "# fetch various model data files                                        #"
echo "#########################################################################"
curl -kL https://PLACEHOLDER/openptrack/models/coco.weights \
  -o ${CATKIN_SRC}/open_ptrack/yolo_detector/darknet_opt/coco.weights
curl -kL https://PLACEHOLDER/openptrack/models/shape_predictor_68_face_landmarks.dat \
  -o ${CATKIN_SRC}/open_ptrack/recognition/data/shape_predictor_68_face_landmarks.dat
curl -kL https://PLACEHOLDER/openptrack/models/nn4.small2.v1.t7 \
  -o ${CATKIN_SRC}/open_ptrack/recognition/data/nn4.small2.v1.t7

echo "#########################################################################"
echo "# use catkin_make to build open_ptrack                                  #"
echo "#########################################################################"
pushd ${CATKIN_WS} || pushd_fail
. /opt/ros/melodic/setup.bash
rosdep install -y -r --from-paths .
# export ROS_PARALLEL_JOBS="-j1 -l1"  # for debugging catkin_make errors
catkin_make -DCMAKE_BUILD_TYPE=Release
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
alias ls='ls -lh --color=auto'" >> ~/.bashrc

echo "
. ${CATKIN_WS}/devel/setup.bash
export ROS_MASTER_URI=http://${IP_ADDR}:11311/
export ROS_IP=${IP_ADDR}
export ROS_PC_NAME=${HOSTNAME}" >> ~/.bashrc

