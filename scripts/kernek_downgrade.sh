#!/usr/bin/env bash
# this is a hacky script/set of instructions for installing an older kernel
# and removing the newer one. There is most certainly a better way. this is
# clunky but clean.

# INSTALL DESIRED KERNEL VERSION PACKAGES
sudo apt-get -y install \
  linux-headers-5.3.0-46 \
  linux-headers-5.3.0-46-generic \
  linux-image-5.3.0-46-generic \
  linux-modules-5.3.0-46-generic \
  linux-modules-extra-5.3.0-46-generic

echo "PLEASE SELECT 5.3.0-46 KERNEL ON BOOT"
reboot

# RUN THESE COMMANDS TO REMOVE THE UNDESIRED NEWER KERNEL
apt list --installed | awk -F/ '/5.3.0-51/{print $1}' | xargs sudo apt-get remove -y

# RE-COMPILE/INSTALL THE NVIDIA DRIVER AND CUDA
cd /opt/src/nvidia/
sudo ./nvidia.run --silent --no-questions
sudo ./cuda.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.2 --samples --samplespath=/opt/src/nvidia/samples
