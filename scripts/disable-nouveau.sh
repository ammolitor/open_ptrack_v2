#!/usr/bin/env bash

sudo init 3
sudo apt purge -y xserver-xorg-video-nouveau-hwe-18.04
sudo tee /etc/modprobe.d/blacklist-nvidia-nouveau.conf > /dev/null << EOF
blacklist nouveau
options nouveau modeset=0
EOF
sudo update-initramfs -u
sudo reboot
