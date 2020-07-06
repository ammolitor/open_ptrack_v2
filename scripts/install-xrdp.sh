#!/usr/bin/env bash

# install instructions adapted from the folloeing:
# https://www.hiroom2.com/2018/04/29/ubuntu-1804-xrdp-gnome-en/
# https://c-nergy.be/blog/?p=11868
# https://github.com/neutrinolabs/xrdp

sudo apt-get -y install xrdp xorgxrdp-hwe-18.04
sudo apt-get -y  install gnome-tweak-tool
sudo sed -i 's/allowed_users=console/allowed_users=anybody/' /etc/X11/Xwrapper.config

sudo sed -e 's/^new_cursors=true/new_cursors=false/g' -i /etc/xrdp/xrdp.ini
sudo systemctl restart xrdp

tee ~/.xsessionrc > /dev/null << EOF
export GNOME_SHELL_SESSION_MODE=ubuntu
export XDG_CURRENT_DESKTOP=ubuntu:GNOME
export XDG_DATA_DIRS=/usr/share/ubuntu:/usr/local/share:/usr/share:/var/lib/snapd/desktop
export XDG_CONFIG_DIRS=/etc/xdg/xdg-ubuntu:/etc/xdg
EOF

sudo bash -c "cat >/etc/polkit-1/localauthority/50-local.d/45-allow.colord.pkla" <<EOF
[Allow Colord all Users]
Identity=unix-user:*
Action=org.freedesktop.color-manager.create-device;org.freedesktop.color-manager.create-profile;org.freedesktop.color-manager.delete-device;org.freedesktop.color-manager.delete-profile;org.freedesktop.color-manager.modify-device;org.freedesktop.color-manager.modify-profile
ResultAny=no
ResultInactive=no
ResultActive=yes
EOF

gnome-shell-extension-tool -e ubuntu-dock@ubuntu.com
gnome-shell-extension-tool -e ubuntu-appindicators@ubuntu.com

sudo reboot






