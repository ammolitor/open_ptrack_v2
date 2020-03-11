#!/usr/bin/env bash
#  adapted from: https://prahladyeri.com/blog/2017/09/how-to-trim-your-new-ubuntu-installation-of-extra-fat-and-make-it-faster.html

sudo sed -i 's/NoDisplay=true/NoDisplay=false/g' /etc/xdg/autostart/*.desktop

sudo systemctl disable cupsd
sudo systemctl disable cups-browsed
sudo systemctl disable avahi-daemon

sudo chmod -x /usr/lib/x86_64-linux-gnu/hud/hud-service # 64bit systems
sudo mv /usr/lib/evolution-data-server /usr/lib/evolution-data-server-disabled
sudo mv /usr/lib/evolution /usr/lib/evolution-disabled

sudo apt purge -y gnome-software
sudo apt purge -y libreoffice\*
sudo apt purge -y update-manager

# check what snaps are installed
snap list
# remove all snaps
sudo apt purge -y snapd
# delete the snap folder
rm -rf snap

sudo apt autoremove
