#!/usr/bin/env bash
set -ex

sudo /bin/sed -i 's/%sudo	ALL=(ALL:ALL) ALL/%sudo	ALL=(ALL:ALL) NOPASSWD:ALL/' /etc/sudoers

sudo apt -y install curl git ntp python3-pip ssh tig tmux tree vim vino
pip3 install awscli
gsettings set org.gnome.Vino require-encryption false
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.lockdown disable-lock-screen true
gsettings set org.gnome.desktop.session idle-delay 0

echo "
set -o vi
export EDITOR=vi
alias ls='ls -lh --color=auto'" >> ~/.bashrc

find . -maxdepth 7 -mindepth 7 -type d -empty -exec rmdir '{}' \;
find . -maxdepth 6 -mindepth 6 -type d -empty -exec rmdir '{}' \;
find . -maxdepth 5 -mindepth 5 -type d -empty -exec rmdir '{}' \;
find . -maxdepth 4 -mindepth 4 -type d -empty -exec rmdir '{}' \;
find . -maxdepth 3 -mindepth 3 -type d -empty -exec rmdir '{}' \;
find . -maxdepth 2 -mindepth 2 -type d -empty -exec rmdir '{}' \;
find . -maxdepth 1 -mindepth 1 -type d -empty ! -name Desktop -exec rmdir '{}' \;

curl https://raw.githubusercontent.com/ammolitor/open_ptrack_v2/master/scripts/make-ubuntu-faster.sh | bash

cat << EOF | sudo tee /etc/ntp.conf
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

git clone -b 1804 https://github.com/OpenPTrack/open_ptrack_docker_config.git
pushd  open_ptrack_docker_config || exit 127
bash setup_1804_host
# sudo docker images | awk '!/REPOSITORY/{print $3}' | xargs sudo docker rmi
# sudo docker ps -a | awk '!/CONTAINER/{print $1}' | sudo xargs docker rm -f
sudo docker pull openptrackofficial/1804:latest
