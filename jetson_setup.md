# JETSON TX2 SETUP
This document will walk through the steps required to configure a new Jetson TX2
device for running Open PTrack.

## Flash the Jetson to a more recent OS. 
A separate linux system is required to complete this part. 
1. Install the Nvidia SDK Manager (https://developer.nvidia.com/nvidia-sdk-manager)
1. Connect the Jetson device per the instructions, and boot it up in force recovery mode.
1. Connect a monitor and keybaord to the Jetson, it is necessary during the OS setup process
1. Use the SDK manager to flash the device to version 4.2.3 **NOTE: this version is critical**
1. Midway through the process the SDK manager will prompt for an IP address, username, and password
1. This where it is necessary to use the monitor and keybaord connected to the Jetson to complete
the OS setup process. for the computer name, username, and password use: 
    * username `nvidia`
    * password `nvidia`
    * computer name `mic720ai-NN` where `NN` equals the device number
    * select the log me in automatically option
1. Once the OS setup process completes and the Jetson is booted up the gnome desktop,
input the ip address, username, and password into the SDK manager and allow the second
 half of the flashing process to complete. 
 
 ## Configure the device. 
 There are many things that need to be done to the Jetson device to make it usable in this
 context. A walk-through of each of them follows:
 
### Install ssh keys, and other config
From the nuc, for each Jetson node
* `scp -pr .ssh mic720ai-01:`
* `scp -pr .vim* mic720ai-01:`

on the Jetson node
* `vim ~/.bashrc`
    ```
    set -o vi
    export EDITOR=vi
    alias ls='ls -lh --color=auto'
    ```

### enable password-less sudo
* `sudo su -`
* `vim .bashrc`
    ```
    set -o vi
    export EDITOR=vi
    alias ls='ls -lh --color=auto'
    ```
* `. .bashrc`
* `visudo`
* `%sudo   ALL=(ALL:ALL) ALL` -> `%sudo   ALL=(ALL:ALL) NOPASSWD: ALL`

### enable automatic login (if it isn't already)
* `sudo vim /etc/gdm3/custom.conf`
    ```
    [daemon]
    AutomaticLoginEnable=true
    AutomaticLogin=nvidia
    ```
 
### Setup an VNC server
* `sudo vim /usr/share/glib-2.0/schemas/org.gnome.Vino.gschema.xml`
```$xml
    <key name='enabled' type='b'>
      <summary>Enable remote access to the desktop</summary>
      <description>
        If true, allows remote access to the desktop via the RFB
        protocol. Users on remote machines may then connect to the
        desktop using a VNC viewer.
      </description>
      <default>false</default>
    </key>
```
* `sudo glib-compile-schemas /usr/share/glib-2.0/schemas`
* `gsettings set org.gnome.Vino require-encryption false`
* `gsettings set org.gnome.Vino prompt-enabled false`
* `gsettings set org.gnome.desktop.screensaver lock-enabled false`
* `gsettings set org.gnome.desktop.session idle-delay 0`

* `export DISPLAY=:0`
* `/usr/lib/vino/vino-server`
* connect with a vnc viewer
* `sudo xrandr --fb 1920x1080`
* create gnome autostart for Vino (via desktop, startup applications)

### configure NTP
sudo apt -y install ntp
```
# /etc/ntp.conf, configuration for ntpd; see ntp.conf(5) for help
driftfile /var/lib/ntp/ntp.drift
# Enable this if you want statistics to be logged.
#statsdir /var/log/ntpstats/
statistics loopstats peerstats clockstats
filegen loopstats file loopstats type day enable
filegen peerstats file peerstats type day enable
filegen clockstats file clockstats type day enable
# Specify one or more NTP servers.
server 192.168.100.101 iburst
disable auth
broadcastclient
```

### cleanup home dir
run this until it no longer outputs anything
* `find . -type d -empty -exec rmdir '{}' \;`

### remove unnecessary packages
* `make-ubuntu-faster.sh`

### install OS updates
* `sudo apt -y update &&  sudo apt -y upgrade`
* `sudo apt -y clean`
* `sudo apt -y autoremove`

### Install OpenPTrack and dependencies
* `./jetson_install_openptrack.sh`

