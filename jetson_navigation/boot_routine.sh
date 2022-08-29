#!/bin/bash 

sleep 10
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'

vncserver -geometry 800x600

sleep 60
sudo nvpmodel -m 0
sudo /usr/bin/jetson_clocks
