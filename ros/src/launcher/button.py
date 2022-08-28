#!/usr/bin/env python3
#coding=utf-8

import RPi.GPIO as GPIO
import time
import os

# Pin Definitions:
but_pin = 18

# Launch scripts
def launch(channel):
	print("Launching files!")
	print("Launching navigation")
	cmd_nav = 'roslaunch turtlebot3_navigation turtlebot3_navigation_real.launch planner:=teb'
	os.system(cmd_nav)
	time.sleep(10)
	print("Launching goals")
	cmd_goals = 'rosrun turtlebot3_navigation goals.py'
	os.system(cmd_goals)

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(but_pin, GPIO.IN)  # button pin set as input

    GPIO.add_event_detect(but_pin, GPIO.FALLING, callback=launch, bouncetime=10)
    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
        	pass
    finally:
        GPIO.cleanup()  # cleanup all GPIOs

if __name__ == '__main__':
    main()