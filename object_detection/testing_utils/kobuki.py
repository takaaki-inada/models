import sys
import os

KeyCode_Right = 67
KeyCode_Left = 68
KeyCode_Up = 65
KeyCode_Down = 66
KeyCode_Space = 32
KeyCode_Enable = 101
KeyCode_Disable = 100

class Kobuki(object):
    def __init__(self):
        print("Kobuki init.")

    def publish(self, key):
        os.system("echo rostopic pub -1 /keyop/teleop kobuki_msgs/KeyboardInput " + str(key))

    def go_forward(self):
        self.publish(KeyCode_Up)
        print("^")

    def turn_right(self):
        self.publish(KeyCode_Right)
        print(">")

    def turn_left(self):
        self.publish(KeyCode_Left)
        print("<")
