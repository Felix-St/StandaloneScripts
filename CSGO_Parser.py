# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:50:55 2021

@author: Felix
"""
import sys

def getMessage(file):
    byte = '0'
    bytes = []
    while byte != b'\x04':
        print(byte)
        byte = file.read(1)
        if byte != 'b\x04':
            bytes.append(byte)
    return bytes
    
def beautify(message):
    print(message)
    sys.exit()
    message.remove(b'\x00')

    new_message = " ".join(str(x) for x in message)
    return new_message

file = open(r"C:\Users\Felix\Desktop\CSGO_Demos\match730_003479444394803724918_0844591879_193.dem", "rb")
byte = file.read(0)

"""
while byte:
    print(byte)
    byte = file.read(1)
"""   
t = getMessage(file) 
t2 = getMessage(file) 
t3 = getMessage(file) 

    

 

    
"""
while True:
    byte = file.read(1)
    print(byte)
    if byte == b'\x00' or byte == b'\x04':
        break;
   
version_demo = file.read(1)
version_network = file.read(1)   
    
    
print(version_demo)
print(version_network)

server_name = file.read(260)
client_name = file.read(260)
map_name = file.read(260)
game_dir = file.read(260)

playback_time = file.read(40)
ticks = file.read(40)

""" 

file.close()


def readData(self):
    length = 0
    
    

    