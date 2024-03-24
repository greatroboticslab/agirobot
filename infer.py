import os
import re
import socket
import time
import csv
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
#import torchvision.transforms as transforms

replayDir = "../robot_sim/Assets/Recordings/replay/"
replay = False

unityIp = "127.0.0.1"

def ReadImage(dir):
    
    print(dir)
    
    image = tf.io.read_file(dir)
    image = tf.io.decode_png(image)
    image = tf.cast(image, tf.float32)# / 1.0
    
    im = Image.fromarray(np.uint8(image))
    #im.save("output_infer.png")
    
    return np.asarray(image)
    

def ReadMetaData(dir):
    metaData = [0.0,0.0,0.0,0.0,0.0,0.0]
    
    firstLine = True
    
    with open(dir, newline='') as inputsFile:
        cReader = csv.reader(inputsFile, delimiter=',')
        for row in cReader:
            i = 0
            if not firstLine:
                metaData[0] = float(row[3])
                metaData[1] = float(row[4])
                metaData[2] = float(row[5])
                metaData[3] = float(row[6])
                metaData[4] = float(row[7])
                metaData[5] = float(row[8])
            firstLine = False
    
    inputsFile.close()
    
    return np.asarray(metaData)

metaData = np.zeros(6)


filenames = []
subfiles = []
if replay:
    for file in os.listdir(replayDir):
        
        
        if file.endswith(".txt"):
            subfiles.append(file[:-4])
        
    
    subfiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for f in subfiles:
        filenames.append(replayDir+f)


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 9001  # Port to listen on (non-privileged ports are > 1023)

bufferSize = 8192
res = 512

model = tf.keras.models.load_model('Saved Model')

acknowledgeMsg = bytearray(bufferSize);
#acknowledgeMsg[0] = ord('X')
for p in range(bufferSize):
    acknowledgeMsg[p] = ord('_')
acknowledgeMsg[bufferSize-1] = ord('Y')

def GetPacketInfo(data):
    command = int(data[0])
    size = int.from_bytes(data[1:5], "big")
    message = data[5:]
    
    print(str(command) + " | " + str(size) + ": " + str(message))
    
    return command, size, message

fileId = 0

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', PORT))
    print("binded")
    s.listen(5)
    print("listening")
    conn, addr = s.accept()
    print("Accepted")
    chunkId = 1
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(bufferSize)
            #conn.send(b'received1')
            conn.send(acknowledgeMsg)
            pCommand, pSize, pMessage = GetPacketInfo(data)
            data = pMessage
            if not data:
                break
            #print("PosDat")
            #print(data)
            datal = data.split()
            #conn.sendall(data)
            latitude = float(datal[0])
            longitude = float(datal[1])
            heading = float(datal[2])
            vel = float(datal[3])
            waypointLat = float(datal[4])
            waypointLong = float(datal[5])
            imgSize = int(datal[6])
            
            numData = np.asarray([latitude, longitude, heading, vel, waypointLat, waypointLong])
            
            bPos = 0
            
            imgData = np.zeros((res,res,3)).astype(np.single)
            
            inDat = conn.recv(imgSize)
            #conn.send(b'received2')
            
            
            #print(datal)
            #print(imgSize)
            #print(len(imgBytes))
            for x in range(res):
                for y in range(res):
                    for c in range(3):
                        
                        #print(str(bPos) + ", " + str(len(inDat)))
                        
                        imgData[x,(res-1)-y,c] = inDat[bPos]
                        bPos += 1
                        
            
            #Image data finished being gathered, do inference here
            #imgData = tf.zeros((512,512,3))
            
            if replay:
            
                numData = ReadMetaData(replayDir+str(fileId)+".txt")
                print(numData)
                imgData = ReadImage(replayDir+str(fileId)+".png")
                
                imgData = cv2.resize(imgData, dsize=(res, res))
                
                #imgData = np.resize(imgData, (res, res, 3))
                imgData = imgData[:,:,:3]
                
                fileId += 1
                if fileId >= len(filenames):
                    fileId = len(filenames) - 1
                    exit()
            
            
            in1 = np.asarray([numData])
            in2 = np.asarray([imgData])
            #in2 = np.flipud(in2)
            
            
            logits = model([in1, in2], training=False)
            print(logits[0])
            
            msgString = "" + str(float(logits[0][0])) + " " + str(float(logits[0][1])) + " " + str(float(logits[0][2]))
            msgData = bytearray()
            msgData.extend(map(ord, msgString))
            
            conn.send(msgData)
            
            im = Image.fromarray(np.uint8(imgData))
            #im.save("output_infer.jpg")
            
            #exit()