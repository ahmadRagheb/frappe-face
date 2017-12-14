# -*- coding: utf-8 -*-
# Copyright (c) 2017, Frappe Technologies and contributors
# For license information, please see license.txt




from __future__ import unicode_literals
import frappe
from frappe.model.document import Document

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import face_recognition
import base64


import json
import logging
from websocket_server import WebsocketServer

from PIL import Image

import io

from io import BytesIO
from io import StringIO

from StringIO import StringIO
import urllib2

import re
import cStringIO
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import os
import scipy.misc

clients = {}
clients_counter={}
one_user_img=''
numer_of_faces_in_img=0


COUNTER = 0
TOTAL = 0

@frappe.whitelist(allow_guest=True)
def blink(usr,request,login_encoding_face,client):


    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 1

    # initialize the frame counters and the total number of blinks

    SUCCESS = False


    s = login_encoding_face
    #decoding the numpy array 
    r = base64.decodestring(s)
    q = np.frombuffer(r, dtype=np.float64)
    obama_face_encoding= q 

    #compare between q and t return ture or false
    #self.z = str(np.allclose(q, t))

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    name = 'uni'

    # path = (frappe.get_site_path('public', "shape_predictor_68_face_landmarks.dat"))
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    pp = '/home/ahmad/Desktop/habash/imageprocess/sites/local22/public/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(str(pp))

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    from binascii import a2b_base64

    strOne = 'b'+request
    strOne = strOne.partition(",")[2]
    pad = len(strOne)%4
    strOne += b"="*pad
    



    request = a2b_base64(strOne)

    image=request
    s = StringIO()
    s.write(image)
    print s.tell()

    size_of_ob = s.tell()
    if (size_of_ob > 0 ):
        image_bytes = io.BytesIO(image)
        image_bytes.seek(0)  # rewind to the start

        try:
            im = Image.open(image_bytes)
            arrw = np.array(im)
            arr = arrw[:,:,:3]

            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            face_locations = face_recognition.face_locations(gray)

            global numer_of_faces_in_img 

            numer_of_faces_in_img = len(face_locations)

            if len(face_locations)>=1:
                print("I found {} face(s) in this photograph.".format(len(face_locations)))

                face_encodings = face_recognition.face_encodings(arr, face_locations)

                face_names = []

                for face_encoding in face_encodings:
                    match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
                    name = "Unknown"

                    if match[0]:
                        name = str(usr)

                    face_names.append(name)

                # if len(face_locations)==1:
                    # global one_user_img 
                    # one_user_img = return_face_encoding



                # detect faces in the grayscale frame
                rects = detector(gray, 0)

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                
                    # Draw a box around the face
                    cv2.rectangle(arr, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(arr, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(arr, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    
                    # Display the resulting image
                    if name == str(usr):

                        for rect in rects:
                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

                            # extract the left and right eye coordinates, then use the
                            # coordinates to compute the eye aspect ratio for both eyes
                            leftEye = shape[lStart:lEnd]
                            rightEye = shape[rStart:rEnd]
                            leftEAR = eye_aspect_ratio(leftEye)
                            rightEAR = eye_aspect_ratio(rightEye)

                            # average the eye aspect ratio together for both eyes
                            ear = (leftEAR + rightEAR) / 2.0

                            # compute the convex hull for the left and right eye, then
                            # visualize each of the eyes
                            leftEyeHull = cv2.convexHull(leftEye)
                            rightEyeHull = cv2.convexHull(rightEye)
                            cv2.drawContours(arr, [leftEyeHull], -1, (0, 255, 0), 1)
                            cv2.drawContours(arr, [rightEyeHull], -1, (0, 255, 0), 1)

                            # check to see if the eye aspect ratio is below the blink
                            # threshold, and if so, increment the blink frame counter

                            # global COUNTER

                            if ear < EYE_AR_THRESH:
                                clients_counter[client['id']]["COUNTER"] += 1

                                print str(clients_counter[client['id']]["COUNTER"])
                            # otherwise, the eye aspect ratio is not below the blink
                            # threshold
                            else:
                                # if the eyes were closed for a sufficient number of
                                # then increment the total number of blinks
                                if clients_counter[client['id']]["COUNTER"] >= EYE_AR_CONSEC_FRAMES:
                                    # global TOTAL
                                    clients_counter[client['id']]["TOTAL"] += 1


                                # reset the eye frame counter
                                clients_counter[client['id']]["COUNTER"] = 0

                            # draw the total number of blinks on the frame along with
                            # the computed eye aspect ratio for the frame
                            cv2.putText(arr, "Blinks: {}".format(clients_counter[client['id']]["TOTAL"]), (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(arr, "EAR: {:.2f}".format(ear), (300, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                imw = Image.fromarray(arr.astype("uint8"))
                rawBytes = io.BytesIO()
                imw.save(rawBytes, "PNG")
                rawBytes.seek(0)  # return to the start of the file

                return [str(base64.b64encode(rawBytes.read())),clients_counter[client['id']]["TOTAL"] ]

            else:

                imw = Image.fromarray(arr.astype("uint8"))
                rawBytes = io.BytesIO()
                imw.save(rawBytes, "PNG")
                rawBytes.seek(0)  # return to the start of the file

                return [str(base64.b64encode(rawBytes.read())),clients_counter[client['id']]["TOTAL"] ]
        
        
        except Exception as e:
            print(e)


    else:
        print 'zeeeeeeero'   
        # return "No Users"

@frappe.whitelist(allow_guest=True)
def eye_aspect_ratio( eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear



def client_left(client, server):
    msg = "Client (%s) left" % client['id']
    print msg
    try:
        clients.pop(client['id'])
        clients_counter.pop(client['id'])

    except:
        print "Error in removing client %s" % client['id']
    for cl in clients.values():
        server.send_message(cl, msg)


def new_client(client, server):
    msg = "New client (%s) connected" % client['id']
    print msg
    for cl in clients.values():
        server.send_message(cl, msg)
    clients[client['id']] = client
    
    clients_counter[client['id']] = {"client":client,"COUNTER" : 0, "TOTAL" : 0}


@frappe.whitelist(allow_guest=True)
def msg_received(client, server, msg):
    import pickle
    import ast

    clientid = client['id']

    for cl in clients:
            if cl == clientid:
                cl = clients[cl]

                try:
                    msg_in_json= json.loads(msg)

                # py_obj = ast.literal_eval(msg)
                    stop = msg_in_json['stop']

                    username= msg_in_json['username']
                    flag = msg_in_json["flag"]
                    img_data = msg_in_json["image"]
                    login_encoding_face = msg_in_json["login_encoding_face"]
                    # path = msg_in_json["path"]

                    if flag == 'login_page':

                        if stop=="True":
                        
                            print "Stop"
                            json_mylist = json.dumps(["Stop",""])
                            server.send_message(cl, json_mylist)


                        else:
                            processed_image = blink(username,img_data,login_encoding_face,client)
                            
                            if processed_image[1] >=1:
                                json_mylist = json.dumps(["SUCCESS",processed_image[0]])
                                server.send_message(cl, json_mylist)
                            else:
                                json_mylist = json.dumps(["Live",processed_image[0]])
                                server.send_message(cl, json_mylist)


                except Exception as e:
                    raise e




server = WebsocketServer(9001,host='0.0.0.0')
server.set_fn_client_left(client_left)
server.set_fn_new_client(new_client)
"""Sets a callback function that will be called when a client sends a message """
server.set_fn_message_received(msg_received)
server.run_forever()



