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
import socket

clients = {}


one_user_img=''
numer_of_faces_in_img=0


def fankosh(request):
    import re
    import cStringIO
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from io import BytesIO
    import os
    import scipy.misc


    
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
                    name = "Unknown"

                    return_face_encoding=face_encoding

                    face_names.append(name)

                if len(face_locations)==1:
                    global one_user_img 
                    one_user_img = return_face_encoding
        

                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):

                    # Draw a box around the face
                    cv2.rectangle(arr, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(arr, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(arr, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


                imw = Image.fromarray(arr.astype("uint8"))
                rawBytes = io.BytesIO()
                imw.save(rawBytes, "PNG")
                rawBytes.seek(0)  # return to the start of the file

                return str(base64.b64encode(rawBytes.read()))

            else:

                imw = Image.fromarray(arr.astype("uint8"))
                rawBytes = io.BytesIO()
                imw.save(rawBytes, "PNG")
                rawBytes.seek(0)  # return to the start of the file

                return str(base64.b64encode(rawBytes.read()))
        
        except:
            print "Unable to load image"
            return "Unable to load image"

    else:
        print 'zeeeeeeero'   
        return "No Users"


def client_left(client, server):
    msg = "Client (%s) left" % client['id']
    print msg
    try:
        clients.pop(client['id'])
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


def msg_received(client, server, msg):

    clientid = client['id']

    for cl in clients:
            if cl == clientid:
                cl = clients[cl]

                if msg=="True":
                    if numer_of_faces_in_img ==1 :
                        print "one"
                        json_mylist = json.dumps(["One",base64.b64encode(one_user_img)])
                        server.send_message(cl, json_mylist)

                    elif numer_of_faces_in_img > 1 :
                        print "Many"
                        json_mylist = json.dumps(["Many Users","hi successfull access to it bro"])

                        server.send_message(cl, json_mylist)
                    elif numer_of_faces_in_img == 0 :
                        print "none"
                        json_mylist = json.dumps(["No User"])
                        server.send_message(cl, json_mylist)
                else:
                    processed_image = fankosh(msg)
                    json_mylist = json.dumps(["Live",processed_image])
                    server.send_message(cl, json_mylist)




# server = WebsocketServer(9001,host='0.0.0.0')
server = WebsocketServer(9001,host=str(socket.gethostname()))
server.set_fn_client_left(client_left)
server.set_fn_new_client(new_client)
"""Sets a callback function that will be called when a client sends a message """
server.set_fn_message_received(msg_received)
server.run_forever()



