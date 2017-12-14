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




@frappe.whitelist(allow_guest=True)
def ping():
    return 'pong'

@frappe.whitelist(allow_guest=True)
def is_face_active():
    return "False valaaa"

    # check_active = frappe.db.get_value("User", str(usr),"face_check")
    # if check_active == 0:
    #     return "False valaaa"
    # else:
    #     return "True aslsl" 

@frappe.whitelist(allow_guest=True)
def blink(usr):
    
    check_active = frappe.db.get_value("User", str(usr),"face_check")
    if check_active == 0:
        return True 


    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 2

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    SUCCESS = False

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
 #    path2 = (frappe.get_site_path('public', 'files', 'ahmad.jpg'))

 #    # Load a sample picture and learn how to recognize it.


 #    obama_image = face_recognition.load_image_file(str(path2))
 #    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    

	# #load numpy array to t var

 #    t = obama_face_encoding

 #    # encoding the numpy array
 #    s = base64.b64encode(t)

    s = frappe.db.get_value("User", str(usr),"login_encoding_face")
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

    path = (frappe.get_site_path('public', "shape_predictor_68_face_landmarks.dat"))
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(path))

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")

    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    fileStream = False
    time.sleep(1.0)

    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process


        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()

        frame = imutils.resize(frame, width=450)

        # Grab a single frame of video

        # Resize frame of video to 1/4 size for faster face recognition processing

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Only process every other frame of video to save time

        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
                name = "Unknown"

                if match[0]:
                    name = str(usr)

                face_names.append(name)

        process_this_frame = not process_this_frame  # Display the results

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
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
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                    else:
                        # if the eyes were closed for a sufficient number of
                        # then increment the total number of blinks
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1

                        # reset the eye frame counter
                        COUNTER = 0

                    # draw the total number of blinks on the frame along with
                    # the computed eye aspect ratio for the frame
                    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if TOTAL == 2:
            SUCCESS = True
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    return SUCCESS

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



"""
@frappe.whitelist(allow_guest=True)
def face_capture():

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    #obama_image = face_recognition.load_image_file("obama.jpg")
    #obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []

    og='s'
    return_face_encoding =[]
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                #match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
                name = "Unknown"

                return_face_encoding=face_encoding
                # if match[0]:
                #     name = "Barack"

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if len(face_locations)>1:
                frappe.throw( 'Not Valid multiple faces captured , Please make sure only your face appers in the cam')
                break
            elif len(face_locations)<1:
                frappe.throw( 'Not Valid, zero face detected please try again')
                break
            elif len(face_locations)==1:
                og= base64.b64encode(return_face_encoding)
                break
            else:
                og= "nothing"
                break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return og
"""


@frappe.whitelist(allow_guest=True)
def face_capture():
    import logging
    from websocket_server import WebsocketServer
    clients = {}

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
        # msg = "Client (%s) : %s" % (client['id'], msg)
        # print msg
        clientid = client['id']
        for cl in clients:
            if cl == clientid:
                cl = clients[cl]
                server.send_message(cl, msg)


    server = WebsocketServer(9001,host='0.0.0.0')
    server.set_fn_client_left(client_left)
    server.set_fn_new_client(new_client)
    """Sets a callback function that will be called when a client sends a message """
    server.set_fn_message_received(msg_received)
    server.run_forever()







import logging
from websocket_server import WebsocketServer


clients = {}
def img_handel(img):


    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []

    og='s'
    return_face_encoding =[]
    process_this_frame = True

    while True:
        # Grab a single frame of video
        frame = img

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                #match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
                name = "Unknown"

                return_face_encoding=face_encoding
                # if match[0]:
                #     name = "Barack"

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if len(face_locations)>1:
                frappe.throw( 'Not Valid multiple faces captured , Please make sure only your face appers in the cam')
                break
            elif len(face_locations)<1:
                frappe.throw( 'Not Valid, zero face detected please try again')
                break
            elif len(face_locations)==1:
                og= base64.b64encode(return_face_encoding)
                break
            else:
                og= "nothing"
                break

    # Release handle to the webcam
    # video_capture.release()
    # cv2.destroyAllWindows()

        return img

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
    # msg = "Client (%s) : %s" % (client['id'], msg)
    # print msg
    clientid = client['id']
   
    

    for cl in clients:
            if cl == clientid:
                cl = clients[cl]
                aa = img_handel(msg)
                server.send_message(cl, aa)


server = WebsocketServer(9001,host='0.0.0.0')
server.set_fn_client_left(client_left)
server.set_fn_new_client(new_client)
"""Sets a callback function that will be called when a client sends a message """
server.set_fn_message_received(msg_received)
server.run_forever()



