# -*- coding: utf-8 -*-
# Copyright (c) 2017, Frappe Technologies and contributors
# For license information, please see license.txt

from __future__ import unicode_literals
import frappe
from frappe.model.document import Document
import face_recognition
import cv2


class VideoImage(Document):
	def validate(self):
		# Raise error anyways to demonstrate validate func
		path = (frappe.get_site_path('public', 'files', 'ahmad.jpg'))
		path2 = (frappe.get_site_path('public', 'files', 'omar.jpg'))

		# picture_of_me = face_recognition.load_image_file(str(path))
		# my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

		# unknown_picture = face_recognition.load_image_file(str(path2))
		# unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

		# results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

		# if results[0] == True:
		#     frappe.throw("It's a picture of me!")
		# else:
		#     frappe.throw("It's not a picture of me!")

		# This is a super simple (but slow) example of running face recognition on live video from your webcam.
		# There's a second example that's a little more complicated but runs faster.

		# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
		# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
		# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


		# Get a reference to webcam #0 (the default one)
		# START FROM HERE 11111
		video_capture = cv2.VideoCapture(0)

		# Load a sample picture and learn how to recognize it.
		obama_image = face_recognition.load_image_file(str(path))
		obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


		# Initialize some variables
		face_locations = []
		face_encodings = []
		face_names = []
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
					match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
					name = "Unknown"

					if match[0]:
						name = "Barack"

					face_names.append(name)

			process_this_frame = not process_this_frame  # Display the results
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
				cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)  # Display the resulting image

				cv2.imshow('Video', frame)

			# Hit 'q' on the keyboard to quit!
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# Release handle to the webcam
		video_capture.release()
		cv2.destroyAllWindows()
