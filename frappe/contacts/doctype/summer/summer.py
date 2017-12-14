# -*- coding: utf-8 -*-
# Copyright (c) 2017, Frappe Technologies and contributors
# For license information, please see license.txt

from __future__ import unicode_literals
import frappe
from frappe.model.document import Document
from PIL import Image, ImageOps
import face_recognition


class Summer(Document):
	def validate(self):
		# Raise error anyways to demonstrate validate func
		path=(frappe.get_site_path('public','files','mohamad.jpg'))
		path2=(frappe.get_site_path('public','files','omar.jpg'))

		picture_of_me = face_recognition.load_image_file(str(path))
		my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

		unknown_picture = face_recognition.load_image_file(str(path2))
		unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

		results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

		if results[0] == True:
		    frappe.throw("It's a picture of me!")
		else:
		    frappe.throw("It's not a picture of me!")				
		     		#frappe.throw(("Invalid condition"))
		#img = Image.open('/files/')

		#img = Image.open(frappe.get_site_path('public', 'files', 'mohammad.jpg'))
#
#
#
#
#face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/

