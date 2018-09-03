import cv2 as cv
import urllib
import urllib.request
import numpy as np
import base64
import face_recognition

class Camera:
    def __init__(self, threshold, faces_dict, url, username, password):
        self.threshold = threshold
        self.faces_dict = faces_dict
        self.url = url
        self.username = username
        self.password = password

    def detect_faces_in_image(self, frame_data):
        # Load the uploaded image file
        frame = np.asarray(bytearray(frame_data), dtype="uint8")
        frame = cv.imdecode(frame, cv.IMREAD_COLOR)
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Get face encodings for any faces in the uploaded image
        uploaded_faces = face_recognition.face_encodings(rgb_small_frame)

        # Defaults for the result object
        faces_found = len(uploaded_faces)
        faces = []

        if faces_found:
            face_encodings = list(self.faces_dict.values())
            for uploaded_face in uploaded_faces:
                match_results = face_recognition.compare_faces(
                    face_encodings, uploaded_face)
                for idx, match in enumerate(match_results):
                    if match:
                        match = list(self.faces_dict.keys())[idx]
                        match_encoding = face_encodings[idx]
                        dist = face_recognition.face_distance([match_encoding],
                                uploaded_face)[0]
                        faces.append({
                            "id": match,
                            "dist": dist
                        })

        return faces_found, faces

    def start_loop(self):
        request = urllib.request.Request(self.url)
        if self.username and self.password:
            base64string = base64.encodestring(("%s:%s" % (self.username, self.password)).encode()).decode().replace("\n", "")
            request.add_header("Authorization", "Basic %s" % base64string) 

        while True:
            img = urllib.request.urlopen(request).read()
            faces_found, faces = self.detect_faces_in_image(img)

            print(faces_found)
            for f in filter(lambda f: float(f["dist"]) >= self.threshold, faces):
                print("%s: %s" % (f["id"], f["dist"]))