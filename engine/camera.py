import cv2 as cv
import urllib
import urllib.request
import numpy as np
import base64
import face_recognition
import time
import sys
import urllib.parse

class GlobalConfiguration:
    def __init__(self, vera_ip: str, threshold: float, faces_dict):
        self.threshold = threshold
        self.vera_ip = vera_ip
        self.faces_dict = faces_dict

class CameraConfiguration:
    def __init__(self, fps: float, url: str, username: str, password: str, deviceId: int):
        self.url = url
        self.fps = fps
        self.deviceId = deviceId
        self.username = username
        self.password = password

    def get_credentials_uri(self):
        if self.username and self.password:
            return self.url.replace("http://", "http://%s:%s@" % (urllib.request.quote(self.username), urllib.request.quote(self.password)))
        return self.url

class Camera:
    def __init__(self, globalConfiguration: GlobalConfiguration, configuration: CameraConfiguration):
        self.configuration = configuration
        self.globalConfiguration = globalConfiguration
        self.lastFrame = 0.0

    def detect_faces_in_image(self, frame_data, raw):
        # Load the uploaded image file
        if raw:
            frame = np.asarray(bytearray(frame_data), dtype="uint8")
            frame = cv.imdecode(frame, cv.IMREAD_COLOR)
        else:
            frame = frame_data
        
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
            face_encodings = list(self.globalConfiguration.faces_dict.values())
            for uploaded_face in uploaded_faces:
                match_results = face_recognition.compare_faces(
                    face_encodings, uploaded_face)
                for idx, match in enumerate(match_results):
                    if match:
                        match = list(self.globalConfiguration.faces_dict.keys())[idx]
                        match_encoding = face_encodings[idx]
                        dist = face_recognition.face_distance([match_encoding],
                                uploaded_face)[0]
                        faces.append({
                            "id": match,
                            "dist": dist
                        })

        return faces_found, faces
    
    def call_vera(self, ids):
        ids = ",".join(ids)
        uri = "http://%s:3480/data_request?id=action&output_format=json&DeviceNum=%s&serviceId=urn:ricciolo:serviceId:PresenceSensor1&action=SetPresent&newPresentValue=%s" %(self.globalConfiguration.vera_ip, self.configuration.deviceId, ids)
        request = urllib.request.Request(uri)
        try:
            stream = urllib.request.urlopen(request)
        finally:
            stream.close()
    
    def start(self):
        while True:
            diff = time.time() - self.lastFrame
            if diff < 1 / self.configuration.fps:
                self.skip_frame()
                time.sleep(1 / self.configuration.fps * diff)
                continue
            
            self.lastFrame = time.time()
            try:
                img, raw = self.get_frame()
                faces_found, faces = self.detect_faces_in_image(img, raw)

                if (faces_found):
                    for f in faces:
                        print("%s: %s" % (f["id"], f["dist"]))
                    self.call_vera(map(lambda f: f["id"], faces))
                        
                    #for f in filter(lambda f: float(f["dist"]) >= self.configuration.threshold, faces):
                    #    print("%s: %s" % (f["id"], f["dist"]))
            except:
                # print("Unexpected error:", sys.exc_info()[0])
                pass

    def get_frame(self):
        pass
    
    def skip_frame(self):
        pass

class JpegCamera(Camera):
    def __init__(self, globalConfiguration: GlobalConfiguration, configuration: CameraConfiguration):
        Camera.__init__(self, globalConfiguration, configuration)

    def start(self):
        self.request = urllib.request.Request(self.configuration.url)
        if self.configuration.username and self.configuration.password:
            base64string = base64.encodestring(("%s:%s" % (self.configuration.username, self.configuration.password)).encode()).decode().replace("\n", "")
            self.request.add_header("Authorization", "Basic %s" % base64string)

        super().start()

    def get_frame(self):
        try:
            stream = urllib.request.urlopen(self.request)
            return stream.read(), True
        finally:
            stream.close()

class MjpegCamera(Camera):
    def __init__(self, globalConfiguration: GlobalConfiguration, configuration: CameraConfiguration):
        Camera.__init__(self, globalConfiguration, configuration)
        self.uri = self.configuration.get_credentials_uri()

    def createVideo(self):
        self.video = cv.VideoCapture()
        if not self.video.open(self.uri):
            print("Cannot open uri %s" % self.uri)
            return False
        return True

    def start(self):
        if not self.createVideo():
            return

        super().start()
    
    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            self.video.release()
            self.createVideo()
        #else:
        #    print("Frame")

        return frame, False

    def skip_frame(self):
        self.video.grab()