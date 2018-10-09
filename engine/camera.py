import cv2 as cv
import urllib
import urllib.request
import numpy as np
import base64
import face_recognition
import time
import sys
import urllib.parse
import json
import datetime
from threading import Thread, Lock

class GlobalConfiguration:
    def __init__(self, vera_ip: str, threshold: float, faces_dict):
        self.threshold = threshold
        self.vera_ip = vera_ip
        self.faces_dict = faces_dict

class CameraConfiguration:
    def __init__(self, fps: float, url: str, username: str, password: str, veraDeviceId: int, webHook: str):
        self.url = url
        self.fps = fps
        self.webHook = webHook
        self.veraDeviceId = veraDeviceId
        self.username = username
        self.password = password

    def get_credentials_uri(self):
        if self.username and self.password:
            return self.url.replace("http://", "http://%s:%s@" % (urllib.request.quote(self.username), urllib.request.quote(self.password)))
        return self.url

class Camera (Thread):
    def __init__(self, globalConfiguration: GlobalConfiguration, configuration: CameraConfiguration):
        Thread.__init__(self)
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
        width = frame.shape[1]
        height = frame.shape[0]
        maxWidth = 640
        maxHeight = 480
        if width > maxWidth or height > maxHeight:
            if (width > height):
                h = int(maxWidth * height / width)
                w = maxWidth
            else:
                w = int(maxHeight * width / height)
                h = maxHeight
            small_frame = cv.resize(frame, (w, h))
        else:
            small_frame = frame

        # cv.imwrite('/home/ricciolo/Downloads/test.jpg', small_frame)

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
    
    def callback(self, ids):
        if self.configuration.veraDeviceId:
            ids = ",".join(ids)
            self.http_get("http://%s:3480/data_request?id=action&output_format=json&DeviceNum=%s&serviceId=urn:ricciolo:serviceId:PresenceSensor1&action=SetPresent&newPresentValue=%s" %(self.globalConfiguration.vera_ip, self.configuration.veraDeviceId, ids))

        if self.configuration.webHook:
            self.http_post(self.configuration.webHook, {
                "camera": self.configuration.url,
                "names": ids
            })
        
    
    def http_get(self, uri):
        self.log('Calling %s' % uri)
        request = urllib.request.Request(uri)
        try:
            stream = urllib.request.urlopen(request)
            self.log('Received %s' % stream.status)
        finally:
            stream.close()
    
    def log(self, msg):
        print("%s - %s" % (datetime.datetime.now(), msg))

    def http_post(self, uri, data):
        self.log('Calling %s' % uri)
        request = urllib.request.Request(uri)
        request.add_header('Content-Type', 'application/json; charset=utf-8')
        jsondata = json.dumps(data)
        jsondataasbytes = jsondata.encode('utf-8')
        request.add_header('Content-Length', len(jsondataasbytes))
        try:
            stream = urllib.request.urlopen(request, jsondataasbytes)
            self.log('Received %s' % stream.status)
        finally:
            stream.close()
    
    def run(self):
        while True:
            diff = time.time() - self.lastFrame
            if diff < 1 / self.configuration.fps:
                self.skip_frame(1 / self.configuration.fps * diff)
                continue
            
            self.lastFrame = time.time()
            try:
                img, raw = self.get_frame()
                faces_found, faces = self.detect_faces_in_image(img, raw)

                if (faces_found and len(faces) > 0):
                    for f in faces:
                        self.log("%s: %s" % (f["id"], f["dist"]))                    
                        
                    for f in filter(lambda f: float(f["dist"]) <= self.globalConfiguration.threshold, faces):
                        self.callback(list(map(lambda f: f["id"], faces)))
                    
                    print("")
            except:
                # print("Unexpected error:", sys.exc_info()[0])
                pass

    def get_frame(self):
        pass
    
    def skip_frame(self, seconds):
        pass

class JpegCamera(Camera):
    def __init__(self, globalConfiguration: GlobalConfiguration, configuration: CameraConfiguration):
        Camera.__init__(self, globalConfiguration, configuration)

    def run(self):
        self.request = urllib.request.Request(self.configuration.url)
        if self.configuration.username and self.configuration.password:
            base64string = base64.encodestring(("%s:%s" % (self.configuration.username, self.configuration.password)).encode()).decode().replace("\n", "")
            self.request.add_header("Authorization", "Basic %s" % base64string)

        super().run()

    def get_frame(self):
        try:
            stream = urllib.request.urlopen(self.request)
            return stream.read(), True
        finally:
            stream.close()

    def skip_frame(self, seconds):
        time.sleep(seconds)

class MjpegCamera(Camera):
    def __init__(self, globalConfiguration: GlobalConfiguration, configuration: CameraConfiguration):
        Camera.__init__(self, globalConfiguration, configuration)
        self.loop = MjpegCameraLoop(self.configuration.get_credentials_uri())

    def run(self):
        self.loop.start()
        self.loop.wait_ready()
        super().run()
    
    def get_frame(self):
        ret, frame = self.loop.get_frame()
        if not ret:
            self.video.release()
            self.createVideo()

        return frame, False

    def skip_frame(self, seconds):
        time.sleep(seconds)

class MjpegCameraLoop (Thread):    
    def __init__(self, uri):
        Thread.__init__(self)
        self.ready = Lock()
        self.ready.acquire()
        self.uri = uri

    def createVideo(self):
        self.video = cv.VideoCapture()
        if not self.video.open(self.uri):
            print("Cannot open uri %s" % self.uri)
            return False
        return True
    
    def wait_ready(self):
        self.ready.acquire()
        self.ready.release()

    def run(self):
        while True:
            if not self.createVideo():
                time.sleep(1)
                continue
            try:
                self.ready.release()
            except:
                pass

            while self.video.grab():
                pass

            self.video.release()

    def get_frame(self):
        return self.video.retrieve()