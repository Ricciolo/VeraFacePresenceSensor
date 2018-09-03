from os import listdir, environ
from os.path import isfile, join, splitext

import face_recognition

import yaml
from camera import Camera

# Global storage for images
faces_dict = {}

# <Picture functions> #


def is_picture(filename):    
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions


def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]


def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]

def calc_face_encoding(image):
    # Currently only use first face found on picture
    loaded_image = face_recognition.load_image_file(image)
    faces = face_recognition.face_encodings(loaded_image)

    # If more than one face on the given image was found -> error
    if len(faces) > 1:
        raise Exception(
            "Found more than one face in the given training image.")

    # If none face on the given image was found -> error
    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]


def get_faces_dict(path):
    image_files = get_all_picture_files(path)
    return dict([(remove_file_ext(image), calc_face_encoding(image))
                 for image in image_files])   

if __name__ == "__main__":

    config_file = environ.get("CONFIG_PATH", "config.yaml")
    print("Opening config " + config_file + "...")
    with open(config_file, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print("Cannot parse " + config_file + " file")
            exit(-1)

    print("Processing faces...")
    # Calculate known faces    
    dir = environ.get("FACES_PATH", "faces")
    faces_dict = get_faces_dict(dir)    

    if len(faces_dict) == 0:
        print("No faces found")
        exit(-2)
    else:
        print("Found %s faces" % len(faces_dict))

    threshold = float(config.get("threshold", 0.05))

    # Start cameras
    cameras = []
    for c in config.get("cameras", []):
        c = Camera(threshold, faces_dict, c["url"], c.get("username"), c.get("password"))
        cameras.append(c)    

        c.start_loop()