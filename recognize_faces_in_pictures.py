import face_recognition

# Load the jpg files into numpy arrays
CaiLei_image = face_recognition.load_image_file("CaiLei.jpg")
XiaHuaDong_image = face_recognition.load_image_file("XiaHuaDong.jpg")
unknown_image = face_recognition.load_image_file("XiaHuaDong2.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    CaiLei_face_encoding = face_recognition.face_encodings(CaiLei_image)[0]
    XiaHuaDong_face_encoding = face_recognition.face_encodings(XiaHuaDong_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    CaiLei_face_encoding,
    XiaHuaDong_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of CaiLei? {}".format(results[0]))
print("Is the unknown face a picture of XiaHuaDong? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
