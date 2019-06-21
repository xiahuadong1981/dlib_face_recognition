import face_recognition
import cv2
import numpy as np
import os
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.
# 这是从您的网络摄像头实时视频运行人脸识别的演示。它比其他例子，但它包括一些基本的性能调整，使事情运行得更快:
# 1.以1/4分辨率处理每个视频帧(但仍以全分辨率显示)
# 2.只在每一帧视频中检测人脸。
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# 请注意:这个例子要求OpenCV(“cv2”库)只能安装在网络摄像头上。
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# OpenCV不需要使用face_recognition库。只有当你想运行这个时才需要它
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
# 具体的演示。如果安装有困难，可以尝试其他不需要它的演示

# Get a reference to webcam #0 (the default one)
# 打开摄像头
video_capture = cv2.VideoCapture(0)

# 读取文件名
def basename(path):
    basename = path.split('.')[0]
    return basename
# Load a sample picture and learn how to recognize it.
# 加载一个示例图片，并学习如何识别它
aimage = "CaiLei.jpg"
obama_image = face_recognition.load_image_file(aimage)
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
bimage = 'XiaHuaDong.jpg'
biden_image = face_recognition.load_image_file('xiahuadong.jpg')
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
# 创建已知人脸编码及其名称的数组
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding]
known_face_names = [
   basename(aimage),
    basename(bimage)
]

# Initialize some variables
# 初始化一些变量
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    # 找一帧视频
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    # 将视频帧大小调整到1 / 4大小，以便更快地进行人脸识别处理
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # 将图像从BGR颜色(OpenCV使用)转换为RGB颜色(face_recognition使用)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    # 只处理每一帧视频，以节省时间
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        # 找到当前视频帧中的所有人脸和人脸编码
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # 查看该人脸是否与已知人脸匹配
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    # 显示结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        # 在脸下面画一个有名字的标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    # 显示结果图像
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()