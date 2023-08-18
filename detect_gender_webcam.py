from keras_preprocessing.image import img_to_array
from keras.models import load_model
from flask import Flask, render_template, Response, request
import numpy as np
import cv2
import cvlib as cv
import os.path
import os
from random import random
                    
# load model
model = load_model('my_model')

# open webcam
webcam = cv2.VideoCapture(0)

classes = ['Woman','Man']
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        try:
            img = request.files['file']
            if img:
                # Save File just upload
                path_for_save = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
                print("Save = ", path_for_save)
                img.save(path_for_save)
                img_read = cv2.imread(path_for_save)
                width = 320
                height = 320 
                dim = (width, height)
                # resize image
                img_read = cv2.resize(img_read, dim, interpolation = cv2.INTER_AREA)
                face, confidence = cv.detect_face(img_read)
                for idx, f in enumerate(face):
                    # get corner points of face rectangle        
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]

                # draw rectangle over face
                    cv2.rectangle(img_read, (startX,startY), (endX,endY), (0,255,0), 2)

                # crop the detected face region
                    face_crop = np.copy(img_read[startY:endY,startX:endX])

                    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (96,96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                idx = np.argmax(conf)
                label = classes[idx]
                label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img_read,label,(startX, Y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
                cv2.imwrite(path_for_save, img_read)

                # Trả về kết quả
                return render_template("indexx.html", user_image = img.filename , rand = str(random()),
                                           msg="UpLoad Successfull!!")
            else:
                return render_template('indexx.html', msg='Select File For Upload')
        except Exception as ex:
            # If Error
            print(ex)
            return render_template('indexx.html', msg='Cannot Recognize !!')
    else:
        # If is GET -> show UI upload
        return render_template('indexx.html')
def gen_frames():
    while (True):
    # read frame from webcam 
        success, frame = webcam.read()
        if not success:
            break
        else:
            # apply face detection
            face, confidence = cv.detect_face(frame)
            # loop through detected faces
            for idx, f in enumerate(face):
                # get corner points of face rectangle        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY,startX:endX])

                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (96,96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]
                label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                # write label and confidence above face rectangle
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def index():
    return render_template('indexx.html')
@app.route('/video_feed')
def video_feed():
     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)