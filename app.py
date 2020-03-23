from flask import Flask, jsonify, escape, request, render_template
from PIL import Image
import random
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import visualize
from mrcnn.config import Config
import tensorflow as tf


from tensorflow.python.keras.backend import set_session

a = tf.ConfigProto()
a.gpu_options.allow_growth=True

sess = tf.Session(config=a)
graph = tf.get_default_graph()

app = Flask(__name__)

model = None

@app.after_request
def set_response_headers(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r

class SeverstalConfig(Config):
    GPU_COUNT = 1
    NAME = "severstal"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 4  # background + steel defects
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.6
    SAVE_BEST_ONLY = True

def load_model():
    global model
    global sess
    set_session(sess)
    severstal_config = SeverstalConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir='static/logs', config=severstal_config)
    model.load_weights("static/model/Mask_RCNN/mrcnn_0319.h5")
    print("load_model() 실행")
    

# app.route 를 사용해 매핑해준다.
# render_template -> 사용해 templates의 html 파일을 불러오겠다는 뜻
@app.route('/')
def cover():
    return render_template('cover.html')
@app.route('/main')
def main():
    return render_template('main.html')
# 철강사진을 업로드 할 메인화면
@app.route('/upload_img' , methods=['POST',"GET"])
def hello():
    try:
        if request.method =="POST":
            f= request.files['file']
            upload_file_name = 'uploaded_img' + '.' + f.filename.split('.')[-1]
            upload_file_url = "./static/img/" + 'uploaded_img' + '.' + f.filename.split('.')[-1]
            f.save(upload_file_url)

            
            image = Image.open(upload_file_url).convert('RGB')
        
            print(image)
            image.save(upload_file_url,"JPEG")
    except KeyError:
        return render_template("error.html")
    return render_template('loading.html',upload_file_name=upload_file_name)

# 철강사진 분석결과
@app.route('/predict' , methods=["POST"])
def predict():
    #그래프가 뭘까..
    # with tf.device(DEVICE):
    origin_img=skimage.io.imread('static/img/uploaded_img.jpg')
    class_names=['1','2','3','4']*30
    img = np.reshape(origin_img,(1, origin_img.shape[0], origin_img.shape[1], 3) )
    
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        results = model.detect(img, verbose=1)
        r = results[0]
        print(r)

    
    image_name = 'result_img'
    visualize.save_image(origin_img, image_name, r['rois'], r['masks'], r['class_ids'], r['scores'], class_names, mode=0)
    #class_ids 값을 중복된 result값을 하나로
    r_class = np.unique(r['class_ids'])
    print("=============================")
    print("클레스 넘버는" , r['class_ids'])
    print("=============================")
    return render_template("result.html",class_num=r_class)
 

if __name__ == '__main__':
    #global severstal_config
    load_model()
    app.run(debug=True,host="0.0.0.0")