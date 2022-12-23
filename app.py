import random
from datetime import datetime
import numpy as np
import os
import cv2
import mindspore
from flask import Flask, request, jsonify, render_template, redirect, url_for
from model import AlexNet
from mindspore.train.serialization import load_param_into_net, load_checkpoint
import config
from flask_bootstrap import Bootstrap


app = Flask(__name__)
app.config.from_object(config)
bootstrap = Bootstrap(app)
baseDir =os.path.abspath(os.path.dirname(__file__))

model = AlexNet()
param_dict = load_checkpoint("checkpoint_alexnet-5_1875.ckpt")
load_param_into_net(model, param_dict)



@app.route('/upload/', methods=['GET','POST'])
def upload_test():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        f = request.files['photo']
        random_num = random.randint(0, 100)
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(random_num) + "." + f.filename.rsplit('.', 1)[1]
        filepath = baseDir + "/static/user_inputData/" + filename
        f.save(filepath)

        my_host = "http://127.0.0.1:5000"
        new_path_file = my_host + "/static/user_inputData/"+filename
        data = {"msg":"success", "url":new_path_file}
        return redirect(url_for('run_inference', pic_name=filename))
    # return render_template('index.html')

@app.route('/run_inference/?<string:pic_name>')
def run_inference(pic_name):
    filepath = baseDir + "/static/user_inputData/" + pic_name
    img = img_preprocessing(filepath)
    input = mindspore.Tensor(img, dtype=mindspore.float32)
    out_tensor = model(input).squeeze(0)
    max_prob = out_tensor.max().asnumpy().item()
    max_prob_index = out_tensor.argmax().asnumpy().item()

    return render_template('run_inference.html', max_prob_index=max_prob_index, max_prob=max_prob, filepath="/static/user_inputData/" + pic_name)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('base.html')

def img_preprocessing(filepath):
    img = cv2.imread(filepath, 0)
    img = cv2.resize(img, (32, 32))
    img = np.array(img).reshape((1, 1, 32, 32))

    # cv2.namedWindow('result', 0)
    # cv2.imshow('result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("result_" + pic_name, img)
    return img



if __name__ == '__main__':
    # app.run()
    bootstrap.run()
