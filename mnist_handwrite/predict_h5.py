import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

from train import CNN




class Predict(object):
    def __init__(self):
        print("Current working directory:", os.getcwd())
        checkpoint_dir = './'
        # 由于使用HDF5格式的权重文件，直接指定文件路径
        latest = os.path.join(checkpoint_dir, 'mnist_model.h5')  # 使用最新的权重文件

        self.cnn = CNN()
        # 直接使用指定的文件路径作为 load_weights 的参数
        self.cnn.model= load_model('mnist_model.h5')

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([1 - img])

        # API refer: https://keras.io/models/model/
        y = self.cnn.model.predict(x)

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(y[0])
        print('        -> Predict digit', np.argmax(y[0]))


if __name__ == "__main__":
    app = Predict()
    app.predict('./test_images/0.png')
    app.predict('./test_images/1.png')
    app.predict('./test_images/4.png')
