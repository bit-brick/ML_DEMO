import tensorflow as tf
from PIL import Image
import numpy as np
import os



class Predict(object):
    def __init__(self):
        print("Current working directory:", os.getcwd())
        checkpoint_dir = './'
        tflite_model_path = os.path.join(checkpoint_dir, 'mnist_model_quantized.tflite')  # 使用TFLite模型文件

        # 加载TFLite模型
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        # 获取输入输出张量的索引
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([1 - img], dtype=np.float32)  # 确保数据类型匹配

        # 设置TFLite模型的输入张量
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        # 运行预测
        self.interpreter.invoke()
        # 获取输出张量
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # 因为x只传入了一张图片，取output_data[0]即可
        # np.argmax()取得最大值的下标，即代表的数字
        print(image_path)
        print(output_data[0])
        print('        -> Predict digit', np.argmax(output_data[0]))


if __name__ == "__main__":
    app = Predict()
    app.predict('./test_images/0.png')
    app.predict('./test_images/1.png')
    app.predict('./test_images/4.png')