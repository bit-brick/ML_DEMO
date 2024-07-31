import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class CNN:
    def __init__(self):
        self.model = models.Sequential([
             # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            # 第二层是最大池化层，使用2x2的池化窗口
            layers.MaxPooling2D((2, 2)),
            # 第三层是另一个卷积层，使用64个3x3的卷积核
            layers.Conv2D(64, (3, 3), activation='relu'),
            # 第四层是另一个最大池化层
            layers.MaxPooling2D((2, 2)),
            # 第五层是另一个卷积层，使用64个3x3的卷积核
            layers.Conv2D(64, (3, 3), activation='relu'),
            # 第六层是展平层，将特征图展开为一维向量
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.model.summary()

class DataSource:
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
	# 6万张训练图片，1万张测试图片 像素值映射到 0 - 1 之间
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images, self.data.train_labels,
                           epochs=5)

        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
        print(f"准确率: {test_acc:.4f}，共测试了{len(self.data.test_labels)}张图片")
        
        # 保存整个模型
        self.cnn.model.save('mnist_model.h5')
        
        # 调用转换函数
        

if __name__ == "__main__":
    app = Train()
    app.train()
    conver_to_tflite('mnist_model.h5', 'mnist_model.tflite')