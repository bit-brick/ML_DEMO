import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class CNN:
    def __init__(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.model.summary()

class DataSource:
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
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