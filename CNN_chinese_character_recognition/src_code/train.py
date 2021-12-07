from __future__ import print_function
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.models import Model, Sequential

#装载数据
data_dir = '/content/drive/My Drive/省份数据集'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

#数据信息
img_width, img_height = 64, 64
charset_size = 68
nb_validation_samples = 270
# 每次epoch验证的数量
nb_samples_per_epoch = 700
# 每次epoch训练的数据数量
nb_nb_epoch = 20;
#训练次数

def train(model):
    ##############################################
    # 图片生成器，批量生产数据防止过拟合
    # 训练数据源
    train_datagen = ImageDataGenerator( 
        rescale=1. / 255,
        # 重缩放因子 RGB系数太高，无法处理，所以用1/255进行缩放处理
        rotation_range=0.1,
        # 整数，随机旋转的度数范围
        # 随机垂直或水平平移图片的范围
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    # 测试数据只进行重缩放

    train_generator = train_datagen.flow_from_directory(
        train_data_dir, 
        #目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用.详情请查看此脚本    
        target_size=(img_width, img_height),
        # target_size: 整数tuple,默认为(256, 256). 图像将被resize成该尺寸(64,64)
        batch_size=10,
        # 一批数据的大小，用一半甚至更少的数据训练出来的梯度与用全部数据训练出来的梯度基本是一样的
        color_mode="grayscale",
        # color_mode: 颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
        class_mode='categorical')
        # class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,文件名即为标签分类
        #shuffle: 是否打乱数据,默认为True

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=10,
        color_mode="grayscale",
        class_mode='categorical')
    # 测试数据，一切同上

    #######################################################################
    model.compile(loss='categorical_crossentropy',
           optimizer='rmsprop',
           metrics=['accuracy'])
    # 编译，损失函数=交叉熵，优化器，评估函数
    model.fit_generator(train_generator,
              steps_per_epoch=nb_samples_per_epoch,
              epochs=nb_nb_epoch,
              validation_data=validation_generator,
              validation_steps=nb_validation_samples)
    #训练集；每次样本数；轮次；验证集；验证集样本数
    # 用python生成器逐批生成数据，按批次训练，提高效率


#定义网络
def build_model():
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())#压平

    # Fully connected layer
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(charset_size))

    model.add(Activation('softmax'))
    return model
    
model = build_model()
train(model)
model.save("/content/drive/My Drive/模型/model2.h5")
print("save model2.h5 successfully!")