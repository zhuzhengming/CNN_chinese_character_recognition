# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:48:57 2020

@author: Zz
"""

from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

char_dict= ['上海','上海市','云南','云南省','内蒙古','内蒙古自治区','北京','北京市','台湾','台湾省','吉林','吉林省','四川','四川省','天津','天津市','宁夏','宁夏回族自治区','安徽','安徽省',
            '山东','山东省','山西','山西省','广东','广东省','广西','广西壮族自治区','新疆','新疆维吾尔自治区','江苏','江苏省','江西','江西省','河北','河北省','河南','河南省',
            '浙江','浙江省','海南','海南省','湖北','湖北省','湖南','湖南省','澳门','澳门特别行政区','甘肃','甘肃省','福建','福建省','西藏','西藏自治区','贵州','贵州省','辽宁','辽宁省',
            '重庆','重庆市','陕西','陕西省','青海','青海省','香港','香港特别行政区','黑龙江','黑龙江省']


#加载模型
model = load_model("/content/drive/My Drive/模型/model2.h5")  
#输出模型信息 
#model.summary()

#生成测试数据流
path="/content/drive/My Drive/省份数据集/验证2"
test_datagen = ImageDataGenerator(rescale=1./255)
    # 测试数据只进行重缩放
img = test_datagen.flow_from_directory(
        path,
        target_size=(img_width, img_height),
        batch_size=3,
        color_mode="grayscale",
        shuffle = False,#不打乱顺序
        class_mode=None)#无标签生成


#模型预测
pre=model.predict(img)
#最大概率对应的标签下标
result = np.argmax(pre,axis=1)

print(result)

print(char_dict[result[0]],end='\n')
print(char_dict[result[1]],end='\n')
print(char_dict[result[2]],end='\n')
print("successful")
