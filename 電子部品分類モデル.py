#モジュールのインポート
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from PIL import Image
import glob
import os
from sklearn.model_selection import train_test_split
import time

np.random.seed(20190213)                                                       #実行するたびに同じ結果となるようにランダムシードを固定
tf.random.set_seed(20190213)

#データ処理定義
labels = ["cond_chip","cond_cylinder", "ic","re_chip","re_cube","transistor"]  
I = []                                                              
L = []
img_size = 50

for index, name in enumerate(labels):
    files = glob.glob(f'parts/{name}/*.jpg')
    #print(files)
    for f in files:
        img = Image.open(f).convert('RGB')
        img = img.resize((img_size, img_size))
        I.append(img)
        L.append(index)

"""print(type(I))
print(I)
print(type(L))
print(L)"""


I = np.array((I))                                                 #np.array型に変換
L = np.array(L, dtype ='uint8')                                   #uint8に変換（符号なし8ビット整数）

I = I.reshape(len(I), 2500 ,3).astype('float32') / 255 
L = tf.keras.utils.to_categorical(L, 6)

"""print(type(I))
print(I.shape)
print(type(L))
print(L.shape)"""

I_train, I_test, L_train, L_test = train_test_split(I, L)                         #デフォルトでテストデータ25％

"""print(type(I_train))
print(I_train.shape)
print(type(L_train))
print(L_train.shape)
print(type(I_test))
print(I_test.shape)
print(type(L_test))
print(L_test.shape)"""


#ニューラルネットワーク定義
model = models.Sequential()                                                    #レイヤーを入力層から順番に並べたリストを指定する。
model.add(layers.Reshape((50, 50, 3), input_shape=(50*50, 3), name='reshape'))
model.add(layers.Conv2D(8, (3, 3), padding='same',
                        kernel_initializer=initializers.TruncatedNormal(),
                        use_bias=True, activation='relu',
                        name='conv_filter1'))
model.add(layers.MaxPooling2D((2, 2), name='max_pooling1'))
model.add(layers.Conv2D(16, (3, 3), padding='same',
                        kernel_initializer=initializers.TruncatedNormal(),
                        use_bias=True, activation='relu',
                        name='conv_filter2'))
model.add(layers.MaxPooling2D((2, 2), name='max_pooling2'))
model.add(layers.Conv2D(32, (3, 3), padding='same',
                        kernel_initializer=initializers.TruncatedNormal(),
                        use_bias=True, activation='relu',
                        name='conv_filter3'))
model.add(layers.MaxPooling2D((2, 2), name='max_pooling3'))
model.add(layers.Flatten(name='flatten'))
model.add(layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.TruncatedNormal(),
                       name='hidden1'))
model.add(layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.TruncatedNormal(),
                       name='hidden1'))
model.add(layers.Dropout(rate=0.2, name='dropout'))
model.add(layers.Dense(6, activation='softmax', name='softmax'))

model.summary()                                                               #モデルの構造を確認する

#学習
model.compile(optimizer='adam',                                               #compile()の引数にそれぞれ最適化アルゴリズム、損失関数、評価関数を指定する
              loss='categorical_crossentropy',
              metrics=['acc'])

start = time.time()

history = model.fit(I_train, L_train,
                    validation_data=(I_test, L_test),
                    batch_size=32, epochs=50)

pass_time = time.time() - start
print(pass_time)

#結果表示
DataFrame({'acc': history.history['acc'],
           'val_acc': history.history['val_acc']}).plot()
plt.xticks([0, 10, 20, 30, 40 ,50])
plt.xlabel("epochs")
plt.ylabel("acc")
plt.savefig('parts正答率.png')

DataFrame({'loss': history.history['loss'],
           'val_loss': history.history['val_loss']}).plot()
plt.xticks([0, 10, 20, 30, 40 ,50])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('parts誤差関数.png')

model.save('parts.hd5', save_format='h5')
model.save_weights('parts_parameter')
