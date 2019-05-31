import keras
from keras import layers
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import Model
import glob

#location = 'C:\Users\JeongNoa\Documents\과제연구\data\Data_Extracted'

tot_list = [] 
aniname = int(input())

ani_num = {0:'공의 경계', 1: '너의 이름은', 2: '문호 스트레이 독스', 3: '바이올렛 에버가든', 4: '빈란드 사가', 
           5: '언어의 정원', 6: '원펀맨', 7: '일상', 8: '작안의 샤나', 9: '잔향의 테러', 10: '진격의 거인', 11: '페그오 X 공의 경계'}

#aniname = location + '\\' + ani_num[aniname]

aniname = 'C:\\Users\\JeongNoa\\Documents\\과제연구\\data\\Data_Extracted_2\\' + ani_num[aniname]

for foldername in glob.glob(aniname + '/*'):
    for filename in glob.glob(foldername + '/*jpg'): 
        im = Image.open(filename) 
        arr = np.array(im)
        tot_list.append(arr)

x_train = tot_list[:(len(tot_list)+1)//2]
x_test = tot_list[(len(tot_list)+1)//2:]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

print(np.shape(x_train), np.shape(x_test))

#location = 'C:\Users\JeongNoa\Documents\과제연구\data\Data_Extracted'

tot_list = [] 

ani_num = {0:'공의 경계', 1: '너의 이름은', 2: '문호 스트레이 독스', 3: '바이올렛 에버가든', 4: '빈란드 사가', 
           5: '언어의 정원', 6: '원펀맨', 7: '일상', 8: '작안의 샤나', 9: '잔향의 테러', 10: '진격의 거인', 11: '페그오 X 공의 경계'}

#aniname = location + '\\' + ani_num[aniname]

foldername = 'C:\\Users\\JeongNoa\\Documents\\과제연구\\data\\Data_Extracted_2\\일상\\편집용550 (5-30-2019 3-42-41 PM)'

for filename in glob.glob(foldername + '/*jpg'): 
    im = Image.open(filename) 
    arr = np.array(im)
    tot_list.append(arr)

x_train = tot_list[:(len(tot_list)+1)//2]
x_test = tot_list[(len(tot_list)+1)//2:]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

print(np.shape(x_train), np.shape(x_test))


img_shape = (120, 214, 3)
batch_size = 16
latent_dim = 2  # 잠재 공간의 차원: 2D 평면

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

img_shape = (120, 214, 3)
batch_size = 16

latent_dim = 2  # 잠재 공간의 차원: 2D 평면

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3*3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)



z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])



# Input에 z를 주입합니다
decoder_input = layers.Input(K.int_shape(z)[1:])

# 입력을 업샘플링합니다
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# 인코더 모델의 마지막 Flatten 층 직전의 특성 맵과 같은 크기를 가진 특성 맵으로 z의 크기를 바꿉니다
x = layers.Reshape(shape_before_flattening[1:])(x)

# Conv2DTranspose 층과 Conv2D 층을 사용해 z를 원본 입력 이미지와 같은 크기의 특성 맵으로 디코딩합니다
x = layers.Conv2DTranspose(32, 3,
                           padding='same', activation='relu',
                           strides=(2, 2))(x)
x = layers.Conv2D(1*3, 3,
                  padding='same', activation='sigmoid')(x)
# 특성 맵의 크기가 원본 입력과 같아집니다

# 디코더 모델 객체를 만듭니다
decoder = Model(decoder_input, x)

# 모델에 z를 주입하면 디코딩된 z를 출력합니다
z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # 출력 값을 사용하지 않습니다
        return x

# 입력과 디코딩된 출력으로 이 층을 호출하여 모델의 최종 출력을 얻습니다
y = CustomVariationalLayer()([input_img, z_decoded])


vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

# MNIST 숫자 이미지에서 VAE를 훈련합니다



x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,         #############################
        validation_data=(x_test, None))


import matplotlib.pyplot as plt

from scipy.stats import norm

# Display a 2D manifold of the digits
 # 15 × 15 숫자의 그리드를 출력합니다
figure = np.zeros((120 * 1, 214 * 15, 3))
# 싸이파이 ppf 함수를 사용하여 일정하게 떨어진 간격마다 잠재 변수 z의 값을 만듭니다
# 잠재 공간의 사전 확률은 가우시안입니다
grid_x = norm.ppf(np.linspace(0.05, 0.95, 1))
grid_y = norm.ppf(np.linspace(0.05, 0.95, 15))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(120, 214, 3)
        figure[i * 120: (i + 1) * 120,
               j * 214: (j + 1) * 214] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

#120, 214
