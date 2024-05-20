from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Definindo caminhos absolutos
train_data_validation = os.path.abspath('data/train')
validation_data_validation = os.path.abspath('data/validation')

# Verifique e crie os diretórios se não existirem
os.makedirs(train_data_validation, exist_ok=True)
os.makedirs(validation_data_validation, exist_ok=True)

# Crie subdiretórios de classes se necessário
class_names = ['class1', 'class2']  # Altere conforme suas classes
for class_name in class_names:
    os.makedirs(os.path.join(train_data_validation, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_data_validation, class_name), exist_ok=True)

print("Diretórios criados com sucesso")

# Definindo o tamanho das imagens
img_width, img_height = 400, 400

# Quantidade de amostras
nb_train_samples = 8
nb_validation_samples = 8
epochs = 80
batch_size = 2

# Definindo a forma de entrada dependendo do formato de dados da imagem
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Geradores de dados
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_validation,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_validation,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Criando o modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.summary()

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)
model.save_weights('assoreamento.h5')

# Predição com uma imagem de exemplo
img_pred = image.load_img('anyImageSample.jpg', target_size=(400, 400))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

# Resultado da validação da imagem
rslt = model.predict(img_pred)
print(rslt)
if rslt[0][0] == 1:
    prediction = "assoreamento"
else:
    prediction = "naoEhAssoreamento"
print(prediction)