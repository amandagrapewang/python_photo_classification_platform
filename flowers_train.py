import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, RandomFlip, RandomRotation, RandomZoom, Attention
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.utils import class_weight


# 数据路径
data_dir = 'dataset/flowers'  # 数据集根目录
batch_size = 32

# 图片生成器，用于从文件夹加载图片数据
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(180, 180),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(180, 180),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 计算样本权重
class_weights = class_weight.compute_sample_weight('balanced', generator.classes)

# 配置早停
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# # 数据增强
# data_augmentation = tf.keras.Sequential(
#     [
#         RandomFlip("horizontal", input_shape=(180, 180, 3)),
#         RandomRotation(0.1),
#         RandomZoom(0.1),
#     ]
# )

# 配置GPU加速
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = tf.keras.Sequential([
        # data_augmentation,
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(180, 180, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(generator.class_indices), activation='softmax')
    ])

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # 配置模型检查点，保存最优模型
# checkpoint_path = "./model/animal_model.h5"
# model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
#                                      save_best_only=True, save_weights_only=False, verbose=1)

# 训练模型
model.fit(generator, epochs=20, validation_data=validation_generator, callbacks=[early_stopping])

# 保存模型为 .h5 文件
model.save("./model/flower_model.h5", save_format='h5')
