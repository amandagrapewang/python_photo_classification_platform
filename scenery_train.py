import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping

# 数据路径
train_data_dir = 'dataset/scenery/seg_train'
test_data_dir = 'dataset/scenery/seg_test'
batch_size = 32

# 图片生成器
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(180, 180),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(180, 180),
    batch_size=batch_size,
    class_mode='categorical'
)

# 配置GPU加速
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 配置早停
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

with strategy.scope():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # 配置模型检查点，保存最优模型
# checkpoint_path = "./model/animal_model.h5"
# model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
#                                      save_best_only=True, save_weights_only=False, verbose=1)

# 训练模型
model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[early_stopping])

# 保存模型为 .h5 文件
model.save("./model/scenery_model.h5", save_format='h5')


#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 自定义回调函数，保存模型为 .h5 格式
# class CustomModelCheckpoint(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         self.model.save("./model/scenery_model.h5")
#
# # 创建自定义回调函数实例
# custom_checkpoint = CustomModelCheckpoint()
#
# # 训练模型时使用自定义回调函数
# model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[custom_checkpoint])
