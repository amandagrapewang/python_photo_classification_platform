import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
print(tf.__version__)

# 导入数据
mnist = tf.keras.datasets.mnist
(train_data, train_target), (test_data, test_target) = mnist.load_data()

# 改变数据维度
# 改变数据维度
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
# 注：在TensorFlow中，在做卷积的时候需要把数据变成4维的格式
# 这4个维度分别是：数据数量，图片高度，图片宽度，图片通道数

# 归一化（有助于提升训练速度）
train_data = train_data / 255.0
test_data = test_data / 255.0

# 独热编码
train_target = tf.keras.utils.to_categorical(train_target, num_classes=10)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=10)  # 10种结果

# 配置早停
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# 配置GPU加速
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    # 构建更复杂的模型
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding = 'same'),
        MaxPooling2D((2, 2), padding = 'same'),
        Conv2D(64, (3, 3), activation='relu', padding = 'same'),
        MaxPooling2D((2, 2), padding = 'same'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # 配置模型检查点，保存最优模型
# checkpoint_path = "./model/number_model.h5"
# model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
#                                      save_best_only=True, save_weights_only=False, verbose=1)

# 训练模型
model.fit(train_data, train_target, epochs=5, validation_data=(test_data, test_target), callbacks=[early_stopping])

# 保存模型为 .h5 文件
model.save("./model/number_model.h5", save_format='h5')