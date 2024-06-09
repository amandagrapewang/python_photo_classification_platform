import sys
import streamlit as st
from PIL import Image, ImageOps
import tensorflow
import numpy as np
import base64
from io import BytesIO
import joblib
import os
from streamlit_drawable_canvas import st_canvas


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 页面布局
st.set_page_config(page_title="图像分类平台", page_icon="🔬", layout="wide")

# 页面布局
st.title('图像分类平台')
st.header('手写数字识别')
# 创建画布
canvas = st_canvas(
    fill_color="#FFFFFF",  # 画布背景色
    stroke_color="#000000",  # 笔触颜色
    height=300,  # 画布高度
    width=300,  # 画布宽度
    drawing_mode="freedraw",  # 绘制模式
    key='canvas'
)

# 添加提交按钮
user_drew = st.button("提交并预测数字")

# 加载模型
model_path = os.path.join(BASE_DIR, "model/number_model.h5")
if os.path.isfile(model_path):
    try:
        num_model = tensorflow.keras.models.load_model(model_path, compile=True)
    except Exception as e:
        st.error(f"加载模型时发生错误: {e}")
        num_model = None
else:
    st.error(f"模型文件不存在: {model_path}")

num_flag = 1

# 执行预测
if user_drew:
    if canvas is not None and canvas.image_data is not None:
        try:
            # 将 NumPy 数组转换为 PIL 图像
            image = canvas.image_data
            # 检查 canvas.image_data 是否是有效的图像数据
            print("Canvas image data shape:", canvas.image_data.shape)
            print("Canvas image data dtype:", canvas.image_data.dtype)

            if canvas.image_data.shape[-1] == 4:
                my_image = canvas.image_data[..., 3:]
            else:
                my_image = canvas.image_data

            # 显示用户绘制的图像
            st.image(my_image, caption='您绘制的数字')

            print("my_image shape:", my_image.shape)

            # 创建一个新的 PIL 图像，模式设置为 'L'（灰度）
            pil_image = Image.new('L', (my_image.shape[1], my_image.shape[0]))

            # 将 my_image 的数据复制到 PIL 图像中
            pil_image.putdata(my_image.reshape(-1))

            image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)  # 调整大小

            st.image(image, caption='调整大小后')

            # 归一化图像数据
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)  # 添加通道维度
            image_array = np.expand_dims(image_array, axis=0)  # 添加批次维度

            # 显示用户绘制的图像
            st.image(image_array[0, :, :, 0], caption='处理后的图像')

            # 打印图像数组的形状和数据类型
            print("Image array shape:", image_array.shape)
            print("Image array dtype:", image_array.dtype)

            # 打印最小和最大像素值
            print("Min pixel value:", image_array.min())
            print("Max pixel value:", image_array.max())

            # 使用模型进行预测
            num_class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            predictions = num_model.predict(image_array)[0]
            st.write(f"predictions：{predictions}")
            predicted_class_index = np.argmax(predictions)
            st.write(f"predicted_class_index：{np.argmax(predictions)}")
            predicted_class = num_class_labels[predicted_class_index]
            st.write(f"predicted_class：{num_class_labels[predicted_class_index]}")

            # 获取预测的概率值
            predicted_probabilities = predictions * 100

            st.write(f"对于您绘制的数字的预测结果是：")
            st.write(f"类别：'{predicted_class}' 概率：{predicted_probabilities[predicted_class_index]:.2f}")

        except Exception as e:
            # 显示错误信息
            st.error("图像处理出错")
            st.exception(e)
            num_flag = 0
    else:
        st.warning("没有检测到图像数据。请在画布上绘制数字。")
        num_flag = 0