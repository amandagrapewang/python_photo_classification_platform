import sys
import streamlit as st
from PIL import Image
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

# 创建两个列，每个列可以放置不同的内容
col1, col2 = st.columns(2)

# 在第一个列中放置内容
with col1:
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
                # st.write(f"predictions：{predictions}")
                predicted_class_index = np.argmax(predictions)
                # st.write(f"predicted_class_index：{np.argmax(predictions)}")
                predicted_class = num_class_labels[predicted_class_index]
                # st.write(f"predicted_class：{num_class_labels[predicted_class_index]}")

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




# 在第二个列中放置内容
with col2:
    models = {
        "动物类别判断": tensorflow.keras.models.load_model(os.path.join(BASE_DIR, "model", "animal_model.h5"), compile=True),
        "花卉类别判断": tensorflow.keras.models.load_model(os.path.join(BASE_DIR, "model", "flower_model.h5"), compile=True),
        "风景地点判断": tensorflow.keras.models.load_model(os.path.join(BASE_DIR, "model", "scenery_model.h5"), compile=True),
    }

    know_advice = ["动物类别判断", "花卉类别判断"]

    def generate_report(selected_model, predicted_class, advice, image_data, predicted_probabilities, class_labels):
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')  # 使用提供的图像数据打开图像
        except Exception as e:
            st.error("处理图片时出现问题，请确认图片格式和数据。")
            st.error(f"错误信息: {e}")
            return

        # 将图片转换为Base64编码
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 构建HTML代码来显示图片
        img_html = f'<img src="data:image/png;base64,{img_str}" alt="Uploaded Image">'

        # 构建预测结果和概率信息
        predictions_info = ""
        for i, prob in enumerate(predicted_probabilities):
            predictions_info += f"{class_labels[i]}: {prob:.2f}%<br>\n"

        # 构建报告内容
        advice_content = ""
        for item in advice:
            advice_content += f"{item}<br>\n"

        report_content = f"""<html>
                                <head>
                                    <style>
                                        body {{
                                            font-family: Arial, sans-serif;
                                            text-align: center; /* 文本居中显示 */
                                            background-color: #f0f0f0; /* 背景色 */
                                        }}
                                        h1 {{
                                            color: #333;
                                        }}
                                        table {{
                                            margin: auto; /* 表格居中显示 */
                                            border-collapse: collapse;
                                            width: 80%; /* 表格宽度 */
                                            background-color: #fff; /* 表格背景色 */
                                            padding: 20px;
                                        }}
                                        th, td {{
                                            border: 1px solid #ccc;
                                            padding: 10px;
                                        }}
                                    </style>
                                </head>
                                <body>
                                    <h1>图像分类分析报告</h1>
                                    <table>
                                        <tr>
                                            <th>项目</th>
                                            <th>内容</th>
                                        </tr>
                                        <tr>
                                            <td>选择的模型分类</td>
                                            <td>{selected_model}</td>
                                        </tr>
                                        <tr>
                                            <td>预测结果</td>
                                            <td>{predicted_class}</td>
                                        </tr>
                                        <tr>
                                            <td>预测概率</td>
                                            <td>
                                                {predictions_info}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>简单介绍</td>
                                            <td>
                                                {advice_content}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>上传的图片</td>
                                            <td>
                                                {img_html}
                                            </td>
                                        </tr>
                                    </table>
                                    <h3>以上结果仅供参考</h3>
                                </body>
                                </html>"""

        report_filename = f"{selected_model}_diagnosis_report.html"

        with open(report_filename, "w") as file:
            file.write(report_content)

        st.success("报告生成成功！")

        # 提供下载链接
        with open(report_filename, "rb") as file:
            report_data = file.read()
            b64 = base64.b64encode(report_data).decode()
            href = f'<a href="data:file/html;base64,{b64}" download="{report_filename}">点击这里下载报告</a>'
            st.markdown(href, unsafe_allow_html=True)


    st.header("选择模型分类并提供图片进行判断")

    selected_model = st.selectbox("选择模型分类", list(models.keys()))

    uploaded_image = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_data = uploaded_image.read()
        image = Image.open(BytesIO(image_data))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = models[selected_model]

        input_shape = model.input_shape[1:3]
        image = image.resize(input_shape)

        image_array = np.array(image)
        image_array = image_array / 255.0

        if len(model.input_shape) == 4:
            image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        class_labels = []

        if selected_model == "动物类别判断":
            class_labels = ["蝴蝶", "猫", "鸡", "牛", "狗", "大象", "马", "羊", "蜘蛛", "松鼠"]
            advice_dict = {
                "蝴蝶": ["蝴蝶是昆虫中的一种，属于鳞翅目。",
                         "它们的特点是身体细长，有两对薄而有色的翅膀。",
                         "蝴蝶通常在花朵周围飞舞，以花蜜为食。",
                         "它们在生命周期中经历幼虫、蛹和成虫三个阶段，是自然界中美丽而独特的生物之一。"],
                "猫": ["猫是家猫的通称，是一种家畜动物，属于哺乳动物。",
                       "猫有着柔软的毛皮和灵活的身体，以及锋利的爪子。",
                       "它们是人类最早驯养的动物之一，广泛分布于世界各地。",
                       "猫通常以捕捉小型啮齿动物和鸟类为生，是人类常见的宠物之一"],
                "鸡": ["鸡是一种家禽，常见于全世界各地。",
                       "它们是人类最早驯养的动物之一，主要被养殖用于食用和产蛋。",
                       "鸡的特征包括具有羽毛的身体、喙和爪子。",
                       "除了食用肉和蛋外，鸡的叫声也是农村常见的声音之一。"],
                "牛": ["牛是哺乳动物，属于偶蹄目。",
                       "它们被人类驯养用于提供肉、奶、皮革等各种用途。",
                       "牛的特征包括强壮的身体、角、四蹄和长长的尾巴。",
                       "在许多文化中，牛被视为重要的家畜，承载着农业和经济上的重要角色。"],
                "狗": ["狗是人类最早驯养的动物之一，属于哺乳动物。",
                       "它们有着各种不同的品种和体型，从小型犬到大型犬不等。",
                       "狗通常被养作宠物，也被用于警戒、搜救、导盲等工作。",
                       "它们以其忠诚、友好和忠诚的品质而受到人类的喜爱。"],
                "大象": ["大象是世界上最大的陆地动物之一，属于哺乳动物。",
                         "它们有着庞大的身躯、长长的象牙和宽大的耳朵。",
                         "大象通常生活在非洲和亚洲的草原、森林和沙漠地带。",
                         "它们是社会性动物，以群体为单位生活，拥有复杂的社会结构和交流方式。"],
                "马": ["马是一种家畜动物，属于哺乳动物。",
                       "它们有着优雅的体态、强壮的四肢和长长的尾巴。",
                       "马被广泛用于运输、农业、体育等各种用途。",
                       "它们以其速度、力量和耐力而闻名，是人类历史上重要的伙伴之一。"],
                "羊": ["羊是一种常见的家畜动物，属于哺乳动物。",
                       "它们有着蓬松的毛皮和弯曲的角。",
                       "羊通常被人类养殖用于提供羊毛、羊肉、羊奶等产品。",
                       "它们是社会性动物，以群体为单位生活，常常在草原和山区地带放牧。"],
                "蜘蛛": ["蜘蛛是一种节肢动物，属于蜘蛛纲。",
                         "它们有着八只长腿和分节的身体。",
                         "蜘蛛通常以捕食昆虫为生，利用自己编织的网来捕捉猎物。",
                         "它们生活在各种环境中，从森林到城市都有发现。"],
                "松鼠": ["松鼠是一种啮齿动物，属于松鼠科。",
                         "它们有着灵活的身体和长长的尾巴。",
                         "松鼠通常生活在树上，以坚果、种子和水果为食。",
                         "它们以其活泼好动和敏捷的特点而闻名，是许多人心目中的可爱动物之一。"]
            }

        elif selected_model == "花卉类别判断":
            class_labels = ["洋甘菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]
            advice_dict = {
                "洋甘菊": ["洋甘菊是一种常见的花卉，具有淡蓝色或白色的花瓣，中间是黄色的花蕊。",
                           "它们被广泛种植作为园艺植物，并且在医药和美容行业中也很受欢迎。",
                           "洋甘菊被用于制作茶和精油，具有舒缓和放松的效果。",
                           "在花语中，洋甘菊通常象征着友谊、温和和平静。"],
                "蒲公英": ["蒲公英是一种常见的野生植物，有着带有细小白丝的黄色花朵，成熟后会变成风吹就会飞散的种子。",
                           "它们生长在各种环境中，包括草地、道路边缘和田野。",
                           "蒲公英在草地上常被认为是杂草，但它们也被一些人视为美丽而坚韧的植物。",
                           "在花语中，蒲公英代表着希望、自由和幸福。"],
                "玫瑰": ["玫瑰是最受欢迎和广泛种植的花之一，有成百上千种不同的品种，颜色和形状各异。",
                         "玫瑰被视为爱情和美丽的象征，是情人节和其他浪漫场合的常见礼物。",
                         "除了作为美丽的花束和花环，玫瑰也被用来提取精油，用于香水和护肤品。",
                         "在花语中，不同颜色的玫瑰代表着不同的情感，例如红色代表热情和爱情，白色代表纯洁和无辜。"],
                "向日葵": ["向日葵是一种高大的开花植物，以其大而明亮的黄色花朵和特殊的生长习性而闻名。",
                           "它们倾向于朝向太阳，并在一天中跟随太阳的运动而转动，因此得名。",
                           "向日葵象征着阳光、活力和希望，在许多文化中被视为吉祥物。",
                           "它们也是一种重要的农业作物，提供了食用油和饲料。"],
                "郁金香": ["郁金香是一种多年生草本植物，有着各种各样的颜色和花型，因此在园艺上受到欢迎。",
                           "它们在花园、花坛和花瓶中都很常见。",
                           "郁金香在荷兰尤其著名，被认为是该国的象征之一。",
                           "花语中，郁金香通常代表着爱情、优雅和温柔。"]
            }

        elif selected_model == "风景地点判断":
            class_labels = ["建筑物", "森林", "冰川", "山", "海", "街道"]

        predictions = model.predict(image_array)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        # 获取预测的概率值
        predicted_probabilities = predictions * 100

        st.write(f"对于选择的模型分类 '{selected_model}' 的预测结果是：")
        st.write(f"类别：'{predicted_class}' 概率：{predicted_probabilities[predicted_class_index]:.2f}")

        if selected_model in know_advice:
            if predicted_class in advice_dict:
                advice = advice_dict[predicted_class]
                st.write("建议：")
                for item in advice:
                    st.write(f"- {item}")

            advice = advice_dict[predicted_class]

            if st.button('生成定制化报告'):
                with st.spinner('正在生成报告...'):
                    generate_report(selected_model, predicted_class, advice, image_data, predicted_probabilities,
                                    class_labels)