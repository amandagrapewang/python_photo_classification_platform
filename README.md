# python_photo_classification_platform
一个有bug但是可以运行的平台，可能是我模型互相干扰了？
本项目包含多个用于训练图像分类模型的Python脚本和一个基于Streamlit的Web应用程序，用于展示模型的预测结果。

## 项目结构
h5文件太大了，我就没上传，还有dataset也是，可以网上自己下载，我随便找的，也可以替换一下
```bash
project/
│
├── dataset/                 # 存储数据集
│   ├── animal
│   ├── flowers
│   └── scenery
│
├── model/                 # 存储训练好的模型文件
│   ├── animal_model.h5
│   ├── flower_model.h5
│   ├── number_model.h5
│   └── scenery_model.h5
│
├── number_train.py        # 手写数字识别模型训练脚本
├── app.py                 # Streamlit应用程序，用于模型的Web展示
├── animal_train.py       # 动物分类模型训练脚本
├── flowers_train.py      # 花卉分类模型训练脚本
├── scenery_train.py      # 风景地点分类模型训练脚本
└── requirements.txt       # 项目依赖文件
```
## 模型训练
各个模型的训练脚本如下：

- `number_train.py`：训练手写数字识别模型。
- `animal_train.py`：训练动物分类模型。
- `flowers_train.py`：训练花卉分类模型。
- `scenery_train.py`：训练风景地点分类模型。

每个训练脚本都会生成一个`.h5`格式的模型文件，保存在`model/`目录下。

## 应用程序
`app.py`是一个Streamlit应用程序，它加载训练好的模型并提供一个简单的界面，用户可以上传图片或在画布上绘制，应用程序将预测并展示结果。

### 使用方法
1. 在干净的conda环境中：
   ```bash
   pip install -r requirements.txt
   ```
2. 训练模型并将`.h5`文件放置在`model/`目录下。
3. 运行`app.py`：
   ```bash
   streamlit run YOUR/DIR/TO/app.py
   ```

