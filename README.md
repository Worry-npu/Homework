# 第二次信息安全作业-数据挖掘入侵检测软件使用手册

## 1. 软件简介与使用方法

本软件是一个基于数据挖掘算法的入侵检测系统，主要通过前端可视化界面展示不同的机器学习算法，并且提供多种数据集与算法模型的测试和验证功能。用户可以通过交互式操作选择不同的算法，上传数据集并查看模型训练、测试和评估的结果。

### 主要功能：

- 支持 20 种常见的数据挖掘算法，包括分类、回归、聚类等算法。
- 支持数据上传、预处理、模型训练、预测和结果可视化。
- 每个算法都配有图形界面，便于用户操作和查看结果。
- 实时展示算法的性能指标（如精度、召回率、F1 值等）。

### 如何使用：

1. **运行程序：**

   - 在本地安装好 Python 环境并确保安装了所有依赖库（详见下文的依赖安装部分）。
   - 打开命令行窗口，进入程序目录。
   - 运行以下命令启动程序：
     ```bash
     streamlit run app.py
     ```
   - 程序启动后会自动打开浏览器，展示前端界面。
   - 你可以选择不同的页面和算法进行操作。

2. **上传数据：**

   - 在前端界面，点击“上传数据”按钮，选择你的数据集（支持 CSV、Excel 等格式）。
   - 上传后，你可以选择需要应用的算法进行训练。

3. **选择算法并查看结果：**

   - 在前端界面选择你想使用的算法。
   - 点击“开始训练”按钮，模型会自动训练并显示训练结果。
   - 你可以查看模型的性能指标（如精度、召回率等）。

---

## 2. Streamlit 前端介绍

**Streamlit** 是一个快速构建数据应用的工具，能够通过简单的 Python 代码实现强大的前端界面。我们使用 Streamlit 作为前端框架，提供一个交互式的用户界面，方便用户进行数据上传、模型选择、参数设置等操作。

### 安装 Streamlit：

```bash
pip install streamlit
```

安装后可使用以下命令检查是否成功安装：

```bash
streamlit --version
```

### 启动程序：

确保你已安装了所有必要的依赖库（可以通过 `requirements.txt` 一键安装）：

```bash
pip install -r requirements.txt
```

启动 Streamlit 应用：

```bash
streamlit run app.py
```

### 前端界面功能：

- **数据上传界面：** 允许用户上传数据文件。
- **算法选择界面：** 用户可以选择要应用的算法。
- **结果展示界面：** 展示模型训练后的结果，并提供性能指标的可视化图表。

---

## 3. 算法介绍

本软件支持以下 20 种数据挖掘算法，涵盖了分类、回归和聚类等多种类型。

### 3.1 分类算法

- **Logistic Regression：** 一种广泛使用的线性分类算法，用于二分类问题。
- **K-Nearest Neighbors (KNN)：** 基于距离度量的分类算法，用于多类分类。
- **Decision Tree：** 通过决策树结构进行分类，简单易懂，广泛应用。
- **Random Forest：** 一种集成学习方法，通过多棵决策树进行分类。
- **Support Vector Machine (SVM)：** 用于二分类问题，基于最大间隔的超平面。
- **Naive Bayes：** 基于贝叶斯定理的分类方法，适合高维数据。
- **Gradient Boosting：** 通过加法模型构建强分类器，通过 boosting 方法逐步改进模型。
- **XGBoost：** 基于梯度提升算法的优化版本，广泛用于竞赛中。

### 3.2 回归算法

- **Linear Regression：** 线性回归，用于预测数值型数据。
- **Lasso Regression：** 在线性回归基础上加入 L1 正则化，适合特征选择。
- **Ridge Regression：** 在线性回归基础上加入 L2 正则化，用于避免过拟合。
- **Decision Tree Regressor：** 用于回归问题的决策树算法。
- **Random Forest Regressor：** 通过集成多个决策树进行回归预测。
- **Gradient Boosting Regressor：** 梯度提升回归模型，适用于复杂的回归任务。

### 3.3 聚类算法

- **K-Means：** 经典的聚类算法，将数据分成 K 个簇。
- **Agglomerative Clustering：** 层次聚类算法，通过逐步合并簇来生成聚类结果。
- **DBSCAN：** 基于密度的聚类方法，能够处理噪声和不规则形状的簇。
- **Mean Shift：** 基于密度的聚类算法，能够自动选择聚类数目。

### 3.4 异常检测

- **Isolation Forest：** 基于决策树的异常检测算法，适用于大规模数据集。

### 3.5 降维算法

- **Principal Component Analysis (PCA)：** 主成分分析，常用于数据降维和可视化。

---

## 4. 如何将算法与前端连接

我们通过将机器学习模型与 Streamlit 前端界面结合，实现了交互式的数据分析和可视化。在后端，我们使用 **Scikit-learn** 实现各类机器学习算法，并将结果返回给前端界面展示。

### 实现步骤：

#### 前端与后端交互：

- 用户通过前端界面选择算法、上传数据、设置参数等。
- 用户点击“开始训练”按钮时，前端会调用后端函数进行模型训练或预测。

#### 后端处理：

- 后端使用 Scikit-learn、XGBoost、LightGBM 等库进行训练与预测。
- 模型训练完成后将结果通过 Streamlit 展示，包括性能评估指标和可视化图表。

### 示例代码：

```python
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加载数据
data = load_iris()
X = data.data
y = data.target

# 创建模型
model = RandomForestClassifier(n_estimators=100)

# 前端界面
st.title('入侵检测软件')
algorithm = st.selectbox('选择算法', ['Random Forest', 'KNN', 'SVM'])

# 训练与展示
if st.button('开始训练'):
    if algorithm == 'Random Forest':
        model.fit(X, y)
        st.write("模型训练完成")
```

- 使用 `st.write()` 展示训练结果与模型指标。
- 使用 `st.pyplot()` 展示图表（如混淆矩阵、ROC 曲线等）。

---

## 5. 在线使用链接

你也可以无需安装环境，直接在线访问使用本软件：

👉https://homework-wyr.streamlit.app/

如需进一步开发或定制此系统，可基于本架构自由扩展新算法或模块。

如需技术支持或反馈建议，请联系项目开发者。
