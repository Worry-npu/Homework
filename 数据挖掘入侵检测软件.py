import streamlit as st
from pathlib import Path

st.set_page_config(page_title="数据挖掘算法原型系统", layout="wide")

st.title("💡 数据挖掘算法原型系统")

st.markdown("""
<style>
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 30px;
}
.block {
    background-color: #f0f2f6;
    padding: 16px;
    border-radius: 10px;
    margin-top: 10px;
    margin-bottom: 20px;
}
.download-button a {
    text-decoration: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 🚀 系统简介
st.markdown("<div class='section-title'>🚀 系统简介</div>", unsafe_allow_html=True)
st.markdown("""
<div class='block'>
本系统为一个数据挖掘算法原型平台，支持从聚类、分类、回归、降维、异常检测到关联规则等 **20 种经典算法** 的可视化交互分析。
适用于教学实验、算法原型验证、快速探索数据关系等场景。
</div>
""", unsafe_allow_html=True)

# 📘 使用手册
st.markdown("<div class='section-title'>📘 使用手册</div>", unsafe_allow_html=True)
manual_path = Path("static/用户手册.pdf")
if manual_path.exists():
    st.markdown(f"""
    <div class='block'>
        👉 <span class="download-button"><a href="{manual_path.as_posix()}" target="_blank">点击查看 / 下载用户手册（PDF）</a></span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("用户手册文件未找到，请将 `用户手册.pdf` 放入项目的 `static/` 目录中。")

# 👨‍🔬 权属信息
st.markdown("<div class='section-title'>👨‍🔬 权属信息</div>", unsafe_allow_html=True)
st.markdown("""
<div class='block'>
本系统由以下成员共同开发与持有产权：  
王杨瑞、唐韧、胥庚炜、刘思宇、潘岷阳、周佳仪、王淘、仇昱博、陈菁菁、王昱焜。
</div>
""", unsafe_allow_html=True)

# 🧠 功能模块一览
st.markdown("<div class='section-title'>🧠 支持功能模块一览</div>", unsafe_allow_html=True)
st.markdown("""
<div class='block'>
- 1️⃣ KMeans 聚类  - 2️⃣ DBSCAN 聚类  - 3️⃣ 层次聚类  - 4️⃣ MeanShift  
- 5️⃣ 决策树分类  - 6️⃣ SVM 分类  - 7️⃣ 随机森林分类  - 8️⃣ KNN 分类  
- 9️⃣ 逻辑回归  - 🔟 朴素贝叶斯  - 11️⃣ 线性回归  - 12️⃣ 决策树回归  
- 13️⃣ 随机森林回归  - 14️⃣ XGBoost 回归  - 15️⃣ PCA 降维  
- 16️⃣ t-SNE 降维  - 17️⃣ AutoEncoder 降维  - 18️⃣ 孤立森林  
- 19️⃣ Apriori 关联规则  - 20️⃣ FP-Growth 关联规则
</div>
""", unsafe_allow_html=True)
