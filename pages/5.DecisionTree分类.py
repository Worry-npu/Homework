import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="决策树分类可视化", layout="wide")
st.title("🌳 决策树分类器")

# 文件上传
uploaded_file = st.file_uploader("请上传包含 4 个特征列和 'class' 列的 CSV 文件", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if 'class' not in data.columns or len(data.columns) < 5:
            st.error("请确保数据集中包含一个 'class' 列和四个特征列。")
        else:
            st.subheader("数据预览")
            st.dataframe(data.head())

            X = data.drop('class', axis=1)
            y = data['class']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)

            clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
            clf.fit(X_train, y_train)

            # 决策树图
            st.subheader("📊 决策树结构")
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(clf,
                      filled=True,
                      feature_names=X.columns,
                      class_names=clf.classes_,
                      rounded=True,
                      fontsize=10,
                      ax=ax)
            st.pyplot(fig)

            # 评估结果
            st.subheader("📈 模型评估")
            accuracy = clf.score(X_test, y_test)
            st.markdown(f"**测试集准确率：** `{accuracy:.2f}`")

            st.markdown("**特征重要性：**")
            importance_data = pd.DataFrame({
                "特征": X.columns,
                "重要性": clf.feature_importances_
            }).sort_values("重要性", ascending=False)
            st.dataframe(importance_data)

    except Exception as e:
        st.error(f"发生错误：{e}")
