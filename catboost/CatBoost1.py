import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# 读取训练集和验证集数据
train_file_path = r"./train_data.xlsx"
test_file_path = r"./test_data.xlsx"

train_df = pd.read_excel(train_file_path)
test_df = pd.read_excel(test_file_path)

# 分离自变量和因变量
x_train = train_df.iloc[:, 1:]  # 特征列
y_train = train_df.iloc[:, 0]  # 标签列
x_train_names = x_train.columns.tolist()

x_test = test_df.iloc[:, 1:]  # 特征列
y_test = test_df.iloc[:, 0]  # 标签列
x_test_names = x_test.columns.tolist()

# 转换 NumPy 数组为 Pandas DataFrame
x_train_df = pd.DataFrame(x_train, columns=x_train_names)
x_test_df = pd.DataFrame(x_test, columns=x_test_names)

# 处理可能的数据问题，如缺失值和无穷大值
x_train_df.fillna(x_train_df.mean(), inplace=True)
x_test_df.fillna(x_test_df.mean(), inplace=True)
x_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
x_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 初始化 CatBoost 回归模型
catboost_model = CatBoostRegressor(iterations=1000, random_state=42, depth=9, learning_rate=0.045, l2_leaf_reg=0.70)
# depth=4 learning_nate=0.01

# 拟合模型
catboost_model.fit(x_train, y_train, verbose=False)

# # 指定保存路径
# model_save_path = "../反演/catboost_model_agb.bin"
# # 保存模型
# catboost_model.save_model(model_save_path)
# print("Model saved successfully.")

# 计算测试集的结果
y_test_pred = catboost_model.predict(x_test)  # 对测试数据进行预测
test_RMSE = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2_score = r2_score(y_test, y_test_pred)  # 计算测试集的模型评分

# 计算训练集的结果
y_train_pred = catboost_model.predict(x_train)  # 对训练数据进行预测
train_RMSE = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2_score = r2_score(y_train, y_train_pred)  # 计算训练集的模型评分

# 计算相对 RMSE
y_mean = np.mean(y_train)
relative_RMSE_train = train_RMSE / y_mean * 100
relative_RMSE_test = test_RMSE / y_mean * 100

print(f"在 {'CatBoost'} 模型下：")
print(
    'Train: RMSE: {:.2f}cm, rRMSE: {:.2f}%, R²: {:.3f}'.format(train_RMSE, relative_RMSE_train, train_r2_score))
print('Test:  RMSE: {:.2f}cm, rRMSE: {:.2f}%, R²: {:.3f}'.format(test_RMSE, relative_RMSE_test, test_r2_score))

# ======================
# 真实值 vs 预测值联合图（训练集+测试集）
# ======================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置全局字体为新罗马（Times New Roman）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'  # 数学公式字体
plt.rcParams['mathtext.rm'] = 'Times New Roman'  # 常规数学字体
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体数学字体
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'  # 粗体数学字体

# 创建数据框
data_train = pd.DataFrame({
    'True': y_train,
    'Predicted': y_train_pred,
    'Data Set': 'Train'
})

data_test = pd.DataFrame({
    'True': y_test,
    'Predicted': y_test_pred,
    'Data Set': 'Test'
})

data = pd.concat([data_train, data_test])

# 自定义调色板
palette = {'Train': '#3498db', 'Test': '#e74c3c'}  # 蓝色和红色

# 创建联合图 - 修正了图形大小设置
plt.figure(figsize=(8, 8))  # 主图形大小
g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Data Set", 
                  height=8, palette=palette, space=0)  # JointGrid高度调整为6

# 绘制散点图
g.plot_joint(sns.scatterplot, alpha=0.7, s=25, edgecolor='w', linewidth=0.5)

# 添加边缘分布
g.plot_marginals(sns.histplot, kde=True, element='step', 
                 alpha=0.5, common_norm=False)

# 添加对角线
g.ax_joint.plot([data['True'].min(), data['True'].max()], 
                [data['True'].min(), data['True'].max()], 
                c="black", alpha=0.7, linestyle='--', label='x=y')

# 添加统计信息
ax = g.ax_joint
ax.text(0.95, 0.23, f'Train $R^2$ = {train_r2_score:.3f}', transform=ax.transAxes, 
        fontsize=14, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))
ax.text(0.95, 0.14, f'Test $R^2$ = {test_r2_score:.3f}', transform=ax.transAxes, 
        fontsize=14, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))
# 新增RMSE指标
#ax.text(0.95, 0.19, f'Train rRMSE = {relative_RMSE_train:.2f}%', transform=ax.transAxes,
        #fontsize=14, verticalalignment='bottom', horizontalalignment='right',
        #bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))
#ax.text(0.95, 0.10, f'Test rRMSE = {relative_RMSE_test:.2f}%', transform=ax.transAxes,
        #fontsize=14, verticalalignment='bottom', horizontalalignment='right',
        #bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))



# 设置轴标签字体大小
ax.set_xlabel('True Value', fontsize=18)  # 修改x轴标签字体大小
ax.set_ylabel('Predicted Value', fontsize=18)  # 修改y轴标签字体大小

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=18)  # 修改刻度标签字体大小

# 强制设置x轴刻度为20,40,60,...（间隔20）
x_ticks = np.arange(0, data['True'].max()+20, 20)  # 从20开始，间隔20
ax.set_xticks(x_ticks)  # 设置x轴刻度位置

# 设置图例和标题
ax.legend(loc='upper left', frameon=True, framealpha=0.8,fontsize=15)
g.fig.tight_layout()  # 使用JointGrid的fig属性进行tight_layout

plt.show()
