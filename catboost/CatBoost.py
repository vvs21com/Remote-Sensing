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
catboost_model = CatBoostRegressor(iterations=1000, random_state=42, depth=9, learning_rate=0.045, l2_leaf_reg=0.60)
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
