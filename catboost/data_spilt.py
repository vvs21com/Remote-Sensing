import pandas as pd
from sklearn.model_selection import train_test_split

# 读取Excel文件
df = pd.read_excel(r'zong15135.xlsx')

# 随机分割数据集
train, test = train_test_split(df, test_size=0.3, random_state=40)

# 保存训练集和验证集到新的Excel文件
train.to_excel(r'train_data.xlsx', index=False)
test.to_excel(r'test_data.xlsx', index=False)
