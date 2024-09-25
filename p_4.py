import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# 读取训练数据
file_path = r'附件一（训练集）.xlsx'

sheet_names = ['材料1', '材料2', '材料3', '材料4']

# 将所有材料的数据读取并合并
data_frames = []
for i, sheet in enumerate(sheet_names):
    df = pd.read_excel(file_path, sheet_name=sheet)
    df['磁芯材料'] = f'材料{i+1}'  # 使用文本名称而不是数值编码
    data_frames.append(df)

# 合并为一个总的DataFrame
data = pd.concat(data_frames, ignore_index=True)

# 转换磁通密度列为数值类型（非数值的会变为NaN）
magnetic_density_columns = data.columns[4:]  # 从第5列开始是磁通密度
data[magnetic_density_columns] = data[magnetic_density_columns].apply(pd.to_numeric, errors='coerce')

# 提取磁通密度峰值
data['磁通密度峰值'] = data[magnetic_density_columns].max(axis=1)

# 确保分类变量没有空值并且类型正确
data['励磁波形'].fillna('正弦波', inplace=True)
data['磁芯材料'].fillna('材料1', inplace=True)

# 将分类变量转换为字符串，以确保编码时正确处理
data['励磁波形'] = data['励磁波形'].astype(str)
data['磁芯材料'] = data['磁芯材料'].astype(str)

# 定义特征和目标变量
X = data[['温度，oC', '频率，Hz', '励磁波形', '磁通密度峰值', '磁芯材料']]
y = data['磁芯损耗，w/m3']

# 处理类别特征的One-Hot编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['温度，oC', '频率，Hz', '磁通密度峰值']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['励磁波形', '磁芯材料'])
    ]
)

# 对训练集进行One-Hot编码
X_encoded = preprocessor.fit_transform(X)

# 使用 XGBRegressor 初始化模型，并启用 GPU
model = XGBRegressor(n_estimators=100, random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')

# 训练模型
model.fit(X_encoded, y)

# 定义交叉验证方法
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 执行交叉验证并进行预测
y_pred = cross_val_predict(model, X_encoded, y, cv=kfold)

# 计算评价指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# 绘制交叉验证R²分数的变化
r2_scores = cross_val_score(model, X_encoded, y, scoring='r2', cv=kfold)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), r2_scores, marker='o', label='R² Score per Fold')
plt.title('R² Score for Each Fold')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('r2_score_per_fold.png')
plt.close()  # 关闭图像
# 绘制实际值和预测值的对比图
plt.figure(figsize=(8, 6))
plt.plot(y.values, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Magnetic Core Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('A-P_COMP.png')
plt.close()  # 关闭图像
# 绘制误差分布的箱形图
errors = y - y_pred
plt.figure(figsize=(8, 6))
sns.boxplot(data=[errors], width=0.5)
plt.title('Error Distribution')
plt.ylabel('Error')
plt.show()
plt.savefig('ERROR.png')
plt.close()  # 关闭图像
# 绘制实际值和预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.5, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs Predicted Scatter Plot')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()
plt.savefig('A-P.png')
plt.close()  # 关闭图像
# 交叉验证的其他指标（MSE、RMSE、MAE）
scores_mse = cross_val_score(model, X_encoded, y, scoring='neg_mean_squared_error', cv=kfold)
scores_rmse = cross_val_score(model, X_encoded, y, scoring='neg_root_mean_squared_error', cv=kfold)
scores_mae = cross_val_score(model, X_encoded, y, scoring='neg_mean_absolute_error', cv=kfold)

# 打印交叉验证平均分数
print(f"Cross-Validation Average MSE: {-scores_mse.mean()}")
print(f"Cross-Validation Average RMSE: {-scores_rmse.mean()}")
print(f"Cross-Validation Average MAE: {-scores_mae.mean()}")
print(f"Cross-Validation Average R²: {r2_scores.mean()}")

# ================== 测试集部分 ==================

# 读取测试集数据
test_file_path = r'附件三（测试集）.xlsx'
test_data = pd.read_excel(test_file_path)

# 处理测试集中的励磁波形
test_data['励磁波形'] = test_data['励磁波形'].map({'正弦波': 0, '三角波': 1, '梯形波': 2})

# 转换磁通密度列为数值类型（非数值的会变为NaN）
magnetic_density_columns_test = test_data.columns[4:]  # 第5列开始是磁通密度
test_data[magnetic_density_columns_test] = test_data[magnetic_density_columns_test].apply(pd.to_numeric, errors='coerce')

# 提取测试集的磁通密度峰值
test_data['磁通密度峰值'] = test_data[magnetic_density_columns_test].max(axis=1)

# 确保分类变量没有空值并且类型正确
test_data['磁芯材料'].fillna('材料1', inplace=True)  # 填充缺失值，假设 '材料1' 是默认值
test_data['励磁波形'] = test_data['励磁波形'].astype(str)
test_data['磁芯材料'] = test_data['磁芯材料'].astype(str)

# 提取测试集的特征（温度，频率，励磁波形，磁通密度峰值，磁芯材料）
X_test_final = test_data[['温度，oC', '频率，Hz', '励磁波形', '磁通密度峰值', '磁芯材料']]

# 使用与训练集相同的预处理器进行One-Hot编码
X_test_final_encoded = preprocessor.transform(X_test_final)

# 进行预测
test_predictions = model.predict(X_test_final_encoded)
test_predictions_df = pd.DataFrame(test_predictions, columns=['Predictions'])
test_predictions_df.to_excel('test_predictions.xlsx', index=False)
# 绘制测试集的预测值曲线
plt.figure(figsize=(8, 6))
plt.plot(test_predictions, label='Predicted', alpha=0.7)
plt.title('Predicted Values for Test Set')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Magnetic Core Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('TEST.png')
plt.close()  # 关闭图像
# 特别展示指定序号的预测结果
sample_indices = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
special_results = pd.DataFrame({
    'Sample Index': sample_indices,
    'Predicted Core Loss': test_predictions[sample_indices]
})

# 打印特别样本的预测结果
print(special_results)
# ==========优化============
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor  # 假设使用随机森林进行回归
from sklearn.model_selection import train_test_split


#
# ================== 遗传算法优化部分 ==================

# 定义两个目标函数：磁芯损耗和传输磁能
def objective_functions(individual, model, preprocessor, X):
    temperature = individual[0]
    frequency = abs(individual[1])  # 确保频率为正
    waveform = individual[2]  # 0: 正弦波, 1: 三角波, 2: 梯形波
    peak_magnetic_flux = abs(individual[3])  # 确保磁通密度峰值为正
    material = individual[4]  # 1: 材料1, 2: 材料2, 3: 材料3, 4: 材料4

    # 将波形和材料分别映射回原来的分类
    waveform = '正弦波' if waveform == 0 else '三角波' if waveform == 1 else '梯形波'
    material = '材料' + str(int(material))

    # 创建原始输入特征
    input_data = pd.DataFrame({
        '温度，oC': [temperature],
        '频率，Hz': [frequency],
        '励磁波形': [waveform],
        '磁通密度峰值': [peak_magnetic_flux],
        '磁芯材料': [material]
    })

    # 合并个体数据与训练数据进行编码，以确保编码一致
    combined_data = pd.concat([X, input_data], ignore_index=True)
    combined_encoded = preprocessor.transform(combined_data)

    # 取出最后一行，即个体的编码
    X_encoded = combined_encoded[-1].reshape(1, -1)

    # 使用模型预测磁芯损耗
    core_loss = model.predict(X_encoded)

    # 计算传输磁能（频率 × 磁通密度峰值）
    transmission_energy = frequency * peak_magnetic_flux

    # 返回两个目标值：第一个是最小化磁芯损耗，第二个是最大化传输磁能（负的传输磁能用于最大化）
    return core_loss[0], -transmission_energy


import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# 固定随机种子以确保每次结果一致
random.seed(42)

# 创建适应度和个体类
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # 多目标优化
creator.create("Individual", list, fitness=creator.FitnessMulti)

# 定义初始化函数
def create_individual():
    return creator.Individual([random.choice([25, 50, 70, 90]),  # 温度作为离散变量
                               random.uniform(50000, 500000),  # 频率
                               random.randint(0, 2),  # 励磁波形 (0:正弦波, 1:三角波, 2:梯形波)
                               random.uniform(0.01, 1),  # 磁通密度峰值
                               random.randint(1, 4)])  # 磁芯材料 (1到4)

# 修正的突变操作（基于离散变量的优化）
def mutate_individual(individual):
    individual[0] = random.choice([25, 50, 70, 90])  # 离散的温度
    individual[1] = max(50000, min(500000, abs(individual[1])))  # 连续的频率
    individual[2] = random.randint(0, 2)  # 离散的波形
    individual[3] = max(0.01, min(1, abs(individual[3])))  # 连续的磁通密度峰值
    individual[4] = random.randint(1, 4)  # 离散的材料
    return individual,

# 修正的交叉操作（离散选择）
def cx_individual(ind1, ind2):
    ind1[0], ind2[0] = random.choice([25, 50, 70, 90]), random.choice([25, 50, 70, 90])  # 离散的温度
    ind1[1], ind2[1] = random.uniform(50000, 500000), random.uniform(50000, 500000)  # 连续的频率
    ind1[3], ind2[3] = random.uniform(0.01, 1), random.uniform(0.01, 1)  # 连续的磁通密度峰值
    ind1[2], ind2[2] = random.choice([0, 1, 2]), random.choice([0, 1, 2])  # 离散的波形
    ind1[4], ind2[4] = random.choice([1, 2, 3, 4]), random.choice([1, 2, 3, 4])  # 离散的材料
    return ind1, ind2

# 注册遗传算法的操作
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评估函数
toolbox.register("evaluate", objective_functions, model=model, preprocessor=preprocessor, X=X)

# 注册自定义交叉和突变操作
toolbox.register("mate", cx_individual)
toolbox.register("mutate", mutate_individual)

# 注册选择操作（NSGA-II 的非支配排序）
toolbox.register("select", tools.selNSGA2)

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# 定义两个目标函数：磁芯损耗和传输磁能
def objective_functions(individual, model, preprocessor, X):
    temperature = individual[0]
    frequency = abs(individual[1])  # 确保频率为正
    waveform = individual[2]  # 0: 正弦波, 1: 三角波, 2: 梯形波
    peak_magnetic_flux = abs(individual[3])  # 确保磁通密度峰值为正
    material = individual[4]  # 1: 材料1, 2: 材料2, 3: 材料3, 4: 材料4

    # 将波形和材料分别映射回原来的分类
    waveform = '正弦波' if waveform == 0 else '三角波' if waveform == 1 else '梯形波'
    material = '材料' + str(int(material))

    # 创建原始输入特征
    input_data = pd.DataFrame({
        '温度，oC': [temperature],
        '频率，Hz': [frequency],
        '励磁波形': [waveform],
        '磁通密度峰值': [peak_magnetic_flux],
        '磁芯材料': [material]
    })

    # 合并个体数据与训练数据进行编码，以确保编码一致
    combined_data = pd.concat([X, input_data], ignore_index=True)
    combined_encoded = preprocessor.transform(combined_data)

    # 取出最后一行，即个体的编码
    X_encoded = combined_encoded[-1].reshape(1, -1)

    # 使用模型预测磁芯损耗
    core_loss = model.predict(X_encoded)

    # 计算传输磁能（频率 × 磁通密度峰值）
    transmission_energy = frequency * peak_magnetic_flux

    # 如果 core_loss 小于 0，给一个非常大的惩罚值
    if core_loss[0] < 0:
        core_loss[0] = 1e6  # 设置为非常大的惩罚值

    # 返回两个目标值：第一个是最小化磁芯损耗，第二个是最大化传输磁能（负的传输磁能用于最大化）
    return core_loss[0], -transmission_energy


# 在 multi_objective_optimization 函数中筛选有效的 core_loss
def multi_objective_optimization():
    population = toolbox.population(n=300)  # 种群大小
    n_gen = 100  # 增加代数
    halloffame = tools.HallOfFame(1)  # 存储最优个体

    # 初始化统计和日志记录
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'avg', 'min']

    # 适应度变化记录
    avg_fitness_vals = []
    min_fitness_vals = []

    # 初始化 Pareto 前沿存储
    pareto_fronts = []  # 初始化空列表

    # 执行 NSGA-II 遗传算法
    for gen in range(n_gen):
        population, logbook_gen = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=100,  # 父代
            lambda_=200,  # 子代
            cxpb=0.7,  # 交叉概率
            mutpb=0.2,  # 突变概率
            ngen=1,
            stats=stats,
            halloffame=halloffame,
            verbose=False
        )

        # 修复：检查所有个体的适应度并处理 core_loss < 0 的情况
        for ind in population:
            core_loss, _ = objective_functions(ind, model, preprocessor, X)
            attempts = 0  # 用来记录突变的次数，防止无限循环
            max_attempts = 10  # 最大突变尝试次数
            while core_loss >= 1e6 and attempts < max_attempts:  # 当 core_loss 为惩罚值时，重新突变
                toolbox.mutate(ind)  # 重新突变
                core_loss, _ = objective_functions(ind, model, preprocessor, X)  # 重新评估
                attempts += 1
                print(f"Invalid core loss: {core_loss} for individual: {ind}. Attempt {attempts}")

            # 如果尝试超过最大次数，重新交叉产生新的个体
            if core_loss >= 1e6:
                ind1, ind2 = toolbox.mate(ind, random.choice(population))  # 交叉生成新个体
                ind[:] = ind1[:]  # 用新个体替换无效个体

        # 记录适应度值
        avg_fitness_vals.append(logbook_gen[0]["avg"])  # 平均适应度
        min_fitness_vals.append(logbook_gen[0]["min"])  # 最小适应度

        # 获取这一代的 Pareto 前沿
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        pareto_fronts.append(pareto_front)

    # 过滤掉负的 core_loss 值，只选择 >= 0 的个体，找出其中最小的值
    valid_core_losses = [ind.fitness.values[0] for ind in population if ind.fitness.values[0] >= 0]

    # 检查是否有有效的 core_loss，并确保不会返回 None
    if valid_core_losses:
        best_core_loss = min(valid_core_losses)
        print(f"Best valid core_loss: {best_core_loss}")
    else:
        print("No valid core_loss found.")
        best_core_loss = None

    # 适应度变化曲线绘制
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_gen + 1), avg_fitness_vals, label="Average Fitness", color="blue")
    plt.plot(range(1, n_gen + 1), min_fitness_vals, label="Minimum Fitness", color="red")
    plt.title("Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('fitness_over_generations.png')
    plt.close()  # 关闭图像

    # 返回最终 Pareto 前沿、日志、最优个体
    return pareto_fronts, logbook, halloffame


# 调用多目标优化函数，获取 Pareto 前沿、日志、以及最优个体
pareto_fronts, logbook, halloffame = multi_objective_optimization()

# 打印最优解和对应的适应度
print("最优个体:", halloffame[0])
print("最优个体的适应度:", halloffame[0].fitness.values)

# 打印最优个体的结果
best_individual = halloffame[0]
print(f"最优个体: 温度={best_individual[0]}, 频率={best_individual[1]}, "
      f"波形={best_individual[2]}, 磁通密度峰值={best_individual[3]}, 材料={best_individual[4]}")




