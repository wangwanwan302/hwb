import pandas as pd
import numpy as np
from pyswarm import pso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取训练数据
file_path = '附件一（训练集）.xlsx'
sheet_names = ['材料1', '材料2', '材料3', '材料4']

# 将所有材料的数据读取并合并
data_frames = []
for i, sheet in enumerate(sheet_names):
    df = pd.read_excel(file_path, sheet_name=sheet)
    df['磁芯材料'] = f'材料{i + 1}'
    data_frames.append(df)

# 合并为一个总的DataFrame并复制
data = pd.concat(data_frames, ignore_index=True).copy()

# 转换磁通密度列为数值类型（非数值的会变为NaN）
magnetic_density_columns = data.columns[4:]
data[magnetic_density_columns] = data[magnetic_density_columns].apply(pd.to_numeric, errors='coerce')

# 提取磁通密度峰值
data['磁通密度峰值'] = data[magnetic_density_columns].max(axis=1)

# 确保分类变量没有空值并且类型正确
data['励磁波形'] = data['励磁波形'].fillna('正弦波')
data['磁芯材料'] = data['磁芯材料'].fillna('材料1')

# 将分类变量转换为字符串，以确保编码时正确处理
data['励磁波形'] = data['励磁波形'].astype(str)
data['磁芯材料'] = data['磁芯材料'].astype(str)

# 定义特征和目标变量
X = data[['温度，oC', '频率，Hz', '励磁波形', '磁通密度峰值', '磁芯材料']]

# 处理类别特征的One-Hot编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['温度，oC', '频率，Hz', '磁通密度峰值']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['励磁波形', '磁芯材料'])
    ]
)

# 对训练集进行One-Hot编码
X_encoded = preprocessor.fit_transform(X)


# 定义安全的对数函数，防止NaN或负数
def safe_log(value):
    return np.log(np.maximum(value, 1e-9))  # 防止对数输入为0或负值


# 定义预测函数
def compute_result(df, sumdb, dB):
    try:
        result = (-0.35 * safe_log(df['温度，oC']) +
                  0.02 * df.get('材料3', 0) +
                  0.09 * df.get('材料4', 0) +
                  1.75 * safe_log(df['频率，Hz']) +
                  0.27 * safe_log(sumdb) +
                  1.33 * safe_log(dB) -
                  0.25 * df.get('材料1', 0) +
                  0.94)
        return result
    except Exception as e:
        print(f"Error in compute_result: {e}")
        return np.inf  # 若发生错误，返回一个大的适应度值


# 计算 sumdb, dB, B, Bm
for idx, df in enumerate([data]):
    last_1024_columns = df.iloc[:, -1024:].apply(pd.to_numeric, errors='coerce')
    data1 = last_1024_columns.shift(1).fillna(0)
    data = (last_1024_columns - data1).fillna(0).abs() ** 4
    cumulative_sum = data.cumsum()

    df['sumdb'] = cumulative_sum.iloc[-1]
    df['B'] = last_1024_columns.max(axis=1)
    df['Bm'] = last_1024_columns.min(axis=1)
    df['dB'] = df['B'] - df['Bm']

# 保存适应度历史和粒子轨迹
fitness_history = []


# 定义目标函数：磁芯损耗和传输磁能
def objective_functions(x):
    try:
        temperature, frequency, waveform, peak_magnetic_flux, material_index = x
        waveform_map = {0: '正弦波', 1: '三角波', 2: '梯形波'}
        material_map = {1: '材料1', 2: '材料2', 3: '材料3', 4: '材料4'}

        # 确保 material_index 在 1 到 4 之间
        material_index = int(np.clip(material_index, 1, 4))
        waveform = waveform_map[int(waveform)]
        material = material_map[material_index]

        # 输入数据
        input_data = pd.DataFrame({
            '温度，oC': [temperature],
            '频率，Hz': [frequency],
            '励磁波形': [waveform],
            '磁通密度峰值': [peak_magnetic_flux],
            '磁芯材料': [material]
        }).fillna(0)

        # 使用预测函数计算磁芯损耗
        sumdb_value = df['sumdb'].iloc[-1]
        dB_value = df['dB'].iloc[-1]
        core_loss = compute_result(input_data, sumdb_value, dB_value)

        # 传输磁能
        transmission_energy = frequency * peak_magnetic_flux

        # 返回一个标量（例如：加权的核心损耗和传输磁能）
        fitness_value = float(core_loss + transmission_energy) / 1000  # 缩放适应度值
        if np.isnan(fitness_value) or np.isinf(fitness_value):
            return np.inf  # 若适应度无效，返回一个较大的值，避免影响优化
        fitness_history.append(fitness_value)  # 记录适应度
        return fitness_value
    except Exception as e:
        print(f"Error in objective_functions: {e}")
        return np.inf  # 返回一个较大的适应度值以避免异常中断


# 粒子群优化
def pso_optimization():
    lb = [25, 50000, 0, 0.01, 1]
    ub = [90, 500000, 2, 1, 4]

    # 粒子群参数
    swarm_size = 200  # 粒子数量
    num_variables = 5  # 变量维度
    max_iterations = 100  # 最大迭代次数
    w = 0.7  # 惯性权重
    c1 = 1.5  # 认知学习因子
    c2 = 1.5  # 社会学习因子

    # 初始化粒子位置和速度
    particles = np.random.rand(swarm_size, num_variables)
    velocities = np.random.randn(swarm_size, num_variables) * 0.1  # 初始化速度

    # 初始化粒子位置边界
    particles[:, 0] = lb[0] + (ub[0] - lb[0]) * particles[:, 0]
    particles[:, 1] = lb[1] + (ub[1] - lb[1]) * particles[:, 1]
    particles[:, 2] = np.floor(lb[2] + (ub[2] - lb[2]) * particles[:, 2]).astype(int)
    particles[:, 3] = lb[3] + (ub[3] - lb[3]) * particles[:, 3]

    # 确保 material_index 在 1 到 4 之间
    particles[:, 4] = np.clip(np.floor(lb[4] + (ub[4] - lb[4]) * particles[:, 4]).astype(int), 1, 4)

    # 记录历史最优解
    pbest_positions = particles.copy()  # 每个粒子的历史最优位置
    pbest_fitness = np.array([objective_functions(p) for p in particles])  # 每个粒子的历史最优适应度

    gbest_position = pbest_positions[np.argmin(pbest_fitness)]  # 全局最优位置
    gbest_fitness = np.min(pbest_fitness)  # 全局最优适应度

    # 保存粒子轨迹
    trajectory = []

    # 开始迭代
    for iteration in range(max_iterations):
        # 计算当前适应度
        fitness_values = np.array([objective_functions(p) for p in particles])

        # 更新每个粒子的历史最优位置
        for i in range(swarm_size):
            if fitness_values[i] < pbest_fitness[i]:
                pbest_fitness[i] = fitness_values[i]
                pbest_positions[i] = particles[i].copy()

        # 更新全局最优位置
        current_gbest_index = np.argmin(fitness_values)
        if fitness_values[current_gbest_index] < gbest_fitness:
            gbest_fitness = fitness_values[current_gbest_index]
            gbest_position = particles[current_gbest_index].copy()

        # 更新粒子位置和速度
        r1 = np.random.rand(swarm_size, num_variables)
        r2 = np.random.rand(swarm_size, num_variables)

        cognitive_velocity = c1 * r1 * (pbest_positions - particles)
        social_velocity = c2 * r2 * (gbest_position - particles)

        velocities = w * velocities + cognitive_velocity + social_velocity
        particles = particles + velocities

        # 保存粒子位置以便绘制轨迹
        trajectory.append(particles.copy())

        # 打印当前迭代和最优适应度
        print(f"Iteration {iteration + 1}/{max_iterations}, Global Best Fitness: {gbest_fitness}")

    return gbest_position, gbest_fitness, trajectory


# 调用粒子群优化函数
xopt, fopt, trajectory = pso_optimization()

# 打印最优解
print("最优解:", xopt)
print("最优适应度:", fopt)

# 将最优解映射回实际参数
best_temperature, best_frequency, best_waveform_index, best_peak_flux, best_material_index = xopt
waveform_str = ['正弦波', '三角波', '梯形波'][int(best_waveform_index)]
material_str = f"材料{int(best_material_index)}"

print(
    f"最优条件: 温度={best_temperature}, 频率={best_frequency}, 波形={waveform_str}, 磁通密度峰值={best_peak_flux}, 材料={material_str}"
)

# 绘制适应度曲线
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, label='适应度变化', color='blue')
plt.title('粒子群算法适应度随迭代次数变化')
plt.xlabel('迭代次数')
plt.ylabel('适应度')
plt.grid()
plt.legend()
plt.savefig('适应度曲线.png')  # 保存适应度曲线
plt.close()

# 绘制粒子轨迹
plt.figure(figsize=(10, 6))
for i in range(len(trajectory)):
    plt.scatter(trajectory[i][:, 0], trajectory[i][:, 1], label=f'Iteration {i + 1}', alpha=0.5)
plt.title('粒子轨迹')
plt.xlabel('温度（°C）')
plt.ylabel('频率（Hz）')
plt.legend()
plt.grid()
plt.savefig('粒子轨迹.png')  # 保存粒子轨迹图
plt.close()
