"""
训练算法：异质神经网络定义、策略网络定义、值网络定义、PPO算法
版本：精细化动作至：机器-工序类型-订单 三级层次；添加订单当前达成率信息；采用流体解定义可选机器;每次随机选择示例训练网络；并设置测试算例
"""
import os
import json
import pandas as pd
import torch
import copy, random
from env.env import SchedulingEnv
from HGHH_model import Memory
from PPO_model import PPO
from collections import deque
from competitionPlatform import CompetitionPlatform
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 临时方案

from visdom import Visdom
viz = Visdom(env='ppo_hgnn')
viz.line([0], [0], win='computation_rate', opts=dict(title='computation_rate'))


# 参赛队伍算法（请封装成类）
class SchedulingAlgorithm:
    def __init__(self):
        self.environment = None  # 环境对象
        self.first_call = True  # 是否首次调用
        # 当前决调度环境的属性
        self.state = None  # 当前状态
        self.action = None  # 当前动作
        self.reward = None  # 当前奖励
        self.done = False  # 是否结束标志
        self.reward_sum = None # 生成的调度方案的总的订单达成率

        # 订单和机器相关数据
        self.orders_df = None  # 订单 DataFrame
        self.machines_df = None  # 机器 DataFrame
        self.current_time = None  # 当前仿真时间
        self.mbom_df = None  # 物料清单 DataFrame
        self.schedule_data = None  # 用于记录调度计划数据
        self.machine_tuple = None  # 机器ID元组

        # 添加设备和模型参数
        self.device = None  # 设备
        self.env_paras = None  # 环境参数
        self.model_paras = None  # 模型参数
        self.train_paras = None  # 训练参数
        self.memory = None  # 记忆库
        self.model = None  # 工序类型-机器对选择 决策模型（每次有可选工序类型-机器对时做出决策）
        # 判断是否加载训练好的模型
        self.load_model = True  # 是否加载模型
        # 文件读取
        self.order_total_count = None  # 订单总数
        self.epoch = None # 当前训练周期
        self.epoch_total = None  # 总的训练周期

        # 初始化初始到达订单ID
        self.current_order_id = 1  # 初始到达订单ID列表
        self.arrival_times = []  # 订单到达时间列表
        # 订单剔除相关参数定义
        self.order_completed_rate = 0  # 订单完成率
        self.order_completion_rate_last = 0  # 上一新订单到达点订单达成率
        self.discard_orders_id_list = []  # 算法运行过程中删除的订单ID列表
        self.finished_order_ids = []
        self.next_time = 0
        self.current_order_completed_rate = 0  # 初始化当前订单达成率

    def reset(self):
        """
        重置算法属性
        :return:
        """
        # 初始化初始到达订单ID
        self.current_order_id = 1  # 初始到达订单ID列表
        # 订单剔除相关参数定义
        self.order_completed_rate = 0  # 订单完成率
        self.order_completion_rate_last = 0  # 上一新订单到达点订单达成率
        self.discard_orders_id_list = []  # 算法运行过程中删除的订单ID列表
        self.finished_order_ids = []
        self.next_time = 0
        self.current_order_completed_rate = 0

    def red_txt_file(self):
        """
        读取测试集文件，返回以下内容：
        - arrival_time: 订单到达时间
        """
        with open(self.path, 'r', encoding='utf-8') as f:
            # 过滤掉空行
            lines = [line.strip() for line in f if line.strip() != '']
            if len(lines) < 4:
                raise ValueError("文件内容行数不足，无法获取订单到达时间")

        # 根据约定，倒数第二行为订单到达时间
        arrival_line = lines[-2]
        # 以空白字符拆分并转换为整数
        arrival_times = list(map(int, arrival_line.split()))

        return arrival_times

    def generate_schedule(self, platform) -> pd.DataFrame:
        """
        动态生成调度计划
        :param platform: 仿真测试平台
        :return: 调度计划DataFrame
                schedule_df: 调度计划DataFrame，包含以下列:
                    - task_id: 任务ID (格式: order_id-process_id)
                    - machine_id: 设备ID
                    - start_time: 计划开始时间
        """

        # 首次调用时实例化环境
        """
        def getMBOM(self) -> pd.DataFrame:

        获取制造BOM(Bill of Materials)信息。

        该方法从仿真实例中提取产品的工艺路线信息，包括每个产品在各生产阶段可选用的设备和相应处理时间。

        Returns:
            pd.DataFrame: 包含制造BOM信息的DataFrame，包含以下列：
                - product_type: 产品类型标识（格式为"0"、"1"、"2"等）
                - stage: 生产阶段标识（格式为"0"、"1"、"2"等）
                - machine_id: 可用于该工序的设备ID
                - process_time(s): 在该设备上完成该工序所需的处理时间（秒）
        """
        if self.first_call:
            self.mbom_df = platform.getMBOM()
            self.machine_tuple = tuple(self.mbom_df['machine_id'].unique())
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # 加载参数配置
            with open("../data/config.json", 'r') as load_f:
                load_dict = json.load(load_f)
            self.env_paras = load_dict["env_paras"]
            self.model_paras = load_dict["model_paras"]
            self.train_paras = load_dict["train_paras"]
            self.model_paras["device"] = self.device
            self.model_paras["actor_in_dim"] = self.model_paras["out_size_ma"] * 2 + self.model_paras["out_size_ope"] * 2
            self.model_paras["critic_in_dim"] = self.model_paras["out_size_ma"] + self.model_paras["out_size_ope"]
            self.memory = Memory()
            self.model = PPO(self.model_paras, self.train_paras)
            self.arrival_times = self.red_txt_file()  # 读取订单信息
            self.order_total_count = len(self.arrival_times)    # 新订单总数
            self.first_call = False  # 设置首次调用标志为False
            print("------------实例化模型成功---------------")

        # 更新bom信息
        self.mbom_df = platform.getMBOM()
        self.machine_tuple = tuple(self.mbom_df['machine_id'].unique())
        # 获取当前仿真状态
        # platform 提供的函数支持
        self.orders_df = platform.getOrders()
        """
        getOrders(self, only_unfinished=True) -> pd.DataFrame:
        获取订单数据及相关状态信息。

        该方法返回包含订单信息的DataFrame，包含订单基本信息、当前进度状态以及全系统订单完成率。

        Args:
            only_unfinished (bool, optional): 是否只返回未完成的订单。默认为True。
                - True: 仅返回尚未完成的订单
                - False: 返回所有订单（包括已完成的）

        Returns:
            pd.DataFrame: 包含订单信息的DataFrame，包含以下列：
                - order_id: 订单唯一标识
                - product_type: 产品类型
                - arrival_time: 订单到达时间
                - due_date: 订单交期
                - current_stage: 当前处理工序（如果当前没有进行中的工序则为上一个工序的结果。若首工序也未开始，则为None）
                - assigned_machine: 分配到的机器ID（如果当前没有分配则为为上一个工序的分配结果。若首工序也未开始，则为None）
                - start_time: 当前工序开始时间（如果未开始则为上一个工序的开始时间。若首工序也未开始，则为None）
                - end_time: 当前工序结束时间（如果未开始则为上一个工序的结束时间。若首工序也未开始，则为None）
                - fulfillment_rate: 当前系统订单达成率（已完成订单/所有已到达订单）
        """
        # 输出订单达成率
        self.current_order_completed_rate = self.orders_df['fulfillment_rate'].values[0]
        # print(f"当前订单达成率: {self.current_order_completed_rate:.2%}")

        # # 删除丢弃的订单
        # self.orders_df = self.orders_df[~self.orders_df['order_id'].isin(self.discard_orders_id_list)]
        # 删除已完工的订单
        self.orders_df = self.orders_df[~self.orders_df['order_id'].isin(self.finished_order_ids)]
        # print('当前调度订单数量：', len(self.orders_df))
        # 读取机器状态
        self.machines_df = platform.getCurrentMachineStatus()
        """
        def getCurrentMachineStatus(self) -> pd.DataFrame:
        获取当前时刻所有机器的状态信息。

        该方法返回一个DataFrame，描述在当前时间点各机器的状态（空闲或正在执行的任务信息）。

        Returns:
            pd.DataFrame: 包含每台机器当前状态的DataFrame，包含以下列：
                - task_id: 当前执行的任务ID（如果机器空闲则为None）
                - start_time: 当前任务的开始时间（如果机器空闲则为None）
                - end_time: 当前任务的结束时间（如果机器空闲则为None）
        """

        # machines_df 中第一列插入 machine_tuple 中的机器ID
        self.machines_df.insert(0, 'machine_id', self.machine_tuple)

        # 获取当前时刻
        self.current_time = platform.getSimulationTime()
        # print(f"当前仿真时间: {current_time} 秒")
        """
        def getSimulationTime(self) -> float:
        获取当前仿真时间（从仿真开始起经过的时间）。

        该方法计算从仿真基础时间点（通常是仿真启动时间）到当前时刻经过的时间。

        Returns:
            float: 仿真经过的时间（以秒为单位）
        """
        self.environment = SchedulingEnv()  # 创建调度环境实例
        self.state = self.environment.reset(self.mbom_df, self.orders_df, self.machines_df, self.current_time)  # 重置环境到初始状态
        self.state.current_order_completed_rate = self.current_order_completed_rate
        self.done = False  # 初始化结束标志为False

        # 所有空闲机器无可选订单，移动时钟到下一个机器空闲点
        while len(self.state.kind_task_available_list) == 0 and sum(self.state.kind_task_unprocessed.values()) > 0:
            # 获取所有机器的空闲时间点
            next_idle_time = min([row['end_time'] for _, row in self.machines_df.iterrows()
                                  if row['end_time'] is not None and row['end_time'] > self.current_time],
                                 default=None)
            if next_idle_time is None:
                raise ValueError("--------------移动时钟报错---------------")
            elif next_idle_time > self.current_time:
                self.current_time = next_idle_time
                # print(f"订单到达时刻无可选动作，移动时钟到下一个机器空闲点: {self.current_time} 秒")

            # 更新机器状态、订单状态
            for _, row in self.machines_df.iterrows():
                if row['end_time'] is not None and row['end_time'] == self.current_time:
                    # 机器空闲，更新状态, 更新machines_df
                    self.machines_df.loc[row.name, 'task_id'] = None
                    self.machines_df.loc[row.name, 'start_time'] = None
                    self.machines_df.loc[row.name, 'end_time'] = None
            # 更新状态对象
            self.state.update(self.current_time, self.orders_df, self.machines_df, fluid_x_resolve=True)
            # 更新环境当前时间
            self.environment.current_time = self.current_time

        # 提取决策截断时刻
        if self.current_order_id < len(self.arrival_times):
            self.next_time = self.arrival_times[self.current_order_id]
        else:
            self.next_time += 1000000

        # 初始化总回报(新订单到达后生成的调度方案的总的订单达成率)
        self.reward_sum = 0
        # 基于图强化学习的调度仿真
        while self.environment.current_time < self.next_time and not self.done:
            with torch.no_grad():
                self.memory.states.append(copy.deepcopy(self.state)) # 在每个决策点，记录当前状态
                state = copy.deepcopy(self.state)  # 深复制当前状态
                action_index, log_prob, action1, action2 = self.model.policy_old.act(state, self.memory, self.epoch)  # 获取当前状态的动作
                self.state, self.reward, self.done = self.environment.step([action1, action2])  # 执行动作，获取下一个状态、奖励和是否结束标志
                # 在每个决策点，记录数据
                self.memory.action_indexes.append(action_index)
                self.memory.log_probs.append(log_prob)
                self.memory.rewards.append(self.reward)
                self.memory.is_terminals.append(self.done)
                self.reward_sum += self.reward

        # # 保存订单剔除决策网络回报
        # print("完工数量：", self.reward_sum)

        if self.current_order_id < len(self.arrival_times):
            self.current_time = self.arrival_times[self.current_order_id]
            if len(self.memory.is_terminals) > 0:
                self.memory.is_terminals[-1] = False
        else:
            self.current_time += 1000000

        # print("当前时间", self.current_time)
        # print("<丢弃订单数量>", self.environment.order_discard_count)

        # 更新丢弃的订单列表
        self.discard_orders_id_list.extend(self.environment.discard_order_ids)
        # 更新已完工的订单
        self.finished_order_ids.extend(self.environment.finished_order_ids)

        self.current_order_id += 1
        self.schedule_data = self.environment.schedule_data  # 获取调度计划数据
        """
        输出DataFrame结果
        schedule_df: 调度计划DataFrame，包含以下列:
                    - task_id: 任务ID (格式: order_id-process_id)
                    - machine_id: 设备ID
                    - start_time: 计划开始时间
        """

        return pd.DataFrame(self.schedule_data), self.current_time


# 运行仿真
if __name__ == '__main__':
    # 1. 定义初始订单完工率
    order_completed_rate = 0  # 订单初始完工率
    # 2. 创建参赛队伍算法实例
    team_algorithm = SchedulingAlgorithm()
    team_algorithm.epoch_total = 1000
    # 训练多个周期
    for epoch in range(1000):
        team_algorithm.epoch = epoch
        # 0. 配置竞赛案例
        # instance_name = 'test_num2000_lam0.05_change0__3.txt'
        instance_names = ['train_num2000_lam0.05_change0__1.txt', 'train_num2000_lam0.05_change0__2.txt',
                          'train_num2000_lam0.05_change0__3.txt', 'train_num2000_lam0.05_change0__4.txt',
                          'train_num2000_lam0.05_change0__5.txt']
        instance_index = epoch % len(instance_names)
        instance_name = instance_names[instance_index]

        # 打印当前工作目录和文件路径，以便调试
        path = os.path.join('../data', 'instance', 'competition', instance_name)
        # print("当前工作目录是:", os.getcwd())
        # print("尝试打开的文件路径是:", path)

        # 1. 创建仿真平台实例
        platform = CompetitionPlatform()

        # 2. 重置算法属性
        team_algorithm.reset()
        team_algorithm.path = os.path.join('../data', 'instance', 'competition', instance_name)

        # 4. 运行仿真
        result = platform.run_simulation(path, team_algorithm, True)
        """
        def run_simulation(self, path, algorithm_module, isTimeout=True) -> dict:
        运行仿真案例
            :param path: 仿真案例的路径
            :param algorithm_module: 动态调度算法
            :param isTimeout:  是否开启30s实时调度限制
            :return: 返回仿真结果
        """
        print("动作执行网络总回报：", sum(team_algorithm.memory.rewards))
        if len(team_algorithm.memory.rewards) > 0:
            # 更新策略网络
            total_loss1 = team_algorithm.model.update(team_algorithm.memory)
            team_algorithm.memory.clear_memory()

        # 保存该训练算例下的订单完工率到该算例的csv文件
        orders = platform.getOrders(False)
        current_completion_rate = orders['fulfillment_rate'].values[0]

        # 创建结果目录
        result_dir = '../result'
        os.makedirs(result_dir, exist_ok=True)

        # 保存到该算例对应的CSV文件
        csv_filename = os.path.join(result_dir, f"train_curve_instance_{instance_index}.csv")

        # 如果文件不存在，创建并写入表头
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', encoding='utf-8') as f:
                f.write("epoch,completion_rate\n")

        # 追加当前周期的数据
        with open(csv_filename, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{current_completion_rate}\n")



        """运行测试算例"""
        instance_name_test = 'train_num2000_lam0.05_change0__3.txt'
        # 打印当前工作目录和文件路径，以便调试
        path = os.path.join('../data', 'instance', 'competition', instance_name_test)
        # print("当前工作目录是:", os.getcwd())
        # print("尝试打开的文件路径是:", path)

        # 1. 创建仿真平台实例
        platform = CompetitionPlatform()

        # 2. 重置算法属性
        team_algorithm.reset()
        team_algorithm.path = os.path.join('../data', 'instance', 'competition', instance_name_test)

        # 4. 运行仿真
        result = platform.run_simulation(path, team_algorithm, True)

        # 5. 输出结果
        orders = platform.getOrders(False)
        print("\n仿真结果:")
        print(f"订单达成率: {orders['fulfillment_rate'].values[0]:.2%}")
        viz.line([-orders['fulfillment_rate'].values[0]], [epoch], win='computation_rate', update='append')

        # 保存最优模型并生成甘特图
        if order_completed_rate <= orders['fulfillment_rate'].values[0]:
            # 更新最优订单达成率模型
            order_completed_rate = orders['fulfillment_rate'].values[0]
            platform.getGantta(0, 1000)
            """
            def getGantta(self, startTime, endTime)
            生成甘特图
            :param startTime: 开始时间点
            :param endTime: 结束时间点
            """
            save_file = os.path.join('../result', f"ppo_policy_model.pt")
            torch.save(team_algorithm.model.policy.state_dict(), save_file)