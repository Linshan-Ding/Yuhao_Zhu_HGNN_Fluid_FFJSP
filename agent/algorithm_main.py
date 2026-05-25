import os
import json
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import copy
from env.env import SchedulingEnv
try:
    from agent.HGHH_model import Memory
    from agent.PPO_model import PPO
except ImportError:
    from HGHH_model import Memory
    from PPO_model import PPO


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
        self.current_order_id = 1
        self.order_completed_rate = 0
        self.order_completion_rate_last = 0
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
        if self.first_call:
            self.mbom_df = platform.getMBOM()
            self.machine_tuple = tuple(self.mbom_df['machine_id'].unique())
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            config_path = PROJECT_ROOT / "data" / "config.json"
            with open(config_path, 'r', encoding='utf-8') as load_f:
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
        self.orders_df = platform.getOrders()
        self.current_order_completed_rate = self.orders_df['fulfillment_rate'].values[0]

        self.orders_df = self.orders_df[~self.orders_df['order_id'].isin(self.finished_order_ids)]
        self.machines_df = platform.getCurrentMachineStatus()

        self.machines_df.insert(0, 'machine_id', self.machine_tuple)
        self.current_time = platform.getSimulationTime()
        self.environment = SchedulingEnv()
        self.state = self.environment.reset(self.mbom_df, self.orders_df, self.machines_df, self.current_time)
        self.state.current_order_completed_rate = self.current_order_completed_rate
        self.done = False

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

            for _, row in self.machines_df.iterrows():
                if row['end_time'] is not None and row['end_time'] == self.current_time:
                    self.machines_df.loc[row.name, 'task_id'] = None
                    self.machines_df.loc[row.name, 'start_time'] = None
                    self.machines_df.loc[row.name, 'end_time'] = None
            self.state.update(self.current_time, self.orders_df, self.machines_df, fluid_x_resolve=True)
            self.environment.current_time = self.current_time

        if self.current_order_id < len(self.arrival_times):
            self.next_time = self.arrival_times[self.current_order_id]
        else:
            self.next_time += 1000000

        self.reward_sum = 0
        while self.environment.current_time < self.next_time and not self.done:
            with torch.no_grad():
                self.memory.states.append(copy.deepcopy(self.state))
                state = copy.deepcopy(self.state)
                action_index, log_prob, action1, action2 = self.model.policy_old.act(state, self.memory, self.epoch)
                self.state, self.reward, self.done = self.environment.step([action1, action2])
                self.memory.action_indexes.append(action_index)
                self.memory.log_probs.append(log_prob)
                self.memory.rewards.append(self.reward)
                self.memory.is_terminals.append(self.done)
                self.reward_sum += self.reward

        if self.current_order_id < len(self.arrival_times):
            self.current_time = self.arrival_times[self.current_order_id]
            if len(self.memory.is_terminals) > 0:
                self.memory.is_terminals[-1] = False
        else:
            self.current_time += 1000000

        self.discard_orders_id_list.extend(self.environment.discard_order_ids)
        self.finished_order_ids.extend(self.environment.finished_order_ids)

        self.current_order_id += 1
        self.schedule_data = self.environment.schedule_data

        return pd.DataFrame(self.schedule_data), self.current_time
