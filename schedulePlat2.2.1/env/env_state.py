"""
环境文件：输入动作，返回各节点的状态、奖励、各节点的邻接矩阵
"""

import sys
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
import pandas as pd
from gym import spaces
from typing import Dict, Tuple, Any, List
import gurobipy as gp
from gurobipy import GRB


# 定义调度状态类
class SchedulingState:
    def __init__(self):
        # 基本属性
        self.current_time = None  # 当前仿真时间
        self.orders_df = None  # 当前订单数据 DataFrame
        self.machines_df = None  # 当前机器状态数据 DataFrame
        self.fluid_x = None  # 流体模型解，表示每种工序在每台机器上的分配比例 {(m, (r, j)): rate}

        # 以下数据结构将在第一次调用generate_schedule时从MBOM初始化
        self.kind_tuple = None  # 所有工件类型元组
        self.task_r_dict = None  # 每种工件类型的工序顺序字典 {r: [0,1,2,...]}
        self.kind_task_tuple = None  # 所有工序类型元组 (r,j) 其中 r 为工件类型，j 为工序类型
        self.machine_tuple = None  # 所有机器ID元组
        self.machine_rj_dict = None  # 每种工序可用的机器 {(r,j): [m1, m2,...]}
        self.kind_task_machine_dict = None  # 每个机器的可选工序类型字典 {m: [(r,j), (r,j), ...]}
        self.time_mrj_dict = None  # 每台机器加工每种工序的时间 {m: {(r,j): time}}
        self.time_rj_dict = None  # 每种工序的平均加工时间 {(r,j): avg_time}

        # 节点和边状态
        self.feat_mas = None  # 机器状态矩阵 [[],...]
        self.feat_opes = None  # 工序类型状态矩阵 [[],...]
        self.proc_times = None  # 机器-工序类型边状态矩阵（加工速率）
        self.task_machine_edge_fluid_rate_matrix = None  # 机器-工序类型边状态矩阵（流体处理速率）
        self.task_machine_edge_fluid_available_matrix = None  # 机器-工序类型边状态矩阵（是否可用）

        # 邻接矩阵
        self.ope_ma_adj = None  # 机器-工序类型实际邻接矩阵
        self.ope_ma_fl_adj = None  # 机器-工序类型流体邻接矩阵
        self.ope_pre_adj = None  # 工序类型紧前工序邻接矩阵
        self.ope_sub_adj = None  # 工序类型紧后工序邻接矩阵

        # 实际离散参数下和流体解下工序类型-机器可选矩阵
        self.eligible = None  # 工序-机器可选矩阵，表示当前时刻工序类型是否可分配到机器上
        self.eligible_fluid = None  # 流体解下工序-机器可选矩阵，表示流体解下工序类型是否可分配到机器上
        self.eligible_rj_orders = None # 工序类型中是否存在待处理订单
        self.eligible_rj_selected_orders = None  # 工序类型中是否存在可选订单

        # 当前状态的其他属性
        self.kind_task_available_list = None  # 当前时刻可选工序类型列表
        self.kind_task_idle_id_list = None  # 当前时刻各工序类型处于所有工序阶段未分配机器的订单ID列表
        self.kind_task_idle_id_due_date_list = None
        self.kind_task_unprocessed = None  # 当前时刻各工序类型的待处理数量
        self.kind_task_unprocessed_id_list = None  # 当前时刻各工序类型的待处理订单ID列表
        self.machine_idle_rj_dict = None  # 当前时刻每种工序的可选空闲机器列表 {(r,j): [m1, m2,...]}

        # 流体属性
        self.machine_rj_fluid_dict = None  # 流体解中每种工序可用的机器 {(r,j): [m1, m2,...]}
        self.rate_total_rj_fluid_x = None  # 流体解中每种工序的总加工速率 {rj: rate}
        # 流体解中每种工序在每个机器上的加工速率
        self.rate_m_rj_fluid_x = None  # {(m, rj): rate}

        # 初始化到达订单数、按时完工订单数和丢弃订单数
        self.order_arrival_count = 0  # 当前决策点到达订单总数
        self.order_completion_count = 0  # 当前决策点按时完工订单总数
        self.order_discard_count = 0  # 当前决策点丢弃订单总数
        self.action_dict = None # 状态动作
        self.current_order_completed_rate = None # 当前订单达成率

    def initialize_from_mbom(self, mbom_df):
        """从MBOM DataFrame初始化必要数据结构和状态的静态特征"""
        # 获取所有产品类型
        self.kind_tuple = tuple(mbom_df['product_type'].unique())
        self.machine_tuple = tuple(mbom_df['machine_id'].unique())

        # 创建每种工件类型的工序顺序字典
        self.task_r_dict = {}
        for r in self.kind_tuple:
            stages = mbom_df[mbom_df['product_type'] == r]['stage'].unique()
            self.task_r_dict[r] = tuple(sorted(stages))

        # 创建每种工序可用的机器
        self.machine_rj_dict = {}
        self.kind_task_machine_dict = {m: [] for m in self.machine_tuple}
        self.time_mrj_dict = {m: {} for m in self.machine_tuple}
        self.time_rj_dict = {}

        # 从MBOM中提取工序-机器-时间关系
        for _, row in mbom_df.iterrows():
            r, j, m, time_val = row['product_type'], row['stage'], row['machine_id'], row['process_time(s)']
            key = (r, j)

            # 添加到工序-机器映射
            if key in self.machine_rj_dict:
                self.machine_rj_dict[key].append(m)
                self.kind_task_machine_dict[m].append(key)
            else:
                self.machine_rj_dict[key] = [m]
                self.kind_task_machine_dict[m] = [key]

            # 添加到机器-工序-时间映射
            self.time_mrj_dict[m][key] = time_val

            # 计算该工序的平均加工时间
            if key in self.time_rj_dict:
                self.time_rj_dict[key].append(time_val)
            else:
                self.time_rj_dict[key] = [time_val]

        # 计算每种工序的平均加工时间
        for key, times in self.time_rj_dict.items():
            self.time_rj_dict[key] = sum(times) / len(times)

        # 创建工序类型元组
        self.kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in self.task_r_dict[r])

        # 初始化各工序类型节点的状态矩阵
        num_tasks = sum(len(v) for v in self.task_r_dict.values())
        num_machines = len(self.machine_tuple)
        self.feat_opes = torch.zeros((num_tasks, 12))  # 各工序类型的状态
        # 初始化各机器节点的状态矩阵：[可选离散工序总数、加工完成时间、是否可用、可选连续机器总数]
        self.feat_mas = torch.zeros((num_machines, 5))  # 各机器的状态矩阵
        # 初始化工序-机器可选矩阵
        self.eligible = torch.full((num_tasks, num_machines), False)  # 工序-机器可选矩阵
        self.eligible_rj_orders = torch.full((len(self.kind_task_tuple), 1), False)
        self.eligible_fluid = torch.full((num_tasks, num_machines), False)  # 流体解下工序-机器可选矩阵
        # 初始化机器-工序类型边的状态矩阵（加工速率、流体处理速率、是否流体可用）
        self.proc_times = torch.zeros((num_tasks, num_machines))
        self.task_machine_edge_fluid_rate_matrix = torch.zeros((num_tasks, num_machines))
        self.task_machine_edge_fluid_available_matrix = torch.zeros((num_tasks, num_machines))
        # 依据工序-机器-时间关系填充机器-工序类型边的速率矩阵
        for i, (r, j) in enumerate(self.kind_task_tuple):
            if (r, j) in self.machine_rj_dict:
                machines = self.machine_rj_dict[(r, j)]
                for m in machines:
                    m_index = self.machine_tuple.index(m)
                    self.proc_times[i, m_index] = 1 / self.time_mrj_dict[m][(r, j)]  # 速率为时间的倒数

        # 初始化机器-工序类型邻接矩阵
        self.ope_ma_adj = torch.full((num_tasks, num_machines), False)  # 机器-工序类型邻接矩阵
        for i, (r, j) in enumerate(self.kind_task_tuple):
            if (r, j) in self.machine_rj_dict:
                machines = self.machine_rj_dict[(r, j)]
                for m in machines:
                    m_index = self.machine_tuple.index(m)
                    self.ope_ma_adj[i, m_index] = True
        # 依据工序顺序创建工序类型紧前、紧后邻接矩阵
        self.ope_pre_adj = torch.full((num_tasks, num_tasks), False)
        for r, stages in self.task_r_dict.items():
            for i in range(len(stages) - 1):
                j1 = stages[i]
                j2 = stages[i + 1]
                idx1 = sum(len(self.task_r_dict[k]) for k in self.kind_tuple if k < r) + self.task_r_dict[r].index(j1)
                idx2 = sum(len(self.task_r_dict[k]) for k in self.kind_tuple if k < r) + self.task_r_dict[r].index(j2)
                self.ope_pre_adj[idx2, idx1] = True
        self.ope_sub_adj = torch.full((num_tasks, num_tasks), False)
        for r, stages in self.task_r_dict.items():
            for i in range(len(stages) - 1):
                j1 = stages[i]
                j2 = stages[i + 1]
                idx1 = sum(len(self.task_r_dict[k]) for k in self.kind_tuple if k < r) + self.task_r_dict[r].index(j1)
                idx2 = sum(len(self.task_r_dict[k]) for k in self.kind_tuple if k < r) + self.task_r_dict[r].index(j2)
                self.ope_sub_adj[idx1, idx2] = True

        # 定义动作空间：二维离散空间，第一个元素范围 [0, num_tasks-1]；第二个元素范围 [0, num_machines-1]
        num_tasks = len(self.kind_task_tuple)
        num_machines = len(self.machine_tuple)
        self.action_dict = {action_index: (task_idx, machine_idx) for action_index, (task_idx, machine_idx) in
                            enumerate([(i, j) for j in range(num_machines) for i in range(num_tasks)])}

    def get_workload_distribution(self, orders_df):
        """根据当前订单数据计算每种工序的待处理数量"""
        # 统计每种工序的待处理数量 {(r, j): count}
        kind_task_unprocessed = {}
        kind_task_unprocessed_id_list = {}

        for _, order_row in orders_df.iterrows():
            r = order_row['product_type']
            stages = self.task_r_dict[r]
            order_id = order_row['order_id']

            # 获取当前工序阶段（如果订单还没开始，则为第一个工序）
            current_stage = order_row['current_stage']
            if current_stage is None:
                # 统计该订单尚未完成的工序
                for j in stages:
                    key = (r, j)
                    kind_task_unprocessed[key] = kind_task_unprocessed.get(key, 0) + 1
                    if key in kind_task_unprocessed_id_list.keys():
                        kind_task_unprocessed_id_list[key].append(order_id)
                    else:
                        kind_task_unprocessed_id_list[key] = [order_id]

            else:
                # 统计该订单尚未完成的工序
                for j in stages:
                    if j > current_stage:  # 只统计后续工序
                        key = (r, j)
                        kind_task_unprocessed[key] = kind_task_unprocessed.get(key, 0) + 1
                        if key in kind_task_unprocessed_id_list.keys():
                            kind_task_unprocessed_id_list[key].append(order_id)
                        else:
                            kind_task_unprocessed_id_list[key] = [order_id]

        return kind_task_unprocessed, kind_task_unprocessed_id_list

    # 函数：计算可选工序类型列表
    def get_available_rj(self, current_time, orders_df, machines_df):
        """
        计算当前时刻可选工序类型列表
        :param orders_df: 当前订单数据
        :param machines_df: 当前机器状态数据
        :param current_time: 当前仿真时间
        :return: 可选工序类型列表
        """
        kind_task_unprocessed, kind_task_unprocessed_id_list = self.get_workload_distribution(orders_df)
        # 初始化并计算各处于所有工序阶段未分配机器的订单ID列表
        kind_task_idle_id_list = {rj: [] for rj in self.kind_task_tuple}
        kind_task_idle_id_due_date_list = {rj: [] for rj in self.kind_task_tuple} # 每个可选工序类型中的订单的交期紧急度值

        for _, order_row in orders_df.iterrows():
            r = order_row['product_type']
            order_id = order_row['order_id']
            stages = self.task_r_dict[r]
            due_date = order_row['due_date']

            # 获取当前工序阶段（如果订单还没开始，则为第一个工序）
            current_stage = order_row['current_stage']
            if current_stage is None:
                # 订单还未开始，当前工序为第一个工序
                current_stage = stages[0] if stages else None
                if order_id not in kind_task_idle_id_list.get((r, current_stage), []):
                    kind_task_idle_id_list[(r, current_stage)].append(order_id)
                    kind_task_idle_id_due_date_list[(r, current_stage)].append(due_date - self.current_time)

            # 如果当前时间等于当前工序的开始时间，则当前工序ID移除
            if order_row['start_time'] is not None and order_row['start_time'] <= current_time:
                # 当前工序正在进行中，添加当前工序
                if order_id in kind_task_idle_id_list.get((r, current_stage), []):
                    kind_task_idle_id_list[(r, current_stage)].remove(order_id)
                    kind_task_idle_id_due_date_list[(r, current_stage)].remove(due_date - self.current_time)

            # 如果当前时间等于当前工序的结束时间，则添加下一个工序
            if order_row['end_time'] is not None and order_row['end_time'] <= current_time and current_stage < stages[-1]:
                # 当前工序已完成，添加下一个工序
                next_stage = stages[stages.index(current_stage) + 1]
                if order_id not in kind_task_idle_id_list.get((r, next_stage), []):
                    kind_task_idle_id_list[(r, next_stage)].append(order_id)
                    kind_task_idle_id_due_date_list[(r, next_stage)].append(due_date - self.current_time)

        # 对kind_task_idle_id_list中每种工序类型rj中的订单ID按照kind_task_idle_id_due_date_list中的值从小到大排序
        # 对kind_task_idle_id_list中每种工序类型rj中的订单ID按照交期紧急度排序
        for rj in self.kind_task_tuple:
            # 获取当前工序类型的订单ID列表和交期紧急度列表
            order_ids = kind_task_idle_id_list.get(rj, [])
            due_dates = kind_task_idle_id_due_date_list.get(rj, [])

            # 仅当有订单需要处理时才进行排序
            if len(order_ids) > 0:
                # 确保两个列表长度一致
                if len(order_ids) != len(due_dates):
                    # 记录错误但不中断程序
                    print(f"警告: 工序{rj}的订单ID与交期列表长度不一致 ({len(order_ids)} vs {len(due_dates)})")
                    continue

                # 按交期紧急度（值越小越紧急）进行排序
                # 同时排序订单ID和交期紧急度列表
                sorted_pairs = sorted(zip(due_dates, order_ids))

                # 解压排序后的结果
                sorted_due_dates, sorted_order_ids = zip(*sorted_pairs)

                # 更新字典中的列表
                kind_task_idle_id_list[rj] = list(sorted_order_ids)
                kind_task_idle_id_due_date_list[rj] = list(sorted_due_dates)

        # 生成每种工序的可选空闲机器列表
        machine_idle_list = [row['machine_id'] for _, row in machines_df.iterrows() if row['task_id'] is None]
        machine_idle_rj_dict = {rj: [] for rj in kind_task_unprocessed.keys()}
        for rj in kind_task_unprocessed.keys():
            for m in self.machine_rj_fluid_dict.get(rj, []):
                if m in machine_idle_list:
                    machine_idle_rj_dict[rj].append(m)

        # 生成当前时刻可选工序类型列表（若处于该工序阶段且未分配机器的订单数量大于0且存在可选空闲机器）
        kind_task_available_list = [rj for rj in kind_task_unprocessed if len(kind_task_idle_id_list[rj]) > 0 and len(machine_idle_rj_dict[rj]) > 0]

        # 返回可选工序类型列表
        return kind_task_available_list, kind_task_idle_id_list, machine_idle_rj_dict, kind_task_idle_id_due_date_list

    def update_fluid_state(self):
        """
        更新流体状态特征
        :param fluid_x: 流体模型解
        """
        # 流体解中每种工序的可选加工机器（分配比例为0则不可选该机器）
        self.machine_rj_fluid_dict = {rj: [] for rj in self.kind_task_unprocessed}
        for (m, rj) in self.fluid_x:
            if self.fluid_x[(m, rj)] > 1e-5:
                self.machine_rj_fluid_dict[rj].append(m)

        # 流体解中每种工序总的加工速率
        self.rate_total_rj_fluid_x = {}
        for rj in self.kind_task_unprocessed:
            self.rate_total_rj_fluid_x[rj] = sum(self.fluid_x.get((m, rj), 0) * (1 / self.time_mrj_dict.get(m, {}).get(rj, {}))
                                                 for m in self.machine_rj_dict.get(rj, []))
        # 流体解中每种工序在每个机器上的加工速率
        self.rate_m_rj_fluid_x = {}
        for (m, rj), rate in self.fluid_x.items():
            if rate > 1e-5:
                self.rate_m_rj_fluid_x[(m, rj)] = rate * (1 / self.time_mrj_dict.get(m, {}).get(rj, {}))

    def get_min_due_dates(self):
        """根据当前订单数据计算每种工序的最小截止时间"""
        min_due_dates = {}
        for _, order_row in self.orders_df.iterrows():
            r = order_row['product_type']
            stages = self.task_r_dict[r]

            # 获取当前工序阶段（如果订单还没开始，则为第一个工序）
            current_stage = order_row['current_stage']
            if current_stage is None:
                current_stage = stages[0] if stages else None
                # 计算该订单的最小截止时间
                due_date = order_row['due_date']
                for j in stages:
                    if j >= current_stage:  # 只统计当前及后续工序
                        key = (r, j)
                        min_due_dates[key] = min(min_due_dates.get(key, float('inf')), due_date)

            if current_stage is not None:
                # 计算该订单的最小截止时间
                due_date = order_row['due_date']
                for j in stages:
                    if j > current_stage:  # 只统计当前及后续工序
                        key = (r, j)
                        min_due_dates[key] = min(min_due_dates.get(key, float('inf')), due_date)

        return min_due_dates

    def solve_fluid_model(self, kind_task_min_due_date_diff, min_due_date=False):
        """
        求解“最大化所有工序处理速率/未处理量的下界”，从而最小化最大完工时间的流体模型。
        参数：
          kind_task_unprocessed：dict，形如 { rj: 待处理数量 }，rj 通常为一个元组，表示 (product_type, stage)
          current_time：当前时刻（可用于后续扩展调度逻辑，这里未使用）
        返回：
          fluid_x：dict，形如 { (machine, rj): 分配比例 }
        """
        # 创建模型
        model = gp.Model("FluidModel")

        # 创建决策变量 X[(m, rj)]，取值范围 [0, 1]，表示机器 m 在工序 rj 上分配的加工比例
        X = {}
        for m in self.machine_tuple:
            for rj in self.kind_task_unprocessed:
                if rj in self.time_mrj_dict.get(m, {}):
                    X[(m, rj)] = model.addVar(lb=0, ub=1, name=f"X_{m}_{rj[0]}_{rj[1]}")

        # 目标变量 objective 表示所有工序的交期内流体加工数量和初始待加工数量的比值 的最小值
        objective = model.addVar(lb=0, name="objective")

        # 添加机器容量约束：每台机器的总分配比例 ≤ 1
        for m in self.machine_tuple:
            assigned_tasks = [X[(m, rj)] for rj in self.kind_task_unprocessed if (m, rj) in X]
            if assigned_tasks:
                model.addConstr(gp.quicksum(assigned_tasks) <= 1, name=f"machine_capacity_{m}")

        # 为每个工序 rj 添加约束：objective ≤ (工序总处理速率/待处理数量)
        for rj, unprocessed_count in self.kind_task_unprocessed.items():
            # 如果待处理量接近0，则跳过该工序
            if unprocessed_count < 1e-5:
                print(f"警告: 工序 {rj} 的待处理数量无效: {unprocessed_count}")
                continue

            process_rates = []
            # 遍历该工序所有可加工的机器
            for m in self.machine_rj_dict.get(rj, []):
                if (m, rj) in X:
                    time_val = self.time_mrj_dict.get(m, {}).get(rj, 0)
                    if time_val > 1e-5:
                        # 单位加工速率 1/加工时间
                        rate = 1 / time_val
                        process_rates.append(X[(m, rj)] * rate)
            if min_due_date:
                total_rate = gp.quicksum(process_rates)
                # 添加约束：objective 不大于 (总处理速率 * 最小交期时长 /待处理数量)
                model.addConstr(objective <= total_rate * kind_task_min_due_date_diff[rj] / unprocessed_count, name=f"fluid_discrete_rate_{rj[0]}_{rj[1]}")
            else:
                total_rate = gp.quicksum(process_rates)
                # 添加约束：objective 不大于 (总处理速率 * 最小交期时长 /待处理数量)
                model.addConstr(objective <= total_rate / unprocessed_count, name=f"fluid_discrete_rate_{rj[0]}_{rj[1]}")

        # 添加产品内部流体完工时间的顺序约束：
        # 对于同一产品类型 r，若 (r, stage1) 是前道工序, (r, stage2) 是后道工序，
        # 则要求 fluid finish time: T₁ = Q₁ / (∑[X*(1/加工时间)]) ≥ T₂ = Q₂ / (∑[X*(1/加工时间)])
        # 等价于： (∑[X*(1/加工时间)])/Q₁ ≤ (∑[X*(1/加工时间)])/Q₂.
        for r, stages in self.task_r_dict.items():
            # 对于连续的两道工序，若两道工序都在待处理集合中，则添加约束
            for idx in range(len(stages) - 1):
                stage_prev = stages[idx]
                stage_next = stages[idx + 1]
                key_prev = (r, stage_prev)
                key_next = (r, stage_next)
                if key_prev in self.kind_task_unprocessed and key_next in self.kind_task_unprocessed:
                    unproc_prev = self.kind_task_unprocessed[key_prev]
                    unproc_next = self.kind_task_unprocessed[key_next]

                    # 构造前道工序的加工表达式
                    expr_prev = gp.quicksum(X[(m, key_prev)] * (1 / self.time_mrj_dict[m][key_prev])
                                            for m in self.machine_rj_dict.get(key_prev, [])
                                            if (m, key_prev) in X)
                    # 构造后道工序的加工表达式
                    expr_next = gp.quicksum(X[(m, key_next)] * (1 / self.time_mrj_dict[m][key_next])
                                            for m in self.machine_rj_dict.get(key_next, [])
                                            if (m, key_next) in X)
                    # 即前道工序的单位处理速率不大于后道工序，从而流体完工时间（= unprocessed/processing_rate）满足 T_prev >= T_next.
                    model.addConstr(expr_prev / unproc_prev >= expr_next / unproc_next,
                                    name=f"order_constr_{r}_{stage_prev}_{stage_next}")

        # 设置目标：最大化 objective（即尽可能让各工序在截止时间之前加工完成）
        model.setObjective(objective, GRB.MAXIMIZE)

        # 关闭输出日志，并设置求解时间限制（单位秒）
        model.Params.LogToConsole = 0
        model.Params.TimeLimit = 30

        # 求解模型
        model.optimize()

        # 检查模型是否求解成功
        if model.status != GRB.OPTIMAL:
            print("警告: 流体模型求解未达到最优解，状态码:", model.status)
            # 所有的变量解都为0，目标值也为0
            fluid_x = {}
            objective = 0
            for (m, rj) in X:
                if m in self.machine_rj_dict.get(rj, []):
                    fluid_x[(m, rj)] = 0.1
                else:
                    fluid_x[(m, rj)] = 0
            return fluid_x, objective
        else:
            # 保存变量解
            fluid_x = {}
            for (m, rj), var in X.items():
                fluid_x[(m, rj)] = var.X
            # 返回流体解和目标值
            return fluid_x, objective.X

    # 计算流体相关参数函数
    def calculate_fluid_parameters(self):
        """
        计算流体相关参数
        :param orders_df: 当前订单数据
        :param machines_df: 当前机器状态数据
        :param current_time: 当前仿真时间
        :return: 流体解和目标值
        """
        # 计算每种工序的最小截止时间（该工序未加工的订单的交期时间的最小值）
        kind_task_min_due_date = self.get_min_due_dates()

        # 计算每种工序最小交期时间和当前时间的差值（采用的最小交期时间）## 后续可考虑采用平均交期时间
        kind_task_min_due_date_diff = {key: min_due_date - self.current_time for key, min_due_date in kind_task_min_due_date.items()}

        # 求解流体模型
        fluid_x, objective = self.solve_fluid_model(kind_task_min_due_date_diff, min_due_date=False)

        # 流体解中每种工序的可选加工机器（分配比例为0则不可选该机器）
        self.machine_rj_fluid_dict = {rj: [] for rj in self.kind_task_unprocessed}
        for (m, rj) in fluid_x:
            if fluid_x[(m, rj)] > 1e-5:
                self.machine_rj_fluid_dict[rj].append(m)

        # 流体解中每种工序总的加工速率
        self.rate_total_rj_fluid_x = {}
        for rj in self.kind_task_unprocessed:
            self.rate_total_rj_fluid_x[rj] = sum(fluid_x.get((m, rj), 0) * (1 / self.time_mrj_dict.get(m, {}).get(rj, {}))
                                                 for m in self.machine_rj_dict.get(rj, []))
        # 流体解中每种工序在每个机器上的加工速率
        self.rate_m_rj_fluid_x = {}
        for (m, rj), rate in fluid_x.items():
            if rate > 1e-5:
                self.rate_m_rj_fluid_x[(m, rj)] = rate * (1 / self.time_mrj_dict.get(m, {}).get(rj, {}))

        return fluid_x, objective

    def update(self, current_time, orders_df, machines_df, fluid_x_resolve=False):
        """
        更新调度状态动态特征
        """
        self.current_time = current_time  # 更新当前时间
        self.orders_df = orders_df  # 更新订单数据
        self.machines_df = machines_df  # 更新机器状态数据
        self.kind_task_unprocessed, self.kind_task_unprocessed_id_list = self.get_workload_distribution(orders_df)
        # 如果需要重新求解流体模型，则调用求解函数
        if fluid_x_resolve:
            self.fluid_x, _ = self.calculate_fluid_parameters()
            self.update_fluid_state()  # 更新流体状态特征
        self.kind_task_available_list, self.kind_task_idle_id_list, self.machine_idle_rj_dict, self.kind_task_idle_id_due_date_list \
            = self.get_available_rj(current_time, orders_df, machines_df)
        # 更新机器状态矩阵 [[],...] 每一行：[可选工序类型数、最早可用时间、是否可用、可选连续工序类型数, 流体模型利用率]
        for i, m in enumerate(self.machine_tuple):
            machine_row = machines_df[machines_df['machine_id'] == m].iloc[0]
            self.feat_mas[i, 0] = len(self.kind_task_machine_dict[m])  # 可选工序类型数
            self.feat_mas[i, 1] = current_time if machine_row['task_id'] is None else machine_row['end_time']  # 可用时间
            self.feat_mas[i, 2] = 1 if machine_row['task_id'] is None else 0  # 是否可用
            self.feat_mas[i, 3] = len([(r, j) for (r, j) in self.kind_task_tuple if (m, (r, j)) in self.fluid_x.keys() and self.fluid_x[(m, (r, j))] > 1e-5])  # 可选连续工序类型数
            self.feat_mas[i, 4] = sum([self.fluid_x.get((m, (r, j)), 0) for (r, j) in self.kind_task_tuple if (m, (r, j)) in self.fluid_x.keys()])  # 流体模型利用率
        # 更新工序类型状态矩阵 [[],...] 每一行：[可选机器数、可选空闲机器数、平均交期时间、最小交期时间、最大交期时间、交期时间标准差、待加工总数、瞬时待加工总数、可选流体机器数、流体处理总速率]
        for i, (r, j) in enumerate(self.kind_task_tuple):
            self.feat_opes[i, 0] = len(self.machine_rj_dict.get((r, j), []))  # 可选机器数
            self.feat_opes[i, 1] = len(self.machine_idle_rj_dict.get((r, j), []))  # 可选空闲机器数
            # 计算(r, j)阶段订单的平均交期时间、最小交期时间、交期时间标准差
            if (r, j) in self.kind_task_unprocessed:
                order_ids = self.kind_task_idle_id_list.get((r, j), [])
                if order_ids:
                    delivery_times = [orders_df.loc[orders_df['order_id'] == order_id, 'due_date'].values[0] - self.current_time for order_id in order_ids]
                    self.feat_opes[i, 2] = np.mean(delivery_times)
                    self.feat_opes[i, 3] = np.min(delivery_times)
                    self.feat_opes[i, 4] = np.max(delivery_times)
                    self.feat_opes[i, 5] = np.std(delivery_times)
                else:
                    self.feat_opes[i, 2:6] = 0
            else:
                self.feat_opes[i, 2:6] = 0
            self.feat_opes[i, 6] = self.kind_task_unprocessed.get((r, j), 0)  # 待加工总数
            self.feat_opes[i, 7] = len(self.kind_task_idle_id_list.get((r, j), []))  # 瞬时待加工总数
            self.feat_opes[i, 8] = len(self.machine_rj_fluid_dict.get((r, j), []))  # 可选流体机器数
            self.feat_opes[i, 9] = self.rate_total_rj_fluid_x.get((r, j), 0)  # 流体处理总速率
            self.feat_opes[i, 10] = int(r)
            self.feat_opes[i, 11] = int(j)
        # 更新机器-工序类型边状态矩阵（流体处理速率、是否可用）
        for i, (r, j) in enumerate(self.kind_task_tuple):
            for m in self.machine_tuple:
                m_index = self.machine_tuple.index(m)
                if (m, (r, j)) in self.fluid_x:
                    self.task_machine_edge_fluid_rate_matrix[i, m_index] = self.fluid_x[(m, (r, j))]
                    self.task_machine_edge_fluid_available_matrix[i, m_index] = 1
                else:
                    self.task_machine_edge_fluid_rate_matrix[i, m_index] = 0
                    self.task_machine_edge_fluid_available_matrix[i, m_index] = 0
        # 更新离散参数下工序类型-机器可选矩阵
        for i, (r, j) in enumerate(self.kind_task_tuple):
            for m in self.machine_tuple:
                m_index = self.machine_tuple.index(m)
                if (r, j) in self.kind_task_available_list and m in self.machine_idle_rj_dict.get((r, j), []):
                    self.eligible[i, m_index] = True
                    # 更新流体解下工序类型-机器可选矩阵
                    if (m, (r, j)) in self.fluid_x and self.fluid_x[(m, (r, j))] > 1e-5:
                        self.eligible_fluid[i, m_index] = True
                    else:
                        self.eligible_fluid[i, m_index] = False
                else:
                    self.eligible[i, m_index] = False
                    self.eligible_fluid[i, m_index] = False
        # 更新self.eligible_rj_orders (该工序类型阶段是否存在待处理订单)
        for i, (r, j) in enumerate(self.kind_task_tuple):
            if len(self.kind_task_idle_id_list.get((r, j), [])) > 0:
                self.eligible_rj_orders[i] = True
            else:
                self.eligible_rj_orders[i] = False