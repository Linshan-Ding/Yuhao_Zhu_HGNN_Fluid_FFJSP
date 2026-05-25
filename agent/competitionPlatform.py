import os
import time
import pandas as pd
import numpy as np
from collections import defaultdict
import copy
import random
import matplotlib.pyplot as plt

class CompetitionPlatform:
    def __init__(self):
        # 初始仿真时间（秒）
        self.current_time = 0 # 当前时间点
        self.last_time = 0 # 局部调度方案开始时间点
        self.next_time = 0 # 算法给出的下一个时间点
        self.orders_df = None # 当前订单状态信息
        self.orders_df_all = None # 所有订单
        self.orders_finished_df = None # 当前时刻调度完工的订单
        self.machines_df = None # 当前机器状态信息
        self.orders_arrival_count = 0 # 订单到达数量
        self.order_completed_count = 0 # 订单按时完成数量
        self.fulfillment_rate = 0.0  # 当前时刻订单达成率

        # 用于记录调度计划（由各决策点生成的任务安排）
        self.schedule_data = []
        # 保存各机器的历史作业记录（字典：machine_id -> list of records）
        self.machine_record = defaultdict(list)
        # 用于存放实例文件解析后的各类数据
        self.instance_data = None

    def parse_instance(self, path: str) -> dict:
        """
        解析测试实例文件。按照说明，文件中：
          第1行：n q m  —— 产品数、生产阶段数、总机器数
          第2行：各阶段机器数划分（例如：3 3 表示第一阶段3台，第二阶段3台）
          第3~(n+2) 行：n*m 的加工时长矩阵，每行对应一种产品在各台设备上的加工时长
          第(n+3) 行：订单ID（空格分割）
          第(n+4) 行：产品类型ID（订单所属产品类型，0...n-1）
          第(n+5) 行：订单到达时间（整数列表）
          第(n+6) 行：订单交期（整数列表）
        返回一个字典，包含 MBOM、订单和设备状态 DataFrame 等。
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() != '']

        if len(lines) < 6:
            raise ValueError("测试实例文件行数不足，检查文件格式。")

        # 解析第一行：产品数、生产阶段数、总机器数
        parts = lines[0].split()
        n = int(parts[0])
        q = int(parts[1])
        m = int(parts[2])

        # 解析第二行：各阶段机器数划分
        machine_div = list(map(int, lines[1].split()))
        if len(machine_div) != q:
            raise ValueError("机器划分数与生产阶段数不匹配。")

        # 解析加工时长矩阵：接下来 n 行，每行 m 个整数
        pt_matrix = []
        for i in range(n):
            pt_line = list(map(int, lines[2 + i].split()))
            if len(pt_line) != m:
                raise ValueError(f"第 {i + 3} 行加工时长数据不完整，应有 {m} 个数。")
            pt_matrix.append(pt_line)
        pt_matrix = np.array(pt_matrix)

        # 解析订单信息
        orders_line = lines[2 + n]
        product_line = lines[3 + n]
        arrival_line = lines[4 + n]
        due_line = lines[5 + n]

        job_ids = list(map(int, orders_line.split()))
        product_ids = list(map(int, product_line.split()))
        arrival_times = list(map(int, arrival_line.split()))
        due_dates = list(map(int, due_line.split()))

        if not (len(job_ids) == len(product_ids) == len(arrival_times) == len(due_dates)):
            raise ValueError("订单信息行中各列长度不一致。")

        # 构造制造BOM（MBOM），表中每一条记录代表：某产品在某生产阶段、某台设备上的处理时间
        mbom_list = []
        # 产品类型假定为 0 ~ n-1
        machine_offset = 0
        for stage_idx in range(q):
            machine_num = machine_div[stage_idx]
            for product in range(n):
                for j in range(machine_num):
                    machine_id = machine_offset + j
                    process_time = pt_matrix[product][machine_id]
                    mbom_list.append({
                        'product_type': str(product),  # 产品类型编号
                        'stage': str(stage_idx),  # 生产阶段
                        'machine_id': str(machine_id),  # 设备编号
                        'process_time(s)': process_time  # 加工时长
                    })
            machine_offset += machine_num

        mbom_df = pd.DataFrame(mbom_list)

        # 构造订单 DataFrame；字段包括订单 ID、产品类型、到达时间、交期及初始状态信息
        orders_list = []
        for i, order_id in enumerate(job_ids):
            orders_list.append({
                'order_id': str(order_id),
                'product_type': str(product_ids[i]),
                'arrival_time': arrival_times[i],
                'due_date': due_dates[i],
                'current_stage': None,
                'assigned_machine': None,
                'start_time': None,
                'end_time': None,
                'fulfillment_rate': 0.0  # 初始达成率
            })
        orders_df = pd.DataFrame(orders_list)
        orders_df.sort_values('arrival_time', inplace=True)
        orders_df.reset_index(drop=True, inplace=True)

        # 构造机器状态 DataFrame，初始均为空闲状态
        machine_ids = list(range(m))
        machines_list = []
        for mid in machine_ids:
            machines_list.append({
                'task_id': None,
                'start_time': None,
                'end_time': None
            })
        machines_df = pd.DataFrame(machines_list)

        self.instance_data = {
            'n': n,
            'q': q,
            'm': m,
            'machine_div': machine_div,
            'mbom_df': mbom_df,
            'orders_df': orders_df,
            'machines_df': machines_df,
        }
        # 仿真从时间0开始
        self.current_time = 0
        # 清空之前的调度计划数据
        self.schedule_data = []
        return self.instance_data

    def getMBOM(self) -> pd.DataFrame:
        """返回制造BOM信息"""
        if self.instance_data is None:
            raise ValueError("请先调用 parse_instance() 读取实例文件。")
        return self.instance_data['mbom_df']

    def getOrders(self, unfinished=True) -> pd.DataFrame:
        """
        将self.current_time 至 self.next_time期间到达的新订单填充进self.orders_df
        """
        if unfinished:
            if self.instance_data is None:
                raise ValueError("请先调用 parse_instance() 读取实例文件。")
            orders_df = copy.deepcopy(self.instance_data['orders_df'])
            # self.current_time 至 self.next_time期间到达的新订单以扩充self.orders_df
            new_orders_df = orders_df[(orders_df['arrival_time'] <= self.next_time) & (orders_df['arrival_time'] > self.last_time)]
            self.orders_arrival_count += new_orders_df.shape[0]
            self.current_time = self.next_time
            # 用new_orders_df 扩充 self.orders_df
            # 若有新订单，则合并到当前已激活的订单中
            if not new_orders_df.empty:
                if self.orders_df is None or self.orders_df.empty:
                    self.orders_df = new_orders_df.copy().reset_index(drop=True)
                else:
                    self.orders_df = pd.concat([self.orders_df, new_orders_df], ignore_index=True)
                self.orders_df['fulfillment_rate'] = self.fulfillment_rate
            return copy.deepcopy(self.orders_df)
        else:
            return copy.deepcopy(self.orders_df_all)

    def getCurrentMachineStatus(self) -> pd.DataFrame:
        """
        返回当前时刻所有机器的状态信息。
        在本示例中，直接返回 instance_data 中的 machines_df，
        若需要更精细模拟各机器状态随时间更新，可在此处添加逻辑。
        """
        if self.instance_data is None:
            raise ValueError("请先调用 parse_instance() 读取实例文件。")
        return copy.deepcopy(self.machines_df)

    def getSimulationTime(self) -> float:
        """返回当前仿真时间（秒）"""
        return self.current_time

    def getMachineRecord(self) -> dict:
        """返回各机器的历史作业记录字典：machine_id -> DataFrame"""
        records = {}
        for mid, rec_list in self.machine_record.items():
            records[mid] = pd.DataFrame(rec_list)
        return records

    def run_simulation(self, path: str, algorithm_module, isTimeout: bool = True) -> dict:
        """
        运行仿真案例。步骤：
          1. 解析实例文件
          2. 循环执行决策：在每个决策点，
             · 调用传入的算法模块（要求实现 generate_schedule(platform) 方法，
               返回的调度方案 schedule_df 要包含以下列：
                   - task_id：格式 "order_id-process_id"
                   - machine_id
                   - start_time
             · 对调度方案进行验证：若存在延期订单（即任务安排的开始时间大于对应订单交期），立即报错
             · 更新调度计划和当前时间
             · 模拟订单完成（本示例中用简单规则标记订单到达后立即完成）。
          3. 返回仿真结果，包含订单达成率和调度计划。
        """
        # 解析实例文件（同时初始化 instance_data、machines_df、orders_df、MBOM 等）
        self.parse_instance(path)
        orders_df = copy.deepcopy(self.instance_data['orders_df'])
        self.orders_df_all = copy.deepcopy(self.instance_data['orders_df'])
        self.orders_finished_df = pd.DataFrame()
        self.orders_arrival_count = 0
        self.order_completed_count = 0
        self.fulfillment_rate = 0.0
        mbom_df = copy.deepcopy(self.instance_data['mbom_df'])
        # 定义仿真结束时刻（例如：最大交期+一定缓冲）
        max_time = orders_df['due_date'].max() + 10000
        begin_time = orders_df['arrival_time'].min()
        # 初始化self.orders_df和self.machines_df
        self.orders_df = pd.DataFrame(columns=orders_df.columns)
        self.machines_df = copy.deepcopy(self.instance_data['machines_df'])
        self.next_time = begin_time  # 初始化当前时间

        # 模拟循环直到所有订单都处理完毕或达到 max_time
        while self.current_time < max_time:
            # 调用算法模块生成调度方案，算法模块通过 platform 提供的接口获取当前状态
            time_start = time.process_time()
            schedule_df, self.next_time = algorithm_module.generate_schedule(self)
            time_end = time.process_time()
            time_cost = time_end - time_start
            if isTimeout and time_cost > 30:
                raise ValueError(f"算法运行超时")
            self.last_time = self.current_time  # 跟新局部调度方案开始点
            # 遍历调度方案，验证是否存在延期订单（延期：任务安排的 end_time 大于对应订单交期）
            for idx, task in schedule_df.iterrows():
                # task_id 格式为 "order_id-process_id"，解析订单号 # 提取任务的订单编号、产品类型、工序阶段、选择的机器和加工时间
                try:
                    order_id = str(task['task_id']).split('-')[1]
                    product_type = orders_df.loc[orders_df['order_id'] == order_id, 'product_type'].values[0]
                    stage_id = str(task['task_id']).split('-')[2]
                    machine_id = task['machine_id']
                    start_time = task['start_time']
                    # 获取加工时间
                    mask = (
                            (mbom_df['product_type'] == product_type) &
                            (mbom_df['stage'] == stage_id) &
                            (mbom_df['machine_id'] == machine_id)
                    )
                    process_time = mbom_df.loc[mask, 'process_time(s)'].values[0]
                except Exception as e:
                    raise ValueError(f"无法解析 task_id {task['task_id']}，确保格式为 'order_id-process_id'。")
                self.current_time = start_time
                # 如果当前时间大于指定的下一个决策点则结束任务排产
                if self.current_time >= self.next_time:
                    break
                # 查找该订单的交期
                order_row = orders_df[orders_df['order_id'] == order_id]
                if order_row.empty:
                    raise ValueError(f"调度方案中订单 {order_id} 不存在或已延期。")
                due_date = order_row.iloc[0]['due_date']
                if start_time + process_time > due_date:
                    raise ValueError(f"调度方案错误：订单 {order_id} 的阶段为{stage_id}，工件类型为{product_type}，"
                                     f"选择的机器为{machine_id}，任务结束时间 {start_time + process_time} 超过交期 {due_date}。")

                # 更新机器状态信息
                self.machines_df.loc[int(machine_id), 'task_id'] = task['task_id']
                self.machines_df.loc[int(machine_id), 'start_time'] = start_time
                self.machines_df.loc[int(machine_id), 'end_time'] = start_time + process_time

                # 更新订单状态：更新当前工序，记录开始和预期结束时间
                self.orders_df.loc[self.orders_df['order_id'] == order_id, 'assigned_machine'] = machine_id
                self.orders_df.loc[self.orders_df['order_id'] == order_id, 'current_stage'] = stage_id
                self.orders_df.loc[self.orders_df['order_id'] == order_id, 'start_time'] = start_time
                self.orders_df.loc[self.orders_df['order_id'] == order_id, 'end_time'] = start_time + process_time

                # 剔除已完成订单
                if int(stage_id) + 1 == self.instance_data['q']:
                    finished_order = self.orders_df[self.orders_df['order_id'] == order_id].copy()
                    if self.orders_finished_df is None or self.orders_finished_df.empty:
                        self.orders_finished_df = finished_order.reset_index(drop=True)
                    else:
                        self.orders_finished_df = pd.concat([self.orders_finished_df, finished_order], ignore_index=True)
                    self.orders_df = self.orders_df[self.orders_df['order_id'] != order_id]  # 剔除完工订单
                    self.order_completed_count += 1
                    self.fulfillment_rate = self.order_completed_count / (self.orders_arrival_count + 1e-8)
                    self.orders_finished_df['fulfillment_rate'] = self.fulfillment_rate
                # 更新完工率
                self.orders_df_all['fulfillment_rate'] = self.fulfillment_rate
                # 记录任务
                self.machine_record[machine_id].append(task)
                # 更新调度方案
                task['end_time'] = start_time + process_time
                self.schedule_data.append(task)

            # 移动当前时间至给定的决策点
            self.current_time = self.next_time
            # 更新机器状态
            for _, row in self.machines_df.iterrows():
                if row['end_time'] is not None and row['end_time'] <= self.current_time:
                    # 机器空闲，更新状态, 更新machines_df
                    self.machines_df.loc[row.name, 'task_id'] = None
                    self.machines_df.loc[row.name, 'start_time'] = None
                    self.machines_df.loc[row.name, 'end_time'] = None

        sim_result = {
            'fulfillment_rate': self.fulfillment_rate,
            'schedule': pd.DataFrame(self.schedule_data)
        }
        return sim_result

    def getGantta(self, startTime, endTime):
        """
        生成并显示甘特图。甘特图横坐标为时间，纵坐标为各台机器，
        同一订单中不同工序采用相同颜色。

        参数：
            startTime: 甘特图显示的起始时间
            endTime:   甘特图显示的结束时间
        """
        # 将调度计划数据转换为DataFrame
        if not self.schedule_data:
            print("暂无调度任务数据，无法生成甘特图。")
            return
        schedule_df = pd.DataFrame(self.schedule_data)

        # 过滤出起始时间在 [startTime, endTime] 区间的任务
        schedule_df = schedule_df[(schedule_df['start_time'] >= startTime) & (schedule_df['end_time'] <= endTime)]
        if schedule_df.empty:
            print("指定时间区间内暂无任务。")
            return

        # 为同一订单统一颜色。订单号在 task_id 中格式假定为 "xxx-订单ID-工序ID"
        # 为了获取订单号，可按“-”分隔字符串。构造订单-颜色映射字典
        order_ids = []
        for t in schedule_df['task_id']:
            parts = str(t).split('-')
            if len(parts) >= 2:
                # 此处假定订单号位于第二个字段，且均为字符串
                order_ids.append(parts[1])
        unique_orders = list(set(order_ids))

        # 预先生成颜色字典（这里随机生成或使用固定颜色列表）
        cmap = plt.get_cmap('tab20')
        order_color = {}
        for idx, oid in enumerate(unique_orders):
            order_color[oid] = cmap(idx % 20)  # 使用tab20，最多20种颜色

        # 按各机器绘图
        machines = sorted(schedule_df['machine_id'].unique(), key=lambda x: int(x))
        machine_y = {mid: i * 10 for i, mid in enumerate(machines)}  # 每台机器在图中对应一个 y 位置

        fig, ax = plt.subplots(figsize=(25, 0.5 * len(machines) + 2))

        # 绘制每个任务为水平条
        for idx, task in schedule_df.iterrows():
            task_id = str(task['task_id'])
            machine_id = str(task['machine_id'])
            start = task['start_time']
            # 根据 MBOM 信息或任务中的 end_time直接获取加工时长
            end = task.get('end_time', start)  # 若未提供，则end_time为start（0时长）
            duration = end - start
            # 提取对应订单ID（假定 task_id 格式为 "xxx-订单ID-工序ID"）
            parts = task_id.split('-')
            if len(parts) >= 2:
                oid = parts[1]
            else:
                oid = task_id
            color = order_color.get(oid, (random.random(), random.random(), random.random()))

            # 绘制任务条，y坐标根据机器位置，任务条高度固定
            ax.broken_barh([(start, duration)], (machine_y[machine_id], 6),
                           facecolors=color, edgecolors='black', label=oid)
            # 在条中间显示任务ID（可以选择不显示或显示订单号+工序）
            ax.text(start + duration / 2, machine_y[machine_id] + 4, task_id,
                    ha='center', va='center', color='white', fontsize=8)

        # 设置坐标轴
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Machine")
        ax.set_title(f"Gantt: [{startTime}, {endTime}]")
        ax.set_yticks([machine_y[mid] + 4 for mid in machines])
        ax.set_yticklabels(machines)
        ax.set_xlim(startTime, endTime)
        ax.grid(True)
        plt.savefig(os.path.join('../result', 'gantt_chart.png'))
