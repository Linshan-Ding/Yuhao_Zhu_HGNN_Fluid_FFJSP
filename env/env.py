from env.env_state import SchedulingState


class SchedulingEnv:
    """
    调度问题 RL 环境
    - 观察空间包括机器状态矩阵和工序状态矩阵（可扩展为 Dict 格式）
    - 动作定义为 (task_idx, machine_idx) 对，其中 task_idx 表示选中的工序类型（即 (r,j)）， machine_idx 表示调度的机器
    - step() 函数根据动作更新订单、机器状态，并返回新的观察值、奖励、done 标志和额外信息
    """
    def __init__(self):
        super(SchedulingEnv, self).__init__()
        # 初始化参数
        self.mbom_df = None  # 物料清单 DataFrame
        self.orders_df = None # 订单 DataFrame
        self.machines_df = None # 机器 DataFrame
        self.current_time = None # 当前仿真时间
        # 初始化状态对象
        self.state = None  # 当前状态
        self.action = None  # 当前动作
        self.next_state = None  # 下一个状态
        self.reward = None  # 当前奖励
        self.done = False  # 是否结束标志

        # 当前决策点到达订单总数
        self.oder_arrival_count = None
        # 当前决策点按时完工订单总数和丢弃订单总数
        self.order_completion_count = 0
        self.order_completion_count_last = 0
        self.order_discard_count = 0
        self.order_discard_count_last = 0
        # 当前决策点订单达成率和丢弃率
        self.fulfillment_rate = 0.0
        self.discard_rate = 0.0
        self.discard_penalty = 1.0
        self.progress_reward_coeff = 0.0
        self.tardiness_penalty = 0.0
        self.dense_progress_reward = 0.0
        self.total_orders = 1
        # 上一决策点订单达成率和丢弃率
        self.last_fulfillment_rate = 0.0
        self.last_discard_rate = 0.0
        # 丢弃的订单ID列表
        self.discard_order_ids = []  # 用于记录丢弃的订单ID
        self.schedule_data = []  # 用于记录调度计划数据
        self.finished_order_ids = [] # 用于记录已完工订单列表

        # 定义动作空间：二维离散空间，第一个元素范围 [0, num_tasks-1]；第二个元素范围 [0, num_machines-1]
        self.num_tasks = None # 工序类型数量
        self.num_machines = None # 机器数量
        self.action_dict = None # 动作空间

    def reset(self, mbom_df, orders_df, machines_df, current_time):
        """
        重置环境到初始状态，返回初始观察值
        """
        self.current_time = current_time
        self.orders_df = orders_df.copy(deep=True)
        self.machines_df = machines_df.copy(deep=True)
        self.mbom_df = mbom_df.copy(deep=True)

        # 初始化订单到达计数
        self.oder_arrival_count = len(self.orders_df)
        # 当前决策点按时完工订单总数和丢弃订单总数
        self.order_completion_count = 0
        self.order_completion_count_last = 0
        self.order_discard_count = 0
        self.order_discard_count_last = 0
        # 当前决策点订单达成率和丢弃率
        self.fulfillment_rate = 0.0
        self.discard_rate = 0.0
        self.discard_penalty = getattr(self, 'discard_penalty', 1.0)
        self.progress_reward_coeff = getattr(self, 'progress_reward_coeff', 0.0)
        self.tardiness_penalty = getattr(self, 'tardiness_penalty', 0.0)
        self.dense_progress_reward = getattr(self, 'dense_progress_reward', 0.0)
        self.total_orders = max(int(getattr(self, 'total_orders', len(self.orders_df))), 1)
        # 上一决策点订单达成率和丢弃率
        self.last_fulfillment_rate = 0.0
        self.last_discard_rate = 0.0
        # 丢弃的订单ID列表
        self.discard_order_ids = []  # 用于记录丢弃的订单ID
        self.schedule_data = []  # 用于记录调度计划数据

        # 重置强化学习元素
        self.state = None  # 当前状态
        self.action = None  # 当前动作
        self.next_state = None  # 下一个状态
        self.reward = None  # 当前奖励
        self.done = False  # 是否结束标志
        # 更新状态对象
        self.state = SchedulingState()
        self.state.initialize_from_mbom(self.mbom_df)
        self.state.fluid_recompute_interval = int(getattr(self, "fluid_recompute_interval", self.state.fluid_recompute_interval))
        self.state.fluid_time_limit = int(getattr(self, "fluid_time_limit", self.state.fluid_time_limit))
        self.state.fluid_enabled = bool(getattr(self, "fluid_enabled", self.state.fluid_enabled))
        self.state.fluid_threads = int(getattr(self, "fluid_threads", self.state.fluid_threads))
        self.state.fluid_profile = bool(getattr(self, "fluid_profile", self.state.fluid_profile))
        self.state.order_top_k = int(getattr(self, "order_top_k", self.state.order_top_k))
        self.state.update(self.current_time, self.orders_df, self.machines_df, fluid_x_resolve = True)
        # 定义动作空间：二维离散空间，第一个元素范围 [0, num_tasks-1]；第二个元素范围 [0, num_machines-1]
        self.num_tasks = len(self.state.kind_task_tuple)
        self.num_machines = len(self.state.machine_tuple)
        self.action_dict = {action_index: (task_idx, machine_idx) for action_index, (task_idx, machine_idx) in
                            enumerate([(i, j) for j in range(self.num_machines) for i in range(self.num_tasks)])}
        return self.state

    def step(self, action):
        """
        执行一个调度动作，更新状态并返回新的观察值、奖励、done 标志和额外信息
        """
        # 动作数据类型转换
        action1 = action[0]
        action2 = action[1]
        # 选择工序类型和机器
        task_idx, machine_idx = self.action_dict.get(action1, (None, None))
        chosen_task = self.state.kind_task_tuple[task_idx]  # (r,j)
        chosen_machine = self.state.machine_tuple[machine_idx] # (m)

        # 选择该工序类型阶段的订单
        selected_order_id = self.state.kind_task_idle_id_list[chosen_task][action2]

        # 更新机器状态：记录分配的任务ID，起始时间及结束时间（结束时间 = 当前时间 + 加工时间）
        proc_time = self.state.time_mrj_dict.get(chosen_machine, {}).get(chosen_task, None)
        task_id = 'M-' + selected_order_id + '-' + str(chosen_task[1])
        self.machines_df.loc[self.machines_df['machine_id'] == chosen_machine, 'task_id'] = task_id
        self.machines_df.loc[self.machines_df['machine_id'] == chosen_machine, 'start_time'] = self.current_time
        self.machines_df.loc[self.machines_df['machine_id'] == chosen_machine, 'end_time'] = self.current_time + proc_time

        # 更新订单状态：更新当前工序，记录开始和预期结束时间
        self.orders_df.loc[self.orders_df['order_id'] == selected_order_id, 'assigned_machine'] = chosen_machine
        self.orders_df.loc[self.orders_df['order_id'] == selected_order_id, 'current_stage'] = chosen_task[1]
        self.orders_df.loc[self.orders_df['order_id'] == selected_order_id, 'start_time'] = self.current_time
        self.orders_df.loc[self.orders_df['order_id'] == selected_order_id, 'end_time'] = self.current_time + proc_time


        # 初始化丢弃订单标识
        order_discard = False
        # 判断该订单是否延期
        order_due_time = self.orders_df.loc[self.orders_df['order_id'] == selected_order_id, 'due_date'].values[0]
        tardiness = max(0, self.current_time + proc_time - order_due_time)
        if self.current_time + proc_time > order_due_time:
            # 订单延期，更新丢弃订单计数
            self.order_discard_count += 1
            self.discard_rate = self.order_discard_count / (self.oder_arrival_count + 1e-6)
            self.discard_order_ids.append(selected_order_id)  # 记录丢弃的订单ID
            # 从self.orders_df中删除该订单
            self.orders_df = self.orders_df[self.orders_df['order_id'] != selected_order_id]
            self.machines_df.loc[self.machines_df['machine_id'] == chosen_machine, 'task_id'] = None
            self.machines_df.loc[self.machines_df['machine_id'] == chosen_machine, 'start_time'] = None
            self.machines_df.loc[self.machines_df['machine_id'] == chosen_machine, 'end_time'] = None
            order_discard = True
        elif self.orders_df.loc[self.orders_df['order_id'] == selected_order_id, 'current_stage'].values[0] == self.state.task_r_dict[chosen_task[0]][-1]:
            self.orders_df = self.orders_df[self.orders_df['order_id'] != selected_order_id]
            # 如果该工序为订单的最后一个工序，则更新订单状态为完成
            self.order_completion_count += 1
            self.finished_order_ids.append(selected_order_id)
            self.fulfillment_rate = self.order_completion_count / (self.oder_arrival_count + 1e-6)

        # 更新调度计划数据和丢弃后的状态
        if not order_discard:  # 如果没有丢弃订单
            self.state.update(self.current_time, self.orders_df, self.machines_df, fluid_x_resolve=True)
            # 更新调度计划数据
            self.schedule_data.append({
                'task_id': 'M-' + selected_order_id + '-' + str(chosen_task[1]),
                'machine_id': str(chosen_machine),
                'start_time': self.current_time
            })
        else:
            self.state.update(self.current_time, self.orders_df, self.machines_df, fluid_x_resolve=True)

        # 所有空闲机器无可选订单，移动时钟到下一个机器空闲点
        while len(self.state.kind_task_available_list) == 0 and sum(self.state.kind_task_unprocessed.values()) > 0:
            # 获取所有机器的空闲时间点
            next_idle_time = min([row['end_time'] for _, row in self.machines_df.iterrows() if row['end_time'] is not None
                                  and row['end_time'] > self.current_time], default=None)
            if next_idle_time is not None and next_idle_time >= self.current_time:
                self.current_time = next_idle_time
                # print(f"移动时钟到下一个机器空闲点: {self.current_time} 秒")
            else:
                print('next_idle_time', next_idle_time)
                raise ValueError("--------------移动时钟报错---------------")
            # 更新机器状态、订单状态
            for _, row in self.machines_df.iterrows():
                if row['end_time'] is not None and row['end_time'] == self.current_time:
                    # 机器空闲，更新状态, 更新machines_df
                    self.machines_df.loc[row.name, 'task_id'] = None
                    self.machines_df.loc[row.name, 'start_time'] = None
                    self.machines_df.loc[row.name, 'end_time'] = None
            # 更新状态对象
            self.state.update(self.current_time, self.orders_df, self.machines_df, fluid_x_resolve=True)

        # 计算奖励：假设奖励为相邻决策点的订单达成率
        reward_rate = self.fulfillment_rate - self.last_fulfillment_rate
        self.last_fulfillment_rate = self.fulfillment_rate

        # 计算奖励: 相邻决策点丢弃订单数的差值
        reward_discard = self.order_discard_count - self.order_discard_count_last
        self.order_discard_count_last = self.order_discard_count

        # 计算奖励：相邻决策点按时达成的订单数差值
        reward_completed = self.order_completion_count - self.order_completion_count_last
        self.order_completion_count_last = self.order_completion_count
        total_orders = max(float(self.total_orders), 1.0)

        self.reward = (
            reward_completed / total_orders
            - self.discard_penalty * reward_discard / total_orders
            - self.tardiness_penalty * tardiness
            + (self.progress_reward_coeff + self.dense_progress_reward) * reward_rate
        )

        # 判断仿真是否结束
        if sum(self.state.kind_task_unprocessed.values()) == 0:
            self.done = True

        return self.state, self.reward, self.done
