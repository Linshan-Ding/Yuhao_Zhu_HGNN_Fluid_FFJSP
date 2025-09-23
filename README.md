# DOCS2025
DOCS2025第一届“辽河杯”数据驱动的复杂系统优化算法竞赛：柔性流水车间动态调度问题

更新记录：

6.25   新增数字签名

6.26   更新competitionPlatform，修改 run_simulation函数，仅在 isTimeout==True 时，输出数字签名

6.26   更新competitionPlatform，修复“最后一个订单到达时，无法动态调度”的问题。

6.27   更新competitionPlatform，修复Linux平台读取包错误的问题。

6.30   更新competitionPlatform，修复“getOrders函数在设置了only_unfinished=False时报错”的问题。  
       当only_unfinished=False时，已完成订单的start_time和end_time,即为订单的第一阶段的开始生产时间和最后阶段的结束时间。

7.1    更新competitionPlatform、 simulation、infobuilder、calendarinfo，修复“某些时刻，machines_status中处理时间与MBOM中不一致”的问题
       增加同一工件只有前阶段完成，后阶段才可以处理。修复“同一工件多阶段可并行处理”的问题

7.7    修正“100%达成率时，无法输出数字签名”的问题。

7.11   增加“超交期的检测”。
