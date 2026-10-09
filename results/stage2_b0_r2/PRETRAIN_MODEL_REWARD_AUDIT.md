# R2运行前模型／奖励一致性审核

冻结登记07e9e5a后执行；本次审计代码07e9e5abadd1077278c2b7572956d7314f4300e6。只读历史证据，不重放、不训练。

LOCAL原生奖励表54（0-based）到达Nominal100，Validation区间**50–200**。
100→200在原文区间；共同任务reward、最多两轮预注册修复、测试集不调仍适用。
Stage2有限修复两轮，不因奖励数值核对通过而自动宣称科学GO。

## 已有详细轨迹的10／5／3m捕获

| 原组／seed／案例 | 最小距离m | 首次10m / 5m / 3m时间s | 终止 |
| --- | ---: | --- | --- |
| R1_1e-4_fixed100k/11/r1_final_s11_R01_horizontal_empty | 5.5168 | 59.646 / null / null | task_horizon |
| R1_1e-4_fixed100k/11/r1_final_s11_R02_deeper_empty | 8.6166 | 62.142 / null / null | task_horizon |
| R1_1e-4_fixed100k/11/r1_final_s11_R03_shallower_empty | 4.9152 | 67.528 / 166.643 / null | task_horizon |
| V1_3e-4_Val_index012100k_HISTORICAL/11/val_s11_m100000_obstacle_free_i0 | 5.8052 | 58.275 / null / null | task_horizon |
| V1_3e-4_Val_index012100k_HISTORICAL/11/val_s11_m100000_obstacle_free_i1 | 7.2755 | 72.189 / null / null | task_horizon |
| V1_3e-4_Val_index012100k_HISTORICAL/11/val_s11_m100000_obstacle_free_i2 | 40.1925 | null / null / null | operational_boundary_failure |
| R1_1e-4_fixed100k/22/r1_final_s22_R01_horizontal_empty | 11.4118 | null / null / null | task_horizon |
| R1_1e-4_fixed100k/22/r1_final_s22_R02_deeper_empty | 14.2875 | null / null / null | task_horizon |
| R1_1e-4_fixed100k/22/r1_final_s22_R03_shallower_empty | 17.9048 | null / null / null | task_horizon |
| V1_3e-4_Val_index012100k_HISTORICAL/22/val_s22_m100000_obstacle_free_i0 | 2.4338 | 42.615 / 46.259 / 48.024 | operational_boundary_failure |
| V1_3e-4_Val_index012100k_HISTORICAL/22/val_s22_m100000_obstacle_free_i1 | 11.7159 | null / null / null | operational_boundary_failure |
| V1_3e-4_Val_index012100k_HISTORICAL/22/val_s22_m100000_obstacle_free_i2 | 5.5629 | 45.983 / null / null | operational_boundary_failure |
| R1_1e-4_fixed100k/33/r1_final_s33_R01_horizontal_empty | 3.3802 | 61.298 / 70.948 / null | task_horizon |
| R1_1e-4_fixed100k/33/r1_final_s33_R02_deeper_empty | 11.1390 | null / null / null | task_horizon |
| R1_1e-4_fixed100k/33/r1_final_s33_R03_shallower_empty | 20.8753 | null / null / null | operational_boundary_failure |
| V1_3e-4_Val_index012100k_HISTORICAL/33/val_s33_m100000_obstacle_free_i0 | 2.6036 | 71.072 / 78.446 / 81.615 | task_horizon |
| V1_3e-4_Val_index012100k_HISTORICAL/33/val_s33_m100000_obstacle_free_i1 | 8.5744 | 72.288 / null / null | task_horizon |
| V1_3e-4_Val_index012100k_HISTORICAL/33/val_s33_m100000_obstacle_free_i2 | 5.4635 | 66.704 / null / null | task_horizon |

详细状态、Body目标、yaw/pitch误差、名义/物理指令与实际响应在同名JSON。
时间是已记录0.05s执行线性分段的首次球进入；不是重新仿真。
V1补充有明确历史身份；R1 Val控制节点缺8D，不把V1状态冒充R1。

## 同行为奖励100／200离线重计

完整episode最大独立残差3.13e-13，
运行前固定float64核对容差1e-08。
失败其他分量不变且差0，成功终点仅额外100；分母按训练／验证点／seed独立。
成功episode全部transition与单个成功终点分别计数，见JSON，不推造Replay抽样频率。
折扣以首个正式reward为gamma^0；进展与终点分别计，不与未折扣回报混用。
原顺序最早实际成功与最早有保留节点的成功分开；缺失详细轨迹写null。
奖励200的离线重计保持行为固定，不能证明它会改变学习或保证到达。

## 模型与停止条件

12个R1小模型现场存在；沿用上一轮真实weights_only身份/有限性审计，
本轮不重新加载网络推断、不计算梯度；不能把历史审计称成本轮模型运算。
此范围未发现新关键L1或实质协议冲突；并非全域无缺陷证明。
本工具新环境步／重放／训练／前向／反向均0。
训练授权和科学Gate由主任务按冻结R2登记处理，本工具不启动训练。
