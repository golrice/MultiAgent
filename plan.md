### **阶段一：前期准备与基础复现（第7-11周）**

**目标**：制定初步的项目计划，学习强化学习相关基础知识，初步完成论文核心算法MADDPG的代码复现

**分工**：
- 各自学习强化学习的基础知识
- 学习资料：老师的课件（一直没发）以及[动手学强化学习](https://hrl.boyuai.com/)

**关键任务**：
1. **代码复现**：
   - 实现集中式Critic网络（输入所有Agent的观测和动作）
   - 实现分散式Actor网络（仅输入局部观测）
   - 添加经验回放与目标网络机制
2. **实验复现**：
   - 使用开源的场景进行验证模型效果

3. **问题记录**：
   - 记录环境配置、超参数调试问题（如奖励函数设计、探索噪声设置）
   - 分析训练不稳定的解决方案（如梯度裁剪、目标网络更新频率）

**目标完成情况**：
- 各自完成代码复现任务【已完成】
- 进行实验验证模型效果【未完成】
- 相关问题记录【未完成】

---

### **阶段二：算法改进与对比分析（第12-14周）**

**目标**：修改网络架构并验证性能提升  

**分工**：

- **成员1（网络架构）**：通过阅读相关论文，设计改进方案（如注意力机制、图神经网络）
- **成员2（对比实验）**：设计消融实验，先验证MADDPG的效果及其优劣性，再验证改进方案的效果及其相对于MADDPG的优势（如移除Critic的全局信息）
- **成员3**：可视化实验结果，攥写项目报告相关部分

**可能的改进方向**：（由AI提供，后续会根据阅读到的论文进行调整）

1. **网络架构优化**：
   • 在Critic中引入注意力机制（替代原始MLP的全局拼接）
   • 引入LSTM或者transformer等方法处理连续动作预测
   • 尝试图神经网络（如GNN、GAT）处理局部观测
2. **训练策略改进**：
   • 添加课程学习（Curriculum Learning）逐步增加任务难度
   • 尝试混合探索策略（如NoisyNet代替OU噪声）

**目标完成情况**：

- 改进代码【未完成】
- 实验验证【未完成】
- 可视化以及写报告【未完成】

---

### **阶段三：高级应用开发（第14-16周）**

**目标**：实现多无人机协作任务（抬重物/球类比赛）  

**分工**：

- **成员1（仿真环境）**：通过AirSim/ROS Gazebo等方式搭建场景，建立智能体与场景相关的互动，并得到相关结果
- **成员2（算法设计）**：通过之前学习和研究的强化学习算法，设计多智能体协作的策略以及核心算法
- **成员3（实验验证）**：验证多智能体协作的效果，并分析原因

**关键任务**：在仿真环境下实现多无人机配合抬重物，机器人足球、篮球。

**可能的技术难点**：
- 处理连续动作空间中的高维协同（使用MADDPG的确定性策略）
- 解决稀疏奖励问题（如通过Hindsight Experience Replay）

**目标完成情况**：
• 无人机协作等场景的仿真演示视频（含轨迹可视化）【未完成】
• 任务性能指标（如抬举稳定性评分、进球成功率）

---

### **阶段四：优化与总结（第16-17周）**

**目标**：性能调优与最终报告  

**任务**：

1. **实验结果测试**：
   - 验证多智能体协作在多场景下的效果
   - 分析原因
   - 记录结果
2. **报告撰写**：
   - 总结项目经验，提出改进建议
   - 完成最终报告的撰写并提交

**分工**：
- 全员参与结果整理与交叉验证
- 指定1人负责报告主框架，其他人补充技术细节

---

### **协作工具**

1. **代码管理**：GitHub仓库分支策略：`main`（稳定版）、`dev`（开发版）、`exp/attention`（实验分支）
2. **沟通机制**：每周定期讨论学习成果以及项目进展