# import os
# # 设置环境变量，允许 OpenMP 库重复加载
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import torch
# import numpy as np
# import gymnasium as gym
# import os
# import time
# from collections import deque
# from torch.utils.tensorboard import SummaryWriter

# # --- 导入自定义模块 ---
# # 请确保这些文件在同一目录下
# from uncertain_env import UncertainComplexEnv # 假设环境文件名
# from CNN_GRU_policy import RobustGRPOPolicy
# from ds_grpo import DSGRPO

# # ==========================================
# # 0. 辅助工具: 动作空间缩放包装器
# # ==========================================
# class ActionScalingWrapper(gym.ActionWrapper):
#     """
#     将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作
#     v: [-1, 1] -> [-1, 2]
#     w: [-1, 1] -> [-1, 1]
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         # 定义新的动作空间为 [-1, 1]
#         self.action_space = gym.spaces.Box(
#             low=-1.0, high=1.0, shape=(2,), dtype=np.float32
#         )

#     def action(self, action):
#         # 神经网络输出的 raw action
#         v_scaled, w_scaled = action[0], action[1]
        
#         # 线性映射
#         # v: [-1, 1] -> [-1, 2]
#         v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
#         # w: [-1, 1] -> [-1, 1] (无需变化)
#         w_real = w_scaled
        
#         return np.array([v_real, w_real], dtype=np.float32)

# # ==========================================
# # 1. 环境工厂函数 (用于并行采样)
# # ==========================================
# def make_env(env_id, seed):
#     def _thunk():
#         # 初始化不确定性环境
#         env = UncertainComplexEnv()
#         # 包装动作缩放
#         env = ActionScalingWrapper(env)
#         # 设置随机种子
#         env.reset(seed=seed)
#         return env
#     return _thunk

# # ==========================================
# # 2. 训练主循环
# # ==========================================
# def train():
#     # --- 超参数设置 ---
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # 环境参数
#     NUM_ENVS = 16        # 并行环境数量 (对应 GRPO 的 Group Size 上限)
#     MAX_STEPS = 500000   # 总训练步数
#     ROLLOUT_STEPS = 64   # 每次更新前，每个环境采样的步数 (序列长度)
    
#     # DAPO 参数
#     G_MIN = 4            # 最小动态组大小
#     G_MAX = 16           # 最大动态组大小 (不能超过 NUM_ENVS)

#     # 算法参数
#     LR = 3e-4
#     HIDDEN_DIM = 256
    
#     # 路径设置
#     exp_name = f"DS_GRPO_Uncertainty_{int(time.time())}"
#     log_dir = f"runs/{exp_name}"
#     model_dir = f"models/{exp_name}"
#     os.makedirs(model_dir, exist_ok=True)
    
#     writer = SummaryWriter(log_dir)

#     # --- 初始化 ---
    
#     # 1. 创建并行环境 (Vectorized Environment)
#     # 使用 AsyncVectorEnv 进行多进程并行采样，极大提高效率
#     envs = gym.vector.AsyncVectorEnv(
#         [make_env(i, i+100) for i in range(NUM_ENVS)]
#     )

#     # 2. 初始化网络
#     # obs_dim = 24 (8 Vector + 16 Lidar), action_dim = 2
#     policy = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=HIDDEN_DIM).to(device)
    
#     # 3. 初始化算法
#     algo = DSGRPO(
#         policy_net=policy,
#         optimizer_lr=LR,
#         beta_kl=0.01,    # KL 惩罚系数
#         clip_ratio=0.2,
#         device=device
#     )

#     # --- 训练变量初始化 ---
#     global_step = 0
#     start_time = time.time()
    
#     # 初始观测
#     obs, _ = envs.reset() # [Num_Envs, 24]
    
#     # GRU 隐状态初始化 [1, Num_Envs, Hidden]
#     # 注意: 这个 hidden 是保持跨步更新的，但在 episode 结束时需要重置
#     hidden_states = torch.zeros(1, NUM_ENVS, HIDDEN_DIM).to(device)
    
#     # 记录上一轮的策略熵，用于 DAPO 动态调整
#     last_entropy = 1.0 

#     print("Starting training...")

#     # --- 主循环 ---
#     while global_step < MAX_STEPS:
        
#         # =================================================
#         # A. DAPO 动态组大小计算 (Dynamic Group Sizing)
#         # =================================================
#         # 根据上一轮的不确定性(熵)，计算本轮需要的 Group Size
#         # 注: 在固定 NUM_ENVS 的实现中，我们实际上总是采集 NUM_ENVS 条数据，
#         # 但在日志中记录这个理论值，或者在 update 时只切片使用前 G_t 个数据(可选)。
#         # 这里为演示方便，我们采集所有数据，但打印 DAPO 建议值。
#         target_group_size = algo.determine_group_size(last_entropy, G_MIN, G_MAX)
        
#         # =================================================
#         # B. 数据收集 (Data Collection / Rollout)
#         # =================================================
#         rollouts = [[] for _ in range(NUM_ENVS)] # 为每个环境单独存储轨迹
        
#         step_rewards = []
        
#         for _ in range(ROLLOUT_STEPS):
#             # 1. 转为 Tensor
#             obs_tensor = torch.FloatTensor(obs).to(device)
            
#             # 2. 策略推理 (Action Selection)
#             # 注意: 这里输入是 [Num_Envs, 24], 网络内部会自动增加 unsqueeze(1) 处理单帧
#             # 输出 action 是 [-1, 1] 的 Tensor
#             actions, log_probs, next_hidden, entropy = algo.select_action(
#                 obs_tensor, hidden_states, deterministic=False
#             )
            
#             # 3. 环境步进
#             # AsyncVectorEnv 会自动让 step 并行执行
#             next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
#             # 4. 存储数据
#             # 我们需要存储当前步的信息
#             for i in range(NUM_ENVS):
#                 rollouts[i].append({
#                     'obs': obs_tensor[i],         # 当前观测 (Tensor)
#                     'action': torch.tensor(actions[i]), # 动作 (Tensor)
#                     'reward': rewards[i],         # 奖励 (Float)
#                     'log_prob': log_probs[i],     # 对数概率 (Float)
#                     'hidden': hidden_states[:, i:i+1, :].clone() # 当前步开始时的隐状态
#                 })
            
#             # 5. 状态更新
#             # 处理隐状态重置: 如果某个环境 Done 了，它的隐状态应该重置为 0
#             dones = np.logical_or(terminations, truncations)
#             if np.any(dones):
#                 # 将结束的环境对应的 hidden state 设为 0
#                 # hidden_states shape: [1, Num_Envs, Hidden]
#                 # dones shape: [Num_Envs]
#                 hidden_states[:, torch.tensor(dones).to(device), :] = 0.0
                
#             obs = next_obs
#             hidden_states = next_hidden
#             step_rewards.append(np.mean(rewards))
#             global_step += NUM_ENVS
            
#         # =================================================
#         # C. 数据整理 (Processing)
#         # =================================================
#         # 将列表转换为 update 函数需要的 batch 格式
#         # 我们将每个环境的 ROLLOUT_STEPS 步作为一个完整的 Sequence
        
#         batch_data = []
#         for i in range(NUM_ENVS):
#             # 堆叠单个环境的时间步
#             env_trace = {
#                 'obs': torch.stack([x['obs'] for x in rollouts[i]]),       # [Seq, 24]
#                 'action': torch.stack([x['action'] for x in rollouts[i]]), # [Seq, 2]
#                 'reward': [x['reward'] for x in rollouts[i]],              # List [Seq]
#                 'log_prob': [x['log_prob'] for x in rollouts[i]],          # List [Seq]
#                 # 只需要序列开始时的 hidden state 用于 GRU forward
#                 'hidden': rollouts[i][0]['hidden']                         # [1, 1, Hidden]
#             }
#             batch_data.append(env_trace)

#         # =================================================
#         # D. 算法更新 (Update)
#         # =================================================
#         # 这一步包含: Safety Filtering -> Advantage Calculation -> Policy Update
#         update_info = algo.update(batch_data)
        
#         # 更新熵记录用于下一轮 DAPO
#         last_entropy = entropy # 使用最后一步的熵作为估计
        
#         # =================================================
#         # E. 日志记录 (Logging)
#         # =================================================
#         avg_step_reward = np.mean(step_rewards)
        
#         if global_step % (NUM_ENVS * ROLLOUT_STEPS * 5) == 0:
#             elapsed_time = time.time() - start_time
#             print(f"Step: {global_step} | "
#                   f"Loss: {update_info['loss']:.4f} | "
#                   f"Avg Reward: {avg_step_reward:.2f} | "
#                   f"Safe Reward: {update_info['avg_reward_safe']:.2f} | "
#                   f"Safety Trig: {update_info['safety_trigger_rate']*100:.1f}% | "
#                   f"DAPO Group: {target_group_size}")
            
#             # TensorBoard
#             writer.add_scalar("Train/Loss", update_info['loss'], global_step)
#             writer.add_scalar("Train/PolicyLoss", update_info['policy_loss'], global_step)
#             writer.add_scalar("Train/KL_Divergence", update_info['kl'], global_step)
#             writer.add_scalar("Train/Raw_Reward", avg_step_reward, global_step)
#             writer.add_scalar("Train/Safety_Corrected_Reward", update_info['avg_reward_safe'], global_step)
#             writer.add_scalar("Safety/Filter_Trigger_Rate", update_info['safety_trigger_rate'], global_step)
#             writer.add_scalar("DAPO/Target_Group_Size", target_group_size, global_step)
#             writer.add_scalar("DAPO/Policy_Entropy", last_entropy, global_step)

#         # =================================================
#         # F. 模型保存
#         # =================================================
#         if global_step % 50000 == 0:
#             save_path = os.path.join(model_dir, f"model_{global_step}.pth")
#             algo.save(save_path)
#             print(f"Model saved to {save_path}")

#     # 训练结束
#     envs.close()
#     writer.close()
#     print("Training finished!")

# if __name__ == "__main__":
#     train()

import os
# 设置环境变量，允许 OpenMP 库重复加载
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import gymnasium as gym
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# --- 导入自定义模块 ---
# 请确保这些文件在同一目录下
# 注意：根据你的环境文件名，这里可能需要改为 from uncertain_env_improved import UncertainComplexEnv
# 这里保持和你提供的脚本一致
from uncertain_env import UncertainComplexEnv 
from CNN_GRU_policy import RobustGRPOPolicy
from ds_grpo import DSGRPO

# ==========================================
# 0. 辅助工具: 动作空间缩放包装器
# ==========================================
class ActionScalingWrapper(gym.ActionWrapper):
    """
    将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作
    v: [-1, 1] -> [-1, 2]
    w: [-1, 1] -> [-1, 1]
    """
    def __init__(self, env):
        super().__init__(env)
        # 定义新的动作空间为 [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        # 神经网络输出的 raw action
        v_scaled, w_scaled = action[0], action[1]
        
        # 线性映射
        # v: [-1, 1] -> [-1, 2]
        v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
        # w: [-1, 1] -> [-1, 1] (无需变化)
        w_real = w_scaled
        
        return np.array([v_real, w_real], dtype=np.float32)

# ==========================================
# 1. 环境工厂函数 (用于并行采样)
# ==========================================
def make_env(env_id, seed):
    def _thunk():
        # 训练环境不需要 render_mode="human"，以提高速度
        env = UncertainComplexEnv() 
        env = ActionScalingWrapper(env)
        env.reset(seed=seed)
        return env
    return _thunk

# ==========================================
# 2. 训练主循环
# ==========================================
def train():
    # --- 超参数设置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 环境参数
    NUM_ENVS = 16        # 并行环境数量
    MAX_STEPS = 500000   # 总训练步数
    ROLLOUT_STEPS = 64   # 每次更新前，每个环境采样的步数
    
    # [新增] 可视化频率: 每隔多少次 Update 进行一次渲染演示
    # 建议设为 10 或 20，太频繁会拖慢训练速度
    VISUALIZE_FREQ = 10 
    
    # DAPO 参数
    G_MIN = 4            
    G_MAX = 16           

    # 算法参数
    LR = 3e-4
    HIDDEN_DIM = 256
    
    # 路径设置
    exp_name = f"DS_GRPO_Uncertainty_{int(time.time())}"
    log_dir = f"runs/{exp_name}"
    model_dir = f"models/{exp_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)

    # --- 初始化 ---
    
    # 1. 创建并行训练环境 (不渲染)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, i+100) for i in range(NUM_ENVS)]
    )

    # [新增] 2. 创建一个独立的测试环境 (用于渲染)
    # 这个环境在主进程运行，专门用于给你看
    visual_env = UncertainComplexEnv(render_mode="human")
    visual_env = ActionScalingWrapper(visual_env)

    # 3. 初始化网络
    policy = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=HIDDEN_DIM).to(device)
    
    # 4. 初始化算法
    algo = DSGRPO(
        policy_net=policy,
        optimizer_lr=LR,
        beta_kl=0.01,    
        clip_ratio=0.2,
        device=device
    )

    # --- 训练变量初始化 ---
    global_step = 0
    update_count = 0 # 记录更新次数
    start_time = time.time()
    
    # 初始观测 (训练环境)
    obs, _ = envs.reset() 
    
    # GRU 隐状态初始化 (训练环境)
    hidden_states = torch.zeros(1, NUM_ENVS, HIDDEN_DIM).to(device)
    
    last_entropy = 1.0 

    print("Starting training...")

    # --- 主循环 ---
    while global_step < MAX_STEPS:
        
        # ... (A. DAPO 逻辑保持不变) ...
        target_group_size = algo.determine_group_size(last_entropy, G_MIN, G_MAX)
        
        # ... (B. 数据收集保持不变) ...
        rollouts = [[] for _ in range(NUM_ENVS)]
        step_rewards = []
        
        for _ in range(ROLLOUT_STEPS):
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            actions, log_probs, next_hidden, entropy = algo.select_action(
                obs_tensor, hidden_states, deterministic=False
            )
            
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            for i in range(NUM_ENVS):
                rollouts[i].append({
                    'obs': obs_tensor[i],
                    'action': torch.tensor(actions[i]),
                    'reward': rewards[i],
                    'log_prob': log_probs[i],
                    'hidden': hidden_states[:, i:i+1, :].clone()
                })
            
            dones = np.logical_or(terminations, truncations)
            if np.any(dones):
                hidden_states[:, torch.tensor(dones).to(device), :] = 0.0
                
            obs = next_obs
            hidden_states = next_hidden
            step_rewards.append(np.mean(rewards))
            global_step += NUM_ENVS
            
        # ... (C. 数据整理保持不变) ...
        batch_data = []
        for i in range(NUM_ENVS):
            env_trace = {
                'obs': torch.stack([x['obs'] for x in rollouts[i]]),
                'action': torch.stack([x['action'] for x in rollouts[i]]),
                'reward': torch.tensor([x['reward'] for x in rollouts[i]], dtype=torch.float32),
                
                # [修正] 使用 torch.tensor 将数值列表直接转为 Tensor
                'log_prob': torch.tensor([x['log_prob'] for x in rollouts[i]], dtype=torch.float32),
                
                'hidden': rollouts[i][0]['hidden']
            }
            batch_data.append(env_trace)

        # ... (D. 算法更新保持不变) ...
        update_info = algo.update(batch_data)
        last_entropy = entropy
        update_count += 1
        
        # =================================================
        # [新增] 可视化演示模块 (Render Block)
        # =================================================
        if update_count % VISUALIZE_FREQ == 0:
            print(f"\n[Visualization] Rendering test episode at step {global_step}...")
            
            # 重置测试环境
            vis_obs, _ = visual_env.reset(seed=42) # 固定种子看同一个场景，或者去掉 seed 看随机场景
            vis_hidden = torch.zeros(1, 1, HIDDEN_DIM).to(device) # 测试环境只有1个并发
            
            vis_done = False
            vis_step = 0
            
            # 跑一个 Episode
            while not vis_done and vis_step < 300: # 限制最大步数防止死循环
                # 渲染画面
                visual_env.render()
                
                # 准备输入数据 (增加 Batch 维度: [24] -> [1, 24])
                vis_obs_tensor = torch.FloatTensor(vis_obs).unsqueeze(0).to(device)
                
                # 策略推理 (deterministic=True 使用确定性策略，效果更稳定)
                # 注意: select_action 会自动处理维度，但我们需要手动处理返回值的维度
                # 因为 select_action 返回的是 batch numpy，我们需要取 [0]
                vis_action_all, _, vis_next_hidden, _ = algo.select_action(
                    vis_obs_tensor, vis_hidden, deterministic=True
                )
                
                # 取出第0个环境的动作 (因为测试环境只有一个)
                vis_action = vis_action_all[0] 
                
                # 环境步进
                vis_obs, vis_reward, vis_term, vis_trunc, _ = visual_env.step(vis_action)
                
                vis_hidden = vis_next_hidden
                vis_done = vis_term or vis_trunc
                vis_step += 1
            
            print(f"[Visualization] Episode finished in {vis_step} steps.\n")
            # 跑完后关闭窗口，或者保持窗口下一次继续用(这里不close，复用窗口)

        # ... (E. 日志记录保持不变) ...
        avg_step_reward = np.mean(step_rewards)
        if global_step % (NUM_ENVS * ROLLOUT_STEPS * 5) == 0:
            # ... (Log logic) ...
            print(f"Step: {global_step} | Loss: {update_info['loss']:.4f} | Avg R: {avg_step_reward:.2f} ...")
            # ... (Tensorboard logic) ...

        # ... (F. 模型保存保持不变) ...
        if global_step % 50000 == 0:
            save_path = os.path.join(model_dir, f"model_{global_step}.pth")
            algo.save(save_path)

    # 训练结束
    envs.close()
    visual_env.close() # 关闭可视化环境
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    train()