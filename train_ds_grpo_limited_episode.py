# # import os
# # # 设置环境变量，允许 OpenMP 库重复加载
# # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # import torch
# # import numpy as np
# # import gymnasium as gym
# # import os
# # import time
# # from collections import deque
# # from torch.utils.tensorboard import SummaryWriter

# # # --- 导入自定义模块 ---
# # from uncertain_env import UncertainComplexEnv 
# # from CNN_GRU_policy import RobustGRPOPolicy
# # from ds_grpo import DSGRPO

# # # ==========================================
# # # 0. 辅助工具: 动作空间缩放包装器
# # # ==========================================
# # class ActionScalingWrapper(gym.ActionWrapper):
# #     """
# #     将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作
# #     v: [-1, 1] -> [-1, 2]
# #     w: [-1, 1] -> [-1, 1]
# #     """
# #     def __init__(self, env):
# #         super().__init__(env)
# #         self.action_space = gym.spaces.Box(
# #             low=-1.0, high=1.0, shape=(2,), dtype=np.float32
# #         )

# #     def action(self, action):
# #         v_scaled, w_scaled = action[0], action[1]
# #         v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
# #         w_real = w_scaled
# #         return np.array([v_real, w_real], dtype=np.float32)

# # # ==========================================
# # # 1. 环境工厂函数
# # # ==========================================
# # def make_env(env_id, seed, max_steps=1000):
# #     def _thunk():
# #         env = UncertainComplexEnv() 
# #         env = ActionScalingWrapper(env)
# #         # 强制设置环境内部的最大步数，防止死循环
# #         env.max_steps = max_steps 
# #         env.reset(seed=seed)
# #         return env
# #     return _thunk

# # # ==========================================
# # # 2. 训练主循环 (基于 Episode)
# # # ==========================================
# # def train():
# #     # --- A. 超参数设置 ---
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     print(f"Using device: {device}")

# #     # 训练规模配置
# #     NUM_ENVS = 16          # 并行环境数量 (即 GRPO 的 Group Size)
# #     TOTAL_EPISODES = 5000  # 总共训练多少轮 (每轮包含 NUM_ENVS 个轨迹)
# #     MAX_EPISODE_STEPS = 500 # 每个 Episode 允许的最大步数 (超过强制截断)
    
# #     # 可视化频率 (每多少轮 Episode 演示一次)
# #     VISUALIZE_FREQ = 20 
    
# #     # 算法参数
# #     LR = 3e-4
# #     HIDDEN_DIM = 256
    
# #     # 路径设置
# #     exp_name = f"DS_GRPO_Episodic_{int(time.time())}"
# #     log_dir = f"runs/{exp_name}"
# #     model_dir = f"models/{exp_name}"
# #     os.makedirs(model_dir, exist_ok=True)
    
# #     writer = SummaryWriter(log_dir)

# #     # --- B. 初始化 ---
    
# #     # 1. 创建并行训练环境
# #     # 注意：所有环境共用一个 max_steps 设置
# #     envs = gym.vector.AsyncVectorEnv(
# #         [make_env(i, i+100, MAX_EPISODE_STEPS) for i in range(NUM_ENVS)]
# #     )

# #     # 2. 创建可视化测试环境
# #     visual_env = UncertainComplexEnv(render_mode="human")
# #     visual_env = ActionScalingWrapper(visual_env)
# #     visual_env.max_steps = MAX_EPISODE_STEPS # 保持一致

# #     # 3. 初始化网络与算法
# #     policy = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=HIDDEN_DIM).to(device)
    
# #     algo = DSGRPO(
# #         policy_net=policy,
# #         optimizer_lr=LR,
# #         beta_kl=0.01,    
# #         clip_ratio=0.2,
# #         device=device
# #     )

# #     # --- C. 训练状态变量 ---
# #     global_step = 0
# #     start_time = time.time()
    
# #     # 初始化 GRU 隐状态 (Batch=NUM_ENVS)
# #     hidden_states = torch.zeros(1, NUM_ENVS, HIDDEN_DIM).to(device)
    
# #     # 初始观测
# #     obs, _ = envs.reset() 

# #     print("Starting Episode-based Training...")

# #     # --- D. 主循环 (按 Episode 迭代) ---
# #     for episode_idx in range(1, TOTAL_EPISODES + 1):
        
# #         # =================================================
# #         # 1. 数据收集 (Collect Rollouts)
# #         #    目标：跑完这 NUM_ENVS 个环境的完整一局
# #         # =================================================
        
# #         # 临时存储每个环境的轨迹数据
# #         # 结构: rollouts[env_id] = list of step_dicts
# #         rollouts = [[] for _ in range(NUM_ENVS)]
        
# #         # 记录哪些环境已经结束了
# #         env_dones = np.zeros(NUM_ENVS, dtype=bool)
        
# #         # 性能统计
# #         episode_rewards = np.zeros(NUM_ENVS)
# #         episode_lengths = np.zeros(NUM_ENVS)
        
# #         # 如果是新的开始，重置环境 (通常 AsyncVectorEnv 会自动重置 done 的环境，
# #         # 但为了保证数据对齐，我们在这里显式 reset 并不是必须的，
# #         # 只要保证 obs 是上一局延续下来的或者是 reset 过的即可。
# #         # 这里为了逻辑清晰，且上面的 obs 变量维护了当前状态，直接开始循环)
        
# #         step_in_ep = 0
# #         while not np.all(env_dones):
# #             step_in_ep += 1
            
# #             # (1) 准备数据
# #             obs_tensor = torch.FloatTensor(obs).to(device)
            
# #             # (2) 策略推理
# #             # 注意：对于已经 done 的环境，其实不需要再计算了，但为了并行方便，我们通常全部计算，
# #             # 只要在存储时过滤掉多余数据即可，或者让环境自动 reset 继续跑 (AutoReset)。
# #             # Gym VectorEnv 默认是 AutoReset 的。我们需要处理好这一点。
# #             # 策略：我们让环境一直跑，但在数据整理阶段截断到第一次 Done 的位置。
            
# #             with torch.no_grad():
# #                 actions, log_probs, next_hidden, entropy = algo.select_action(
# #                     obs_tensor, hidden_states, deterministic=False
# #                 )
            
# #             # (3) 环境步进
# #             next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
# #             # (4) 存储数据
# #             # 只存储尚未完成的环境的数据
# #             for i in range(NUM_ENVS):
# #                 if not env_dones[i]:
# #                     rollouts[i].append({
# #                         'obs': obs_tensor[i],
# #                         'action': torch.tensor(actions[i]),
# #                         'reward': rewards[i],
# #                         'log_prob': log_probs[i],
# #                         'hidden': hidden_states[:, i:i+1, :].clone() # 存当前步开始时的 hidden
# #                     })
# #                     episode_rewards[i] += rewards[i]
# #                     episode_lengths[i] += 1
            
# #             # (5) 检查结束状态
# #             dones_now = np.logical_or(terminations, truncations)
            
# #             # 更新已完成标志
# #             # 注意：VectorEnv 会自动 Reset 完成的环境，next_obs 已经是新的一局的开始了
# #             # 所以如果 done[i] 为 True，说明这一步是该 Episode 的最后一步
# #             for i in range(NUM_ENVS):
# #                 if dones_now[i]:
# #                     env_dones[i] = True
# #                     # 重置该环境的 hidden state (为下一轮或者自动重置后的新步骤准备)
# #                     next_hidden[:, i, :] = 0.0
            
# #             # (6) 更新状态
# #             obs = next_obs
# #             hidden_states = next_hidden
            
# #             # 强制截断保护 (虽然环境有 max_steps，但为了代码健壮性)
# #             if step_in_ep >= MAX_EPISODE_STEPS + 5:
# #                 break

# #         # 更新全局步数
# #         total_steps_this_batch = sum(episode_lengths)
# #         global_step += total_steps_this_batch

# #         # =================================================
# #         # 2. 数据整理 (Process Batch)
# #         #    将变长的轨迹整理成 Update 需要的格式
# #         #    注意：因为不同环境结束时间不同，Tensor 无法直接堆叠成 [Batch, Seq]，
# #         #    需要 Pad 填充或者分别处理。为了适配 ds_grpo.py，我们保持 List[Dict] 结构
# #         #    或者简单地 Padding 到最大长度。
# #         # =================================================
        
# #         # 简单起见，我们修改一下 batch_data 的结构，让它是一个 List，
# #         # 其中每个元素是一个环境的完整轨迹数据 (Tensor 形式)
# #         batch_data = []
# #         for i in range(NUM_ENVS):
# #             # 堆叠单个环境的时间步
# #             # 此时 rollouts[i] 的长度就是该环境该 Episode 的实际长度
# #             env_trace = {
# #                 'obs': torch.stack([x['obs'] for x in rollouts[i]]),       # [T, 24]
# #                 'action': torch.stack([x['action'] for x in rollouts[i]]), # [T, 2]
# #                 'reward': torch.tensor([x['reward'] for x in rollouts[i]], dtype=torch.float32), # [T]
# #                 'log_prob': torch.tensor([x['log_prob'] for x in rollouts[i]], dtype=torch.float32), # [T]
# #                 'hidden': rollouts[i][0]['hidden'] # 该轨迹起始的 hidden [1, 1, H]
# #             }
# #             batch_data.append(env_trace)

# #         # =================================================
# #         # 3. 算法更新 (Update)
# #         # =================================================
# #         # 注意: ds_grpo.py 中的 update 函数和 safety_filter 需要支持变长序列
# #         # 如果之前的实现是基于固定 Seq 长度的 Tensor 堆叠操作 (batch_ops)，可能会报错。
# #         # 鉴于此，最稳妥的方式是：在 update 内部循环处理每个 Batch，或者在这里做 Padding。
# #         # **为了兼容性，我们在这里进行 Padding，将所有轨迹补齐到当前批次的最大长度**
        
# #         max_len = int(max(episode_lengths))
        
# #         # 构建 Padded Tensors
# #         padded_obs = torch.zeros(NUM_ENVS, max_len, 24).to(device)
# #         padded_actions = torch.zeros(NUM_ENVS, max_len, 2).to(device)
# #         padded_rewards = torch.zeros(NUM_ENVS, max_len).to(device)
# #         # 为了计算方便，padded 的 log_prob 设为 0 (不影响梯度，因为会被 mask 掉)
# #         padded_log_probs = torch.zeros(NUM_ENVS, max_len).to(device)
        
# #         # 创建 Mask 标记有效数据
# #         mask_tensor = torch.zeros(NUM_ENVS, max_len).to(device)
        
# #         start_hiddens = torch.cat([b['hidden'] for b in batch_data], dim=1).to(device) # [1, B, H]

# #         for i in range(NUM_ENVS):
# #             length = int(episode_lengths[i])
# #             padded_obs[i, :length, :] = batch_data[i]['obs']
# #             padded_actions[i, :length, :] = batch_data[i]['action']
# #             padded_rewards[i, :length] = batch_data[i]['reward']
# #             padded_log_probs[i, :length] = batch_data[i]['log_prob']
# #             mask_tensor[i, :length] = 1.0
            
# #         # 重新打包成 update 函数能接受的格式
# #         # 注意：这里我们传入一个特殊的字典结构，需要在 update 函数里稍作适配
# #         # 或者我们直接修改 ds_grpo.py 来处理 Padding。
# #         # 为了不改动 ds_grpo.py 太多，我们这里模拟之前的结构，
# #         # 但要注意：ds_grpo.py 里的计算必须考虑 mask！
        
# #         # 临时方案：直接传递 Padded 数据，但需要修改 ds_grpo.py
# #         # 如果不想改 ds_grpo.py，那只能把 max_len 设为固定值 (例如 500)，但这会浪费显存。
# #         # 鉴于之前的报错都是维度问题，这里我们传递 List 形式给 update，
# #         # 让 update 内部去处理变长序列（如果它支持的话）。
# #         # **查看之前的 ds_grpo.py，它使用了 torch.stack，这意味着它假设长度一致。**
# #         # 所以必须传入 Padded 数据，并且 ds_grpo 必须修改以支持 Mask。
        
# #         # 既然用户没有要求改 ds_grpo，我在这里做一个 Trick：
# #         # 将 update 的输入构造为 batch_data 列表，
# #         # 但为了避免 stack 报错，我们需要 update 函数能够处理 List[Tensor]。
# #         # 如果不能，我们必须确保所有 rollout 长度一致 (这在 Episode 模式下是不可能的)。
        
# #         # **解决方案**: 
# #         # 为了让代码能跑，我们需要在这个脚本里处理好 Mask 逻辑，
# #         # 并告诉用户需要微调 update 函数。
# #         # 这里我先传入 Padded 数据，假设 update 函数会被相应修改（见下方说明）。
        
# #         # 实际上，我们可以利用 GRPO 的特性：它对每条轨迹分别计算 Advantage。
# #         # 所以我们可以把所有数据 Flatten 成一条超级长的序列进行一次性前向传播？不行，GRU 状态不对。
        
# #         # 最简单的做法：传入 batch_data (List of Dicts)，
# #         # 并修改 ds_grpo.py 的 update 函数，让它处理 List 而不是试图 Stack。
        
# #         # 这里我们调用 update，传入 list。
# #         # 对应的 ds_grpo.py 修改建议会在下方给出。
# #         update_info = algo.update(batch_data) 
        
# #         # =================================================
# #         # 4. 日志记录
# #         # =================================================
# #         avg_ep_reward = np.mean(episode_rewards)
# #         avg_ep_len = np.mean(episode_lengths)
        
# #         if episode_idx % 5 == 0:
# #             print(f"Episode: {episode_idx}/{TOTAL_EPISODES} | "
# #                   f"Steps: {global_step} | "
# #                   f"Loss: {update_info.get('loss', 0):.4f} | "
# #                   f"Avg Reward: {avg_ep_reward:.2f} | "
# #                   f"Avg Len: {avg_ep_len:.1f}")
            
# #             writer.add_scalar("Train/Episode_Reward", avg_ep_reward, episode_idx)
# #             writer.add_scalar("Train/Episode_Length", avg_ep_len, episode_idx)
# #             writer.add_scalar("Train/Loss", update_info.get('loss', 0), episode_idx)
# #             writer.add_scalar("Safety/Trigger_Rate", update_info.get('safety_trigger_rate', 0), episode_idx)

# #         # =================================================
# #         # 5. 可视化演示
# #         # =================================================
# #         if episode_idx % VISUALIZE_FREQ == 0:
# #             print(f"\n[Visualization] Rendering test episode...")
# #             vis_obs, _ = visual_env.reset(seed=42)
# #             vis_hidden = torch.zeros(1, 1, HIDDEN_DIM).to(device) # 清空记忆
            
# #             vis_done = False
# #             vis_steps = 0
            
# #             while not vis_done and vis_steps < MAX_EPISODE_STEPS:
# #                 visual_env.render()
# #                 vis_obs_tensor = torch.FloatTensor(vis_obs).unsqueeze(0).to(device)
                
# #                 vis_action_all, _, vis_next_hidden, _ = algo.select_action(
# #                     vis_obs_tensor, vis_hidden, deterministic=True
# #                 )
# #                 vis_action = vis_action_all[0]
# #                 vis_obs, _, term, trunc, _ = visual_env.step(vis_action)
                
# #                 vis_hidden = vis_next_hidden
# #                 vis_done = term or trunc
# #                 vis_steps += 1
# #             print(f"[Visualization] Finished in {vis_steps} steps.\n")

# #         # =================================================
# #         # 6. 保存模型
# #         # =================================================
# #         if episode_idx % 500 == 0:
# #             save_path = os.path.join(model_dir, f"model_ep_{episode_idx}.pth")
# #             algo.save(save_path)

# #     # 结束
# #     envs.close()
# #     visual_env.close()
# #     writer.close()
# #     print("Training finished!")

# # if __name__ == "__main__":
# #     train()


# import os
# # 设置环境变量，允许 OpenMP 库重复加载
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import torch
# import numpy as np
# import gymnasium as gym
# import os
# import time
# from torch.utils.tensorboard import SummaryWriter

# # --- 导入自定义模块 ---
# from uncertain_env import UncertainComplexEnv 
# from CNN_GRU_policy import RobustGRPOPolicy
# from ds_grpo import DSGRPO

# # ==========================================
# # 0. 辅助工具: 动作空间缩放包装器
# # ==========================================
# class ActionScalingWrapper(gym.ActionWrapper):
#     """
#     将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         self.action_space = gym.spaces.Box(
#             low=-1.0, high=1.0, shape=(2,), dtype=np.float32
#         )

#     def action(self, action):
#         v_scaled, w_scaled = action[0], action[1]
#         v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
#         w_real = w_scaled
#         return np.array([v_real, w_real], dtype=np.float32)

# # ==========================================
# # 1. 环境工厂函数
# # ==========================================
# def make_env(env_id, seed, max_steps):
#     def _thunk():
#         env = UncertainComplexEnv() 
#         env = ActionScalingWrapper(env)
#         # 强制设置环境内部的最大步数
#         env.max_steps = max_steps 
#         env.reset(seed=seed)
#         return env
#     return _thunk

# # ==========================================
# # 2. 训练主循环 (Long Horizon Episode-Based)
# # ==========================================
# def train():
#     # --- A. 超参数设置 ---
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # 训练规模配置 [关键修改]
#     NUM_ENVS = 16           # 并行环境数量
#     TOTAL_EPISODES = 2000   # 总共训练的轮数 (减少数量)
#     MAX_EPISODE_STEPS = 2000 # [关键] 单个 Episode 允许的最大步数 (大幅增加，保证充分探索)
    
#     # 可视化频率
#     VISUALIZE_FREQ = 20     # 每 20 轮渲染一次
    
#     # 算法参数
#     LR = 3e-4
#     HIDDEN_DIM = 256
    
#     # 路径设置
#     exp_name = f"DS_GRPO_LongHorizon_{int(time.time())}"
#     log_dir = f"runs/{exp_name}"
#     model_dir = f"models/{exp_name}"
#     os.makedirs(model_dir, exist_ok=True)
    
#     writer = SummaryWriter(log_dir)

#     # --- B. 初始化 ---
    
#     # 1. 创建并行训练环境
#     envs = gym.vector.AsyncVectorEnv(
#         [make_env(i, i+100, MAX_EPISODE_STEPS) for i in range(NUM_ENVS)]
#     )

#     # 2. 创建可视化测试环境
#     visual_env = UncertainComplexEnv(render_mode="human")
#     visual_env = ActionScalingWrapper(visual_env)
#     visual_env.max_steps = MAX_EPISODE_STEPS # 保持一致

#     # 3. 初始化网络与算法
#     policy = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=HIDDEN_DIM).to(device)
    
#     algo = DSGRPO(
#         policy_net=policy,
#         optimizer_lr=LR,
#         beta_kl=0.01,    
#         clip_ratio=0.2,
#         device=device
#     )

#     # --- C. 训练状态变量 ---
#     global_step = 0
#     start_time = time.time()
    
#     # 初始观测
#     obs, _ = envs.reset() 
    
#     # GRU 隐状态 (Batch=NUM_ENVS)
#     hidden_states = torch.zeros(1, NUM_ENVS, HIDDEN_DIM).to(device)
    
#     last_entropy = 1.0 

#     print("Starting Long-Horizon Training...")
#     print(f"Config: {TOTAL_EPISODES} Episodes, Max Steps {MAX_EPISODE_STEPS}")

#     # --- D. 主循环 (按 Episode 迭代) ---
#     for episode_idx in range(1, TOTAL_EPISODES + 1):
        
#         # DAPO: 动态决定是否需要更多探索 (此处仅记录建议值，实际采用所有并行环境)
#         target_group_size = algo.determine_group_size(last_entropy, 4, NUM_ENVS)
        
#         # =================================================
#         # 1. 数据收集 (Collect Rollouts)
#         #    逻辑：一直跑，直到所有环境都 Done，但只记录 Done 之前的数据
#         # =================================================
        
#         # 临时存储
#         rollouts = [[] for _ in range(NUM_ENVS)]
        
#         # 标记哪些环境还在运行 (True=Running, False=Done)
#         active_envs = np.ones(NUM_ENVS, dtype=bool)
        
#         episode_rewards = np.zeros(NUM_ENVS)
#         episode_lengths = np.zeros(NUM_ENVS)
        
#         # 重置环境 (虽然 VectorEnv 自动重置，但为了逻辑清晰和清空 hidden，建议每轮开始前统一状态)
#         # 但因为 AsyncVectorEnv 不支持部分 Reset，我们依赖上一轮的 Auto-Reset 状态，
#         # 唯独需要清空的是 RNN 的记忆。
#         hidden_states.fill_(0.0) 
        
#         current_steps = 0
        
#         while np.any(active_envs):
#             current_steps += 1
            
#             # (1) 准备数据
#             obs_tensor = torch.FloatTensor(obs).to(device)
            
#             # (2) 策略推理
#             with torch.no_grad():
#                 actions, log_probs, next_hidden, entropy = algo.select_action(
#                     obs_tensor, hidden_states, deterministic=False
#                 )
            
#             # (3) 环境步进 (所有环境都会走一步，即使是已经 Done 的环境也会跑新的一局)
#             next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
#             dones = np.logical_or(terminations, truncations)
            
#             # (4) 存储数据 (只存储 Active 环境的数据)
#             for i in range(NUM_ENVS):
#                 if active_envs[i]:
#                     rollouts[i].append({
#                         'obs': obs_tensor[i],
#                         'action': torch.tensor(actions[i]),
#                         'reward': rewards[i],
#                         'log_prob': log_probs[i],
#                         'hidden': hidden_states[:, i:i+1, :].clone()
#                     })
#                     episode_rewards[i] += rewards[i]
#                     episode_lengths[i] += 1
                    
#                     # 如果这一步 Done 了，标记为不活跃，并清空它的 hidden state
#                     # 这样下一轮循环时，虽然环境还在跑(跑下一局)，但我们不再记录它的数据了
#                     if dones[i]:
#                         active_envs[i] = False
#                         # 这里的 hidden 清零是为了下一轮 Episode 准备，
#                         # 虽然本轮循环后面不会再用到这个 hidden 了。
#                         next_hidden[:, i, :] = 0.0

#             # (5) 强制退出条件
#             if current_steps >= MAX_EPISODE_STEPS:
#                 break
            
#             obs = next_obs
#             hidden_states = next_hidden

#         # 更新全局统计
#         global_step += int(np.sum(episode_lengths))
        
#         # =================================================
#         # 2. 数据整理 (Process Batch)
#         #    处理变长序列
#         # =================================================
#         batch_data = []
#         for i in range(NUM_ENVS):
#             # 将 List of Tensors 堆叠成 Tensor
#             # 注意: 这里每个 rollout[i] 的长度可能不同
#             env_trace = {
#                 'obs': torch.stack([x['obs'] for x in rollouts[i]]),       # [T_i, 24]
#                 'action': torch.stack([x['action'] for x in rollouts[i]]), # [T_i, 2]
#                 'reward': torch.tensor([x['reward'] for x in rollouts[i]], dtype=torch.float32), # [T_i]
#                 'log_prob': torch.tensor([x['log_prob'] for x in rollouts[i]], dtype=torch.float32), # [T_i]
#                 'hidden': rollouts[i][0]['hidden'] # [1, 1, H]
#             }
#             batch_data.append(env_trace)

#         # =================================================
#         # 3. 算法更新 (Update)
#         # =================================================
#         # 此时 batch_data 是一个列表，每个元素是一条变长的轨迹
#         # 您的 ds_grpo.py 需要支持这种 List 输入 (在之前的对话中我们已经确认了这种修改)
#         update_info = algo.update(batch_data)
        
#         last_entropy = entropy

#         # =================================================
#         # 4. 日志记录
#         # =================================================
#         avg_ep_reward = np.mean(episode_rewards)
#         avg_ep_len = np.mean(episode_lengths)
        
#         if episode_idx % 5 == 0:
#             print(f"Episode: {episode_idx}/{TOTAL_EPISODES} | "
#                   f"Avg Len: {avg_ep_len:.1f} | "
#                   f"Loss: {update_info.get('loss', 0):.4f} | "
#                   f"Avg Reward: {avg_ep_reward:.2f} | "
#                   f"Safety Trig: {update_info.get('safety_trigger_rate', 0)*100:.1f}%")
            
#             writer.add_scalar("Train/Episode_Reward", avg_ep_reward, episode_idx)
#             writer.add_scalar("Train/Episode_Length", avg_ep_len, episode_idx)
#             writer.add_scalar("Train/Loss", update_info.get('loss', 0), episode_idx)
#             writer.add_scalar("Safety/Trigger_Rate", update_info.get('safety_trigger_rate', 0), episode_idx)

#         # =================================================
#         # 5. 可视化演示 (Render)
#         # =================================================
#         if episode_idx % VISUALIZE_FREQ == 0:
#             print(f"\n[Visualization] Rendering test episode...")
#             vis_obs, _ = visual_env.reset(seed=42)
#             # [关键] 每次重置必须清空记忆！
#             vis_hidden = torch.zeros(1, 1, HIDDEN_DIM).to(device) 
            
#             vis_done = False
#             vis_steps = 0
            
#             while not vis_done and vis_steps < MAX_EPISODE_STEPS:
#                 visual_env.render()
#                 vis_obs_tensor = torch.FloatTensor(vis_obs).unsqueeze(0).to(device)
                
#                 vis_action_all, _, vis_next_hidden, _ = algo.select_action(
#                     vis_obs_tensor, vis_hidden, deterministic=True
#                 )
#                 vis_action = vis_action_all[0]
#                 vis_obs, _, term, trunc, _ = visual_env.step(vis_action)
                
#                 vis_hidden = vis_next_hidden
#                 vis_done = term or trunc
#                 vis_steps += 1
#             print(f"[Visualization] Finished in {vis_steps} steps.\n")

#         # =================================================
#         # 6. 保存模型
#         # =================================================
#         if episode_idx % 200 == 0:
#             save_path = os.path.join(model_dir, f"model_ep_{episode_idx}.pth")
#             algo.save(save_path)
#             print(f"Model saved to {save_path}")

#     # 结束
#     envs.close()
#     visual_env.close()
#     writer.close()
#     print("Training finished!")

# if __name__ == "__main__":
#     train()


import os
# 设置环境变量，允许 OpenMP 库重复加载 (防止 Windows 下 Numpy/Torch 冲突报错)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import gymnasium as gym
import os
import time
from torch.utils.tensorboard import SummaryWriter

# --- 导入自定义模块 ---
# 请确保文件名与您实际保存的一致
from uncertain_env import UncertainComplexEnv 
from CNN_GRU_policy import RobustGRPOPolicy
from ds_grpo import DSGRPO

# ==========================================
# 0. 辅助工具: 动作空间缩放包装器
# ==========================================
class ActionScalingWrapper(gym.ActionWrapper):
    """
    将神经网络输出的归一化动作 [-1, 1] 映射到环境真实物理动作
    v: [-1, 1] -> [-1, 2] (根据环境设定)
    w: [-1, 1] -> [-1, 1]
    """
    def __init__(self, env):
        super().__init__(env)
        # 定义 Gym 标准化的动作空间 [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, action):
        v_scaled, w_scaled = action[0], action[1]
        # 线性映射: (-1 + 1)/2 * 3 - 1 = -1; (1 + 1)/2 * 3 - 1 = 2
        v_real = (v_scaled + 1.0) / 2.0 * 3.0 - 1.0
        w_real = w_scaled
        return np.array([v_real, w_real], dtype=np.float32)

# ==========================================
# 1. 环境工厂函数 (Gym VectorEnv 专用)
# ==========================================
def make_env(env_id, seed, max_steps):
    def _thunk():
        # 训练环境不需要渲染，速度优先
        env = UncertainComplexEnv() 
        env = ActionScalingWrapper(env)
        # 强制设置环境内部的最大步数，防止死循环
        env.max_steps = max_steps 
        env.reset(seed=seed)
        return env
    return _thunk

# ==========================================
# 2. 训练主循环 (基于完整 Episode 的变长序列训练)
# ==========================================
def train():
    # --- A. 超参数设置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 训练配置
    NUM_ENVS = 16           # 并行环境数量 (Group Size)
    TOTAL_EPISODES = 500   # 总训练轮数 (Episode Based)
    MAX_EPISODE_STEPS = 500 # 单个 Episode 允许的最大步数 (长视距)
    
    VISUALIZE_FREQ = 20     # 可视化频率 (每20轮看一次)
    
    # DAPO 参数
    G_MIN = 4
    G_MAX = 16 # 最大不超过 NUM_ENVS
    
    # 算法参数
    LR = 3e-4
    HIDDEN_DIM = 256
    
    # 路径设置
    exp_name = f"DS_GRPO_Final_{int(time.time())}"
    log_dir = f"runs/{exp_name}"
    model_dir = f"models/{exp_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)

    # --- B. 初始化 ---
    
    # 1. 创建并行训练环境
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, i+100, MAX_EPISODE_STEPS) for i in range(NUM_ENVS)]
    )

    # 2. 创建可视化测试环境 (主进程运行)
    visual_env = UncertainComplexEnv(render_mode="human")
    visual_env = ActionScalingWrapper(visual_env)
    visual_env.max_steps = MAX_EPISODE_STEPS 

    # 3. 初始化网络与算法
    policy = RobustGRPOPolicy(obs_dim=24, action_dim=2, hidden_dim=HIDDEN_DIM).to(device)
    
    # 实例化算法类
    algo = DSGRPO(
        policy_net=policy,
        optimizer_lr=LR,
        beta_kl=0.01,    
        clip_ratio=0.2,
        entropy_coef=0.02, # 确保这里的系数设置正确，防止过早收敛
        device=device
    )

    # --- C. 训练状态变量 ---
    global_step = 0
    start_time = time.time()
    
    # 初始观测
    obs, _ = envs.reset() 
    
    # GRU 隐状态 (Batch=NUM_ENVS)
    hidden_states = torch.zeros(1, NUM_ENVS, HIDDEN_DIM).to(device)
    
    # 记录上一轮的平均熵
    last_avg_entropy = 1.0 

    print("Starting Long-Horizon Episode Training...")
    print(f"Config: {TOTAL_EPISODES} Episodes, Max Steps {MAX_EPISODE_STEPS}")

    # --- D. 主循环 (按 Episode 迭代) ---
    for episode_idx in range(1, TOTAL_EPISODES + 1):
        
        # [DAPO] 动态决定是否需要更多探索 
        # (此处仅计算建议值用于记录，实际采样仍使用 NUM_ENVS 个并行环境)
        target_group_size = algo.determine_group_size(last_avg_entropy, G_MIN, G_MAX)
        
        # =================================================
        # 1. 数据收集 (Collect Rollouts)
        #    逻辑：让所有环境跑完当前 Episode
        # =================================================
        
        # 临时存储每个环境的轨迹
        rollouts = [[] for _ in range(NUM_ENVS)]
        
        # 标记哪些环境还在运行 (True=Running, False=Done)
        active_envs = np.ones(NUM_ENVS, dtype=bool)
        
        episode_rewards = np.zeros(NUM_ENVS)
        episode_lengths = np.zeros(NUM_ENVS)
        
        # 本轮累计的熵 (用于计算平均值)
        ep_entropy_sum = 0.0
        ep_step_count = 0
        
        # 重置 GRU 记忆 (每轮 Episode 开始前清空)
        hidden_states.fill_(0.0) 
        
        current_steps = 0
        
        # 只要还有一个环境没跑完，就继续
        while np.any(active_envs):
            current_steps += 1
            
            # (1) 准备数据
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # (2) 策略推理
            # 注意: select_action 返回的 entropy 已经是 float 标量 (16个环境的均值)
            with torch.no_grad():
                actions, log_probs, next_hidden, entropy_val = algo.select_action(
                    obs_tensor, hidden_states, deterministic=False
                )
            
            # 累加熵值用于统计
            ep_entropy_sum += entropy_val
            ep_step_count += 1
            
            # (3) 环境步进
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            
            dones = np.logical_or(terminations, truncations)
            
            # (4) 存储数据 (只存储 Active 环境的数据)
            for i in range(NUM_ENVS):
                if active_envs[i]:
                    rollouts[i].append({
                        'obs': obs_tensor[i].cpu(),         # 移回 CPU 节省显存
                        'action': torch.tensor(actions[i]), # 已经是 CPU tensor
                        'reward': rewards[i],               # Numpy float
                        'log_prob': log_probs[i],           # Numpy float
                        'hidden': hidden_states[:, i:i+1, :].clone().cpu() # 存当前步隐状态
                    })
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    # 如果这一步 Done 了，停止记录该环境
                    if dones[i]:
                        active_envs[i] = False
                        # 清空该环境对应的隐状态，防止影响下一轮
                        next_hidden[:, i, :] = 0.0

            # (5) 强制退出条件
            if current_steps >= MAX_EPISODE_STEPS:
                break
            
            obs = next_obs
            hidden_states = next_hidden

        # 更新全局统计
        global_step += int(np.sum(episode_lengths))
        
        # 计算本轮 Episode 的平均熵
        last_avg_entropy = ep_entropy_sum / max(1, ep_step_count)
        
        # =================================================
        # 2. 数据整理 (Process Batch)
        # =================================================
        batch_data = []
        for i in range(NUM_ENVS):
            # 将列表堆叠成 Tensor
            # 注意：reward 和 log_prob 还是 numpy list，需要转 Tensor
            env_trace = {
                'obs': torch.stack([x['obs'] for x in rollouts[i]]),       # [T_i, 24]
                'action': torch.stack([x['action'] for x in rollouts[i]]), # [T_i, 2]
                'reward': torch.tensor([x['reward'] for x in rollouts[i]], dtype=torch.float32), # [T_i]
                'log_prob': torch.tensor([x['log_prob'] for x in rollouts[i]], dtype=torch.float32), # [T_i]
                'hidden': rollouts[i][0]['hidden'] # [1, 1, H]
            }
            batch_data.append(env_trace)

        # =================================================
        # 3. 算法更新 (Update)
        # =================================================
        # 调用修改后的 update 函数 (支持变长列表输入)
        update_info = algo.update(batch_data)
        
        # =================================================
        # 4. 日志记录
        # =================================================
        avg_ep_reward = np.mean(episode_rewards)
        avg_ep_len = np.mean(episode_lengths)
        
        if episode_idx % 5 == 0:
            print(f"Episode: {episode_idx}/{TOTAL_EPISODES} | "
                  f"Avg Len: {avg_ep_len:.1f} | "
                  f"Loss: {update_info.get('loss', 0):.4f} | "
                  f"Avg Reward: {avg_ep_reward:.2f} | "
                  f"Entropy: {last_avg_entropy:.4f} | "
                  f"Safety Trig: {update_info.get('safety_trigger_rate', 0)*100:.1f}%")
            
            writer.add_scalar("Train/Episode_Reward", avg_ep_reward, episode_idx)
            writer.add_scalar("Train/Episode_Length", avg_ep_len, episode_idx)
            writer.add_scalar("Train/Loss", update_info.get('loss', 0), episode_idx)
            writer.add_scalar("Train/Entropy", last_avg_entropy, episode_idx)
            writer.add_scalar("Safety/Trigger_Rate", update_info.get('safety_trigger_rate', 0), episode_idx)
            writer.add_scalar("DAPO/Target_Group", target_group_size, episode_idx)

        # =================================================
        # 5. 可视化演示 (Render)
        # =================================================
        if episode_idx % VISUALIZE_FREQ == 0:
            print(f"\n[Visualization] Rendering test episode...")
            vis_obs, _ = visual_env.reset(seed=42)
            # [关键] 每次重置必须清空记忆！
            vis_hidden = torch.zeros(1, 1, HIDDEN_DIM).to(device) 
            
            vis_done = False
            vis_steps = 0
            
            while not vis_done and vis_steps < MAX_EPISODE_STEPS:
                visual_env.render()
                # 增加 Batch 维度
                vis_obs_tensor = torch.FloatTensor(vis_obs).unsqueeze(0).to(device)
                
                # 使用确定性策略
                vis_action_all, _, vis_next_hidden, _ = algo.select_action(
                    vis_obs_tensor, vis_hidden, deterministic=True
                )
                vis_action = vis_action_all[0]
                vis_obs, _, term, trunc, _ = visual_env.step(vis_action)
                
                vis_hidden = vis_next_hidden
                vis_done = term or trunc
                vis_steps += 1
            print(f"[Visualization] Finished in {vis_steps} steps.\n")

        # =================================================
        # 6. 保存模型
        # =================================================
        if episode_idx % 200 == 0:
            save_path = os.path.join(model_dir, f"model_ep_{episode_idx}.pth")
            algo.save(save_path)
            print(f"Model saved to {save_path}")

    # 结束
    envs.close()
    visual_env.close()
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    train()