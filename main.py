import numpy as np
import random
import time

class HarmoniQAgent:
    def __init__(self):
        # 动作空间：音量调整的幅度 (dB)
        # -4: 大幅降低, -2: 小幅降低, 0: 保持原样, +2: 小幅提高, +4: 大幅提高
        self.actions = [-4, -2, 0, 2, 4]
        
        # 强化学习参数
        self.alpha = 0.1      # 学习率：新知识覆盖旧知识的速度
        self.gamma = 0.9      # 折扣因子：对未来奖励的重视程度
        self.epsilon = 0.2    # 探索率：20%的概率随机尝试新音量，80%使用已知最优经验
        
        # Q-Table 记忆库：记录在特定状态下，采取不同动作的预期收益
        self.q_table = {}

    def _discretize_state(self, noise_db, song_lufs):
        """感知层：将连续的传感器数据转化为离散的 '状态 (State)'"""
        # 划分环境噪音状态
        if noise_db < 50:
            env_state = "Quiet"      # 安静 (如卧室)
        elif noise_db > 75:
            env_state = "Noisy"      # 嘈杂 (如地铁)
        else:
            env_state = "Normal"     # 正常 (如办公室)
            
        # 划分歌曲响度状态
        if song_lufs < -14:
            song_state = "Soft"      # 轻柔 (如钢琴曲、民谣)
        elif song_lufs > -8:
            song_state = "Loud"      # 炸耳 (如摇滚、EDM)
        else:
            song_state = "Balanced"  # 均衡 (标准流行乐)
            
        return f"{env_state}_{song_state}"

    def get_action(self, noise_db, song_lufs):
        """决策层：根据当前状态选择动作"""
        state = self._discretize_state(noise_db, song_lufs)
        
        # 如果遇到新状态，初始化该状态的 Q 值
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

        # Epsilon-Greedy 策略
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions) # 随机探索
        else:
            # 利用已有经验：选择当前状态下 Q 值最高的动作
            action = max(self.q_table[state], key=self.q_table[state].get)
            
        return action, state

    def learn(self, state, action, reward, next_state):
        """学习层：根据用户的反馈（奖励/惩罚）更新 Q-Table"""
        if next_state and next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        # 获取当前 Q 值
        predict = self.q_table[state][action]
        
        # 计算目标 Q 值 (贝尔曼方程)
        if next_state:
            target = reward + self.gamma * max(self.q_table[next_state].values())
        else:
            target = reward # 终端状态
            
        # 更新 Q-Table
        self.q_table[state][action] += self.alpha * (target - predict)

# ==========================================
# 模拟运行环境与用户交互
# ==========================================
def simulate_user_environment():
    agent = HarmoniQAgent()
    
    print("🎵 HarmoniQ Agent 启动...")
    print("开始模拟用户听歌场景 (目标：Agent 需学会适应用户在不同环境下的音量偏好)\n")
    
    # 模拟用户的“真实偏好”（Agent 不知道这个，需要自己学出来）
    # 规则1: 在嘈杂(Noisy)环境听轻柔(Soft)歌曲，用户希望音量 +4
    # 规则2: 在安静(Quiet)环境听炸耳(Loud)歌曲，用户希望音量 -4
    def get_user_feedback(state, agent_action):
        expected_action = 0
        if state == "Noisy_Soft": expected_action = 4
        elif state == "Quiet_Loud": expected_action = -4
        
        # 计算 Agent 动作与用户期望的偏差
        error = abs(expected_action - agent_action)
        if error == 0:
            return 10, "👍 完美！用户没有按音量键。"
        elif error <= 2:
            return 2, "👌 还行，用户微调了一下。"
        else:
            return -10, "😠 糟糕！吓到用户或听不清，用户大骂并狂按音量键！"

    # 训练循环：模拟切歌 500 次
    epochs = 500
    for epoch in range(epochs):
        # 随机生成场景（噪音 dB，歌曲 LUFS）
        noise_db = random.uniform(30, 90)
        song_lufs = random.uniform(-18, -4)
        
        # 1. Agent 做出决策
        action, current_state = agent.get_action(noise_db, song_lufs)
        
        # 2. 获取用户反馈 (Reward)
        reward, user_reaction = get_user_feedback(current_state, action)
        
        # 3. Agent 学习并更新记忆
        # 简化版：这里假设每个动作独立，暂不考虑下一个状态 next_state 的链式影响
        agent.learn(current_state, action, reward, next_state=None)

        # 打印部分训练过程
        if epoch < 5 or epoch > epochs - 5:
            print(f"[{'前期摸索' if epoch < 5 else '后期成熟'}] 第 {epoch+1} 首歌 | 场景: {current_state}")
            print(f"   Agent 调整: {action:+d} dB -> 用户反馈: {user_reaction}")
            
        if epoch == 5:
            print("\n... Agent 正在后台疯狂学习中 (静默模拟 490 次切歌) ...\n")
            # 随着时间推移，降低探索率，让 Agent 越来越稳重
            agent.epsilon = 0.01 

    # 打印最终学到的“大脑模型”
    print("\n🧠 训练结束。查看 Agent 沉淀的用户画像 (Q-Table 片段):")
    target_states = ["Noisy_Soft", "Quiet_Loud"]
    for state in target_states:
        if state in agent.q_table:
            best_action = max(agent.q_table[state], key=agent.q_table[state].get)
            print(f" -> 遇到 [{state}] 场景时，Agent 认为最优调整方案是: {best_action:+d} dB")

if __name__ == "__main__":
    simulate_user_environment()