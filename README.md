
Automatically adjust the volume of music
# 🎵 HarmoniQ: 基于上下文感知的情境化音量 Agent

## 📖 项目简介

**HarmoniQ** 是一个运行在音乐播放器后台的自主智能体（Agent）原型项目。它的核心目标是：**消除切歌时的“音量刺客”现象，并利用强化学习（Reinforcement Learning）自动拟合用户的听觉偏好。**

传统的音乐软件仅依赖静态的音频响度标准化（如固定的 LUFS 调整），而 HarmoniQ 能够综合分析**环境噪音**与**歌曲本身的情感起伏**，动态且平滑地调节每首歌曲的初始音量，为用户提供无缝、沉浸式的听觉体验。

## ✨ 核心特性

* **环境感知 (Context Awareness):** 实时根据外部环境噪音（如安静的卧室、嘈杂的地铁）划分状态。
* **音频动态评估 (Audio Dynamics):** 根据预读取的下一首歌曲的感知响度（LUFS）判断曲风。
* **自主强化学习 (Q-Learning Brain):** * Agent 采取动作（调整音量 dB）。
  * 监听用户反馈（若用户手动微调音量则记为惩罚 `-10`，若无操作则记为奖励 `+10`）。
  * 通过贝尔曼方程不断更新 Q-Table，最终学会针对不同场景的最优音量补偿策略。
* **开箱即用 (Standalone):** 支持打包为独立的可执行文件，无需配置 Python 环境即可运行模拟测试。

## ⚙️ 算法原理

本项目的核心逻辑基于 **强化学习 (Q-Learning)** 构建：
* **状态空间 (State):** `[环境噪音级别 (Quiet/Normal/Noisy)]` + `[歌曲响度级别 (Soft/Balanced/Loud)]`
* **动作空间 (Action):** `[-4dB, -2dB, 0dB, +2dB, +4dB]` 
* **奖励机制 (Reward):** 通过模拟用户的真实偏好进行打分，指导 Agent 更新策略。

## 🚀 快速开始

### 1. 环境依赖
本项目原型仅依赖 Python 标准库以及 `numpy` 进行数值计算。
```bash
pip install numpy
