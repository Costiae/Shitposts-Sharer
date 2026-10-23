"""Shitposts Sharer 核心逻辑。

模块划分：
- models      : 数据结构（群档案、屎、上下文、分析结果）
- storage     : SQLite 持久化（群表、屎表、搬运记录、设置）
- vector      : 关键词向量的构建、持久化与余弦相似度
- terms       : 口味词清洗（停用词、别名归一、同根词合并）
- embed       : 调用 AstrBot 的 Embedding 服务商（可用时）
- similarity  : 向量相似度与“关键词”来源
- detector    : 候选屎的识别（合并转发 / 图片 / B站 / 文本）
- bili        : B 站信息抓取（软依赖 astrbot_plugin_bilibili，自带兜底）
- context     : 上下文窗口冷却与话题聚合
- analyzer    : 调用 LLM 判断“是不是屎”、尾巴与话题、群体符合度
- router      : 搬运路由（谁进来谁出去、阈值、去重、目标群选择）
- commands    : 群注册与调试指令
"""
