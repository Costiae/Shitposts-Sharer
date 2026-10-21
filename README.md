# Shitposts Sharer（搬屎机器人）

AstrBot 扩展：监听已注册群的全部消息，识别"潜在的屎"（合并转发、图片、B 站分享、离谱长文本），
分析它和上下文的关系，比对各群的"吃屎偏好"与"可信度"，把屎搬到对味且够格的群。

> 适配 AstrBot **4.26+**（在 4.29.0 上开发）。

## 它怎么工作

```
群消息 ──► 粗筛闸门 ──► 上下文窗口(90s) ──► 多模态 LLM 判屎
                                              │
              ┌───────────────────────────────┘
              ▼
        关键词向量(屎) ⇄ 关键词向量(群口味)  ── 粗筛候选群
              │
              ▼
        LLM 复核候选群口味契合度  ── 过阈值就搬，多个群都过就都搬
              │
              ▼
        去重(内容指纹) ──► 跨群发送（带上下文）
```

### 1. 什么算"屎"

粗筛很松，真正的判断交给多模态 LLM：

- **合并转发**（聊天记录）
- **图片**（连同配文一起送给视觉模型）
- **B 站分享**（BV 号 / b23 短链 / QQ 小程序卡片，另附视频标题、UP 主、时长、播放量）
- **够长的纯文本**（默认 ≥ 60 字）
- 其余短句只进上下文，不送审

"是不是屎"没有硬编码标准，全部由提示词决定，可以在 WebUI 里改。

### 2. 上下文关系

命中候选后，插件保留前文（默认最近 40 条），并继续吸收随后 90 秒的消息，
统一交给模型判断：这坨屎是被群友顺着什么话题引出来的？群友后来怎么接的？
搬运时可以**把这段讨论一起带过去**（默认开，`with_context`）。

### 3. 群档案与可信度

每个群登记：

| 字段 | 说明 |
| --- | --- |
| 方向 | `in` 只收 / `out` 只发 / `both` 进出 |
| 口味 | 关键词向量，由该群**吃过的屎**（判为 `is_shit` 的内容）累积而成 |
| 可信度 | 由**样本量**、**偏移度**、**采纳率**共同决定 |

可信度公式（`core/profile.py`）：

```
可信度 = 样本饱和度 × 稳定度 × (0.7 + 0.3 × 采纳率)

样本饱和度 = N / (N + k)          N = √(条数 × 字数/40)，k 默认 10
稳定度     = 1 - 0.6 × 偏移度     偏移度 = 1 - cos(旧口味, 新样本)
采纳率     = 采纳 / (采纳 + 拒绝)
```

"可信度"只是**防样本过少而武断搬屎**的温和惩罚，不是精度核心。

### 4. 匹配与搬运

两种匹配来源**都实现了**，由配置开关组合：

- **关键词向量**（默认开，零成本）：余弦相似度过阈值即合格，阈值下方留容差带
- **LLM 复核**（默认开）：关键词粗筛出前 N 个群，把候选清单注入提示词让模型打分

门槛：`口味匹配分` 与 `群可信度` 分别过各自阈值 → 全部搬运（可设 `max_targets` 限流）。
同一条屎对同一个群只搬一次；**别人已经先搬过的也会被内容指纹拦下**。

## 安装

```bash
# 放进 AstrBot 的插件目录
cd <AstrBot>/data/plugins
git clone https://github.com/Costiae/Shitposts-Sharer.git astrbot_plugin_shitposts_sharer
```

或在 WebUI「插件市场 → 安装插件」填仓库地址。
Docker 部署时插件目录通常是挂载卷，`git clone`/`git pull` 后重载插件即可，无需重建镜像。

## 使用

所有指令为纯 ASCII。

**`<target>` 是目标群**，省略时默认当前会话。因为搬屎的接收方是静默的——
群友不会、也不该去配置"自己这个群怎么收屎"，实际使用者是坐在 A 群里管 B 群的运营者。

| `<target>` 写法 | 例子 |
| --- | --- |
| 省略 | `/shit off` → 停掉当前群 |
| 群号 | `/shit off 200002` |
| 完整 UMO | `/shit off aiocqhttp:GroupMessage:200002` |
| 序号 | `/shit off #2`（配合 `/shit list`） |
| 群名片段 | `/shit off 吃屎甲` |

| 指令 | 说明 |
| --- | --- |
| `/shit register <in\|out\|both> [target] [threshold]` | 加入搬运网络，默认 `both`，阈值可写 `0.8` 或 `80%` |
| `/shit unregister [target]` | 移出网络 |
| `/shit on <target>` | 开启该群搬运 |
| `/shit off <target>` | 暂停该群搬运（静默） |
| `/shit status [target]` | 口味、可信度、是否够格收屎 |
| `/shit list` | 所有群一览（带 `#序号`） |
| `/shit threshold <0-1\|xx%> [target]` | 设单群阈值 |
| `/shit purge [target]` | 彻底删除该群记录（仅管理员） |
| `/shit help` | 帮助 |
| `/shoot <文本>` | 管理员：把一段文本直接丢进流水线 |

**权限**：操作当前群不需要管理员（群友自助注册）；`<target>` 指向别的群时必须是管理员。

`/shit` 的别名：`/spam`、`/ss`。

## 配置要点（WebUI）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | true | 总开关 |
| `watch_only_registered` | true | 只处理已注册群 |
| `shit_min_intensity` | 0.6 | 离谱分低于此值不搬 |
| `window_seconds` | 90 | 上下文聚合窗口 |
| `confidence_k` | 10 | 可信度下限参数 |
| `keyword_threshold` / `keyword_tolerance` | 0.75 / 0.10 | 关键词匹配的阈值与容差带 |
| `llm_rerank` | true | 用 LLM 复核候选群 |
| `with_context` | true | 搬运时附带群友讨论 |
| `anon_source` | true | 不显示屎来自哪个群 |
| `dry_run` | false | 演练模式：只判定记录，不真发 |
| `shit_prompt` / `context_prompt` / `group_prompt` / `extra_prompt` | 空 | 自定义提示词 |

## 依赖

- 必装：无（只用 AstrBot 自带能力）
- 可选：`astrbot_plugin_bilibili` —— 装了就走它的 `BiliClient` 取视频信息；
  没装则直连官方 `x/web-interface/view`；都失败时降级为只有 BV 号，不阻塞流程
- Python 包：无额外依赖（`aiohttp` 走 AstrBot 自带环境）

## 开发

```bash
python -m unittest discover -s tests -t . -v
```

> 本机没有 `astrbot` 包，测试用替身覆盖 `Plain` 组件；
> 核心逻辑（向量、可信度、路由、存储、上下文窗口、流水线）全部可在纯 Python 下测试。

目录：

```
main.py            # Star 入口：注册、事件监听、指令
metadata.yaml      # 插件元数据
_conf_schema.json  # WebUI 配置表单
core/
  models.py        # 数据结构（GroupProfile / ShitRecord / Message ...）
  storage.py       # SQLite：群、屎、搬运记录、设置
  config.py        # 配置读取与阈值解析
  vector.py        # 关键词向量：分词、加权、余弦、容差打分
  profile.py       # 口味更新与可信度公式
  similarity.py    # 关键词/LLM 两种匹配来源
  detector.py      # 候选识别（转发/图片/B站/长文本）
  bili.py          # B 站信息抓取（软依赖 + 直连 + 降级）
  accumulator.py   # 上下文窗口
  llm.py           # LLM 调用与 JSON 解析
  analyzer.py      # 判屎 / 上下文关系 / 群契合度 三套提示词
  router.py        # 搬运路由
  sender.py        # 跨群消息组装与发送
  commands.py      # /shit 指令（纯 ASCII）
  pipeline.py      # 串起一切
tests/             # unittest
```
