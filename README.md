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

- **合并转发**（聊天记录，按 `forward_min_nodes` 过滤过短的）
- **图片**（连同配文一起送给视觉模型）
- **语音 / 视频 / 文件**
- **B 站分享**（BV 号 / b23 短链 / QQ 小程序卡片，另附视频标题、UP 主、时长、播放量）
- **普通链接分享**（`detect_links`，默认开）
- **够长的纯文本**（默认 ≥ 60 字）
- **松闸门**（`loose_detect`，默认开）：
  - 中等长度文本（默认 ≥ 24 字）
  - 带梗的短句（`哈哈哈哈哈` / `草` / `逆天` / `绷不住` / `笑死` …）

"是不是屎"没有硬编码标准，全部由提示词决定，可以在 WebUI 里改。
粗筛只负责**不漏**：越靠后的规则越松，模型有最终否决权。

### 2. 上下文关系

命中候选后，插件保留前文（默认最近 40 条），并继续吸收随后 90 秒的消息，
统一交给模型判断：这坨屎是被群友顺着什么话题引出来的？群友后来怎么接的？
搬运时可以**把这段讨论一起带过去**（默认开，`with_context`）。

### 3. 群档案与可信度

每个群登记：

| 字段 | 说明 |
| --- | --- |
| 方向 | `receive` 只收 / `emit` 只发 / `both` 进出 |
| 口味 | 关键词向量，由该群**吃过的屎**（判为 `is_shit` 的内容）累积而成 |
| 可信度 | 由**样本量**、**偏移度**、**采纳率**共同决定 |

> **方向描述的是「这个群」，不是插件。**
>
> - `receive` —— 这个群收别处的屎；它自己产的内容不会被搬走
> - `emit` —— 这个群产屎，被别人搬走；它不接收
> - `both` —— 既产又收
>
> 字段值与 `can_receive` / `can_emit` 同名对齐。旧版本的 `in` / `out` 会在
> 插件加载时自动迁移为 `receive` / `emit`，已注册的群无需重新登记。

可信度公式（`core/profile.py`）：

```
可信度 = 样本饱和度 × 稳定度 × (0.7 + 0.3 × 采纳率)

样本饱和度 = N / (N + k)          N = √(条数 × 字数/40)，k 默认 10
稳定度     = 1 - 0.6 × 偏移度     偏移度 = 1 - cos(旧口味, 新样本)
采纳率     = 采纳 / (采纳 + 拒绝)
```

偏移度会做指数平滑后存进群档案（`drift`），因此**路由判定、`/shit status`、
`/shit debug` 用的是同一个 `GroupProfile.confidence()`**，不会出现
“状态说够格、实际却不搬”的矛盾。老库在加载时自动补 `drift` 列。

"可信度"只是**防样本过少而武断搬屎**的温和惩罚，不是精度核心。

### 4. 匹配与搬运

两种匹配来源**都实现了**，由配置开关组合：

- **关键词向量**（默认开，零成本）：余弦相似度过阈值即合格，阈值下方留容差带
- **相似关键词合并**：`抽象 / 抽象话 / 抽象人` 这类同一概念的写法只计一次权，
  避免一个方向被重复累加（阈值 `keyword_merge_threshold`，默认 0.6）。
  **每次新增关键词都会把整张口味表重新合并一遍**，之前登记过的老词条也会收敛。
- **LLM 复核**（默认开）：关键词粗筛出前 N 个群，把候选清单注入提示词让模型打分

门槛：`口味匹配分` 与 `群可信度` 分别过各自阈值 → 全部搬运（可设 `max_targets` 限流）。
同一条屎对同一个群只搬一次；**别人已经先搬过的也会被内容指纹拦下**。

#### 图片标签与匹配

色图、二次元图也算“屎”，和文本走同一套搬运流程，但多一道独立门：

- 模型给图片打**视觉标签**：角色名、作品/IP、属性（发色/服装/姿势/场景）、画风
  （`image_tagging`，默认开）
- 图片的标签向量必须单独达到 `image_match_threshold`（默认 0.6，容差 0.12）
  才允许搬给该群；**文字话题对上了也不能代替图片匹配**
- 群的口味同样会从图片标签里学习：某个群吃过的图，会让它以后更容易收到同角色/IP 的图
- 成人向图片会在标签里带上 `NSFW`，方便按群口味区分
- **安全硬拦截**：涉及未成年人的性化内容永不搬运、不入库、不参与口味学习，
  没有配置开关

图片的内容指纹使用图片 URL + 标签，避免“同标签不同图”被误判成重复。

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
| `/shit register <receive\|emit\|both> [target] [threshold]` | 加入搬运网络，默认 `both`，阈值可写 `0.8` 或 `80%` |
| `/shit unregister [target]` | 移出网络 |
| `/shit on <target>` | 开启该群搬运 |
| `/shit off <target>` | 暂停该群搬运（静默） |
| `/shit status [target]` | 口味、可信度、是否够格收屎 |
| `/shit list` | 所有群一览（带 `#序号`） |
| `/shit threshold <0-1\|xx%> [target]` | 设单群阈值 |
| `/shit purge [target]` | 彻底删除该群记录（仅管理员） |
| `/shit help` | 帮助 |

调试指令（全部仅管理员，会真实发消息）：

| 指令 | 说明 |
| --- | --- |
| `/shit debug [target]` | 摊开每一道闸门：插件开关、群是否注册、方向、可信度 vs 阈值、LLM 是否可用、待处理窗口数 |
| `/shit tags [target]` | 导出群的口味词表（省略则汇总所有群） |
| `/shit relay [target]` | **无条件搬出**：忽略方向/口味/可信度/LLM 复核，直接推到目标群 |
| `/shit intake [target]` | **无条件搬入**：只更新目标群口味，不往外发 |
| `/shoot <文本>` | 把一段文本直接丢进判定流水线 |

> `relay` / `intake` 描述的是**这条消息**的去向，与 `receive` / `emit`
> （描述**群**的角色）是两个不同的维度，所以刻意用了不同的词。

**权限**：操作当前群不需要管理员（群友自助注册）；`<target>` 指向别的群时必须是管理员。

`/shit` 的别名：`/spam`、`/ss`。

## 配置要点（WebUI）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `log_level` | normal | `off` / `normal` / `full`，见下节 |
| `judge_timeout_sec` | 180 | 单条判定超时，防止一条卡死拖垮收盘循环 |
| `enabled` | true | 总开关 |
| `watch_only_registered` | true | 只处理已注册群 |
| `text_min_chars` | 60 | 长文本触发阈值 |
| `loose_detect` | true | 松闸门：中等文本 + 带梗短句也送审 |
| `soft_text_min_chars` | 24 | 松闸门的文本阈值 |
| `detect_links` / `detect_media` | true | 普通链接、语音/视频/文件是否送审 |
| `expand_forward` | true | 自动调 `get_forward_msg` 展开合并转发 |
| `shit_min_intensity` | 0.45 | 离谱分低于此值不搬 |
| `window_seconds` | 90 | 上下文聚合窗口 |
| `confidence_k` | 10 | 可信度下限参数 |
| `keyword_threshold` / `keyword_tolerance` | 0.75 / 0.10 | 关键词匹配的阈值与容差带 |
| `keyword_merge_threshold` | 0.6 | 相似关键词合并阈值（每次更新重合并整张口味表） |
| `image_llm` | true | 把图片一起送给视觉模型 |
| `image_tagging` | true | 给图片打角色 / IP / 属性 / 风格标签 |
| `max_images_per_message` | 4 | 单条消息最多分析几张图 |
| `image_match_threshold` / `image_match_tolerance` | 0.6 / 0.12 | 图片标签独立门槛：达不到不搬，文字匹配不能代替 |
| `llm_rerank` | true | 用 LLM 复核候选群 |
| `with_context` | true | 搬运时附带群友讨论 |
| `anon_source` | true | 不显示屎来自哪个群 |
| `dry_run` | false | 演练模式：只判定记录，不真发 |
| `shit_prompt` / `context_prompt` / `group_prompt` / `extra_prompt` | 空 | 自定义提示词 |

## 日志

所有日志前缀 `[shit]`，格式 `[shit] <阶段> | <内容>`，方便 `grep '\[shit\]'`。

| 级别 | 输出内容 |
| --- | --- |
| `off` | 只写最终搬运结果 |
| `normal`（默认） | 每条消息**是否被俘获**，没被俘获的说明**卡在哪道闸门** |
| `full` | 上面全部 + 解码后的分段、转发展开情况、模型输入摘要、模型原始判定、关键词、候选群打分、逐群投递结果、完整 trail |

阶段依次为 `entry`（AstrBot 交给插件）→ `group`（群是否注册/启用）→
`decode`（消息解析与转发展开）→ `detect`（是否可疑）→ `judge`（LLM 判定）→
`route`（选目标群）→ `result`。

典型输出：

```
[shit] detect | SKIP not suspicious need=60chars soft=24 got=1 kinds=['plain']
[shit] judge  | SKIP intensity 0.20 < min 0.45 (reason='不够劲')
[shit] judge  | SKIP model says NOT shit (intensity=0.10 reason='日常闲聊')
[shit] route  | SKIP no candidate group passed the keyword+confidence gate
[shit] route  | DELIVERED to aiocqhttp:GroupMessage:200002 (score=0.80 confidence=0.86)
```

**关键**：如果群里发了消息，但日志里连 `[shit] entry` 都没有，说明消息在到
插件之前就被 AstrBot 拦掉了（未唤醒 / 白名单 / 平台没上报）—— 这不是插件的问题。

## 收盘循环的健壮性

「抓到了东西，但之后既没有 result 也没有错误」是这个插件最容易出现的故障。
原因一定是 90 秒窗口没能收盘。`_ticker` 为此做了四重保护：

| 保护 | 日志 | 作用 |
| --- | --- | --- |
| 异常不退出循环 | `ticker \| CRASHED: ...` | 一次 `closed database` 不会永久废掉收盘机制 |
| 取消也留痕 | `ticker \| CANCELLED` | 被谁取消的不再是谜 |
| 上一轮未返回 | `ticker \| flush still running ...` | 长 LLM 重试导致的空转不再伪装成安静 |
| 单条超时 | `judge \| TIMEOUT after 180s` | 一条卡住不拖死整个循环 |

`/shit debug` 会直接给出 `ticker alive` 和 `flush blocked` 两项状态，不用猜。

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
