# Shitposts Sharer（搬屎机器人）

AstrBot 扩展：监听已注册群的全部消息，识别"潜在的屎"（合并转发、图片、B 站分享、离谱长文本），
分析它和上下文的关系，比对各群的"吃屎偏好"与"可信度"，把屎搬到对味且够格的群。

图片也会被抓取、打视觉标签，并走独立的图片匹配门槛；每个群还可以单独关掉图片搬运。

> 适配 AstrBot **4.26+**（在 4.29.0 上开发）。

## 它怎么工作

```
群消息 ──► 粗筛闸门 ──► 上下文窗口(90s) ──► 多模态 LLM 判屎 + 打图片标签
                                              │
                                              ├─ minor_sexual=true ──► 硬拦截，不入库
                                              ▼
                        文本关键词向量        图片标签向量
                              │                    │
                              ▼                    ▼
                    文本匹配 ⇄ 群口味      图片匹配 ⇄ 群口味
                              │                    │
                              └────────┬───────────┘
                                       ▼
                    LLM 复核候选群 + 可信度/阈值/方向/单群图片开关
                                       │
                                       ▼
                              去重(内容指纹) ──► 跨群发送（带上下文）
```

### 1. 什么算"屎"

粗筛很松，真正的判断交给多模态 LLM：

- **合并转发**（聊天记录，按 `forward_min_nodes` 过滤过短的）
- **图片**（连同配文一起送给视觉模型；要求 `image_llm=true`）
- **语音 / 视频 / 文件**（`detect_media`）
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

同一个群在窗口期内连续命中多条候选时，会各自开一个窗口，后一条不会覆盖前一条。
后续消息会同时进入这些窗口，直到各自到点收盘。

### 3. 群档案、方向与可信度

每个群登记：

| 字段 | 说明 |
| --- | --- |
| 方向 | `receive` 只收 / `emit` 只发 / `both` 进出 |
| 口味 | 关键词向量，由该群**自己产的**、被判为 `is_shit` 的内容累积而成 |
| 图片开关 | 该群是否参与图片搬运，默认开；关闭后不产也不收图片，文本不受影响 |
| 可信度 | 由**样本量**、**偏移度**、**采纳率**共同决定 |

> **方向描述的是「这个群」，不是插件。**
>
> - `receive` —— 这个群收别处的屎；它自己产的内容不会被搬走
> - `emit` —— 这个群产屎，被别人搬走；它不接收
> - `both` —— 既产又收
>
> 字段值与 `can_receive` / `can_emit` 同名对齐。旧版本的 `in` / `out` 会在
> 插件加载时自动迁移为 `receive` / `emit`，已注册的群无需重新登记。

**冷启动**：匹配分来自「屎的向量」和「群口味向量」的余弦。口味为空的群，
关键词分是 0，因此不会收到任何搬运。纯 `receive` 群自己不产内容，也就不会
积累原生口味，建议：

1. 先以 `both` 运行一段时间，让群自己产生几条被判为屎的内容；或
2. 管理员用 `/shit intake <target>` 把一条有代表性的消息喂进目标群口味；然后
3. 再切到 `receive`。

另外，`threshold_overall > 0` 时，纯 `receive` 群没有原生样本，可信度恒为 0，
会被可信度门槛挡住。这类群建议保持全局阈值为 0，或先积累原生样本。

可信度公式（`core/profile.py`）：

```
可信度 = 样本饱和度 × 稳定度 × (0.7 + 0.3 × 采纳率)

样本饱和度 = N / (N + k)          N = √(条数 × 字数/40)，k 默认 10
稳定度     = 1 - 0.6 × 偏移度     偏移度 = 1 - cos(旧口味, 新样本)
采纳率     = 采纳 / (采纳 + 拒绝)
```

偏移度会做指数平滑后存进群档案（`drift`），因此**路由判定、`/shit status`、
`/shit debug` 用的是同一个 `GroupProfile.confidence()`**，不会出现
“状态说够格、实际却不搬”的矛盾。老库在加载时自动补 `drift` 和
`image_enabled` 列。

> 拒绝计数目前没有自动反馈入口（`apply_rejection` 只是预留），
> 所以采纳率通常是 100%。它不会凭空降低可信度。

"可信度"只是**防样本过少而武断搬屎**的温和惩罚，不是精度核心。

### 4. 匹配与搬运

匹配由两层组成：

- **关键词向量**（始终启用，零成本）：余弦相似度过阈值即合格，阈值下方留容差带
- **LLM 复核**（`llm_rerank`，默认开）：关键词粗筛出前 N 个群，把候选清单注入提示词让模型打分；
  只有当前上下文里存在可用的 chat provider 时才会执行

相似关键词会合并：`抽象 / 抽象话 / 抽象人` 这类同一概念的写法只计一次权，
避免一个方向被重复累加（阈值 `keyword_merge_threshold`，默认 0.6）。
**每次新增关键词都会把整张口味表重新合并一遍**，之前登记过的老词条也会收敛。

门槛分三层，互不替代：

1. **关键词分**：`keyword_threshold` / `keyword_tolerance`，低于容差带直接淘汰；
2. **图片分**：图片额外要求视觉标签向量达到 `image_match_threshold` /
   `image_match_tolerance`，文字话题对上了也不能代替图片匹配；
3. **可信度**：群可信度必须 ≥ `threshold_overall`（单群可用 `/shit threshold` 覆盖）。

全部通过才会搬；多个群都过就都搬（可用 `max_targets` 限流）。
同一条屎对同一个群只搬一次；**别人已经先搬过的也会被内容指纹拦下**。

#### 图片标签与匹配

色图、二次元图也算“屎”，和文本走同一套搬运流程，但多一道独立门：

- 模型给图片打**视觉标签**：角色名、作品/IP、属性（发色/服装/姿势/场景）、画风
  （`image_tagging`，默认开；要求 `image_llm=true` 且模型支持视觉）
- 图片的标签向量必须单独达到 `image_match_threshold`（默认 0.6，容差 0.12）
  才允许搬给该群；**文字话题对上了也不能代替图片匹配**
- `image_tagging` 开着但模型没有返回可用标签时，图片会被图片门槛拒绝，
  不会退化成纯文本匹配放行
- 关掉 `image_tagging` 后，图片退回纯文本匹配，但 `image_llm` 仍需开启，
  否则无法做视觉安全检查，图片消息会直接跳过
- 群自己发过的图会进入该群口味；目标群收到的图不会反向改写它的口味，
  避免搬运内容把接收群的口味带歪
- 成人向图片会在标签里带上 `NSFW`，方便按群口味区分
- **每个群有独立的图片开关**（默认开）：`/shit image on|off [target]`。
  关闭后这个群既不产图片屎，也不收图片屎，文本搬运完全不受影响；
  `/shit status` 和 `/shit list` 会显示图片开关状态
- **安全硬拦截**：直接图片中涉及未成年人的性化内容永不自动搬运、不入库、
  不参与口味学习，没有配置开关；`/shit relay` 强制搬运图片时也会额外做一次
  安全检查，无法确认时一律拒绝

图片的内容指纹使用图片 URL + 标签，避免“同标签不同图”被误判成重复。

> 合并转发里的图片目前不单独拆出来打视觉标签，仍按“转发整体”走文本流程；
> 图片开关和直图安全门只作用于顶层图片消息。

## 安装

```bash
# 放进 AstrBot 的插件目录
cd <AstrBot>/data/plugins
git clone https://github.com/Costiae/Shitposts-Sharer.git astrbot_plugin_shitposts_sharer
```

或在 WebUI「插件市场 → 安装插件」填仓库地址。
Docker 部署时插件目录通常是挂载卷，`git clone`/`git pull` 后重载插件即可，无需重建镜像。

数据库默认在 AstrBot 的 `data/plugin_data/shitposts_sharer/shitposts.sqlite3`；
拿不到 AstrBot data 路径时退到插件目录下的 `data/`。

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
| `/shit on [target]` | 开启该群搬运 |
| `/shit off [target]` | 暂停该群搬运（静默） |
| `/shit image [on\|off] [target]` | 单群图片搬运开关；省略 on/off 时查看状态（别名 `img` / `pic`） |
| `/shit status [target]` | 口味、可信度、方向、图片开关、是否够格收屎 |
| `/shit list` | 所有群一览（带 `#序号` 和 `[paused]` / `[image off]` 标记） |
| `/shit threshold <0-1\|xx%> [target]` | 设单群阈值 |
| `/shit purge [target]` | 彻底删除该群记录（仅管理员） |
| `/shit help [basic\|admin\|all]` | 帮助；默认基础指令，`admin` 看调试指令，`all` 看全部 |

调试指令（全部仅管理员）：

| 指令 | 说明 | 会发消息 / 改数据 |
| --- | --- | --- |
| `/shit debug [target]` | 摊开每一道闸门：插件开关、群是否注册、方向、可信度 vs 阈值、LLM 是否可用、待处理窗口数 | 只读 |
| `/shit tags [target]` | 导出群的口味词表（省略则汇总所有群） | 只读 |
| `/shit tidy [target]` | 用 LLM 清理口味词表：删碎片/无效词、合并同义词 | 会改口味，不发消息 |
| `/shit relay [target]` | **无条件搬出**：忽略方向/口味/可信度/LLM 复核；不指定 target 时推到所有已启用群（源群除外） | 会发消息；图片仍过安全检查，遵守图片开关与 `dry_run` |
| `/shit intake [target]` | **无条件搬入**：只更新目标群口味，不往外发 | 会改口味，不发消息 |
| `/shoot <文本>` | 把一段文本直接丢进判定流水线 | 可能发消息 |

> `relay` / `intake` 描述的是**这条消息**的去向，与 `receive` / `emit`
> （描述**群**的角色）是两个不同的维度，所以刻意用了不同的词。

**权限**：

- 操作**当前群**不需要管理员（群友可自助注册、开关自己的群）
- `<target>` 指向别的群时，修改类指令需要管理员
- `debug` / `tags` / `tidy` / `relay` / `intake` / `shoot` 仅管理员
- `status` / `list` 是只读查询，不强制管理员

`/shit` 的别名：`/spam`、`/ss`、`/sps`。QQ 的表情输入有时会把 `/shit`
当成表情搜索吃掉，推荐用 `/sps`（ShitPosts Sharer 的缩写）。

帮助按主题拆成三段（basic / admin / all），每段都控制在 AstrBot
`platform_settings.forward_threshold` 的默认值（1500 字）以内。否则 AstrBot
会把超长回复折叠成 QQ 的“转发消息 / 聊天记录”，既难读，NapCat 在部分版本
里还可能直接发不出去。如果你的 `forward_threshold` 调得比 1500 更低，
可以在 WebUI 里调高它，或者只发 `/shit help basic`。

## 配置（WebUI）

### 触发与输入

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | true | 总开关 |
| `watch_only_registered` | true | 只处理已注册群 |
| `text_min_chars` | 60 | 长文本触发阈值 |
| `loose_detect` | true | 松闸门：中等文本 + 带梗短句也送审 |
| `soft_text_min_chars` | 24 | 松闸门的文本阈值 |
| `detect_links` | true | 普通 http(s) 链接是否送审 |
| `detect_media` | true | 语音 / 视频 / 文件是否送审 |
| `forward_min_nodes` | 1 | 合并转发最少几条才值得看 |
| `expand_forward` | true | 自动调 `get_forward_msg` 展开合并转发 |
| `trigger_cooldown_sec` | 30 | 同一群两次开窗的最小间隔，避免刷屏 |
| `image_llm` | true | 把图片交给视觉模型；关闭后图片不自动搬运 |
| `max_images_per_message` | 4 | 单条消息最多分析几张图 |
| `image_tagging` | true | 给图片打角色 / IP / 属性 / 风格标签，并用独立图片门槛 |

### 上下文与判定

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `history_max_messages` | 40 | 为判断上下文关系保留的历史条数 |
| `window_seconds` | 90 | 命中候选后的聚合窗口（最小 5 秒） |
| `max_messages_per_shit` | 20 | 一个窗口最多吸收多少条后续消息 |
| `judge_timeout_sec` | 180 | 单条判定超时，防止一条卡死拖垮收盘循环 |
| `shit_min_intensity` | 0.45 | 离谱分低于此值不搬 |
| `with_context` | true | 搬运时附带群友讨论 |
| `max_context_messages` | 6 | 附带几条上下文（0 会被当作 6） |
| `shit_prompt` / `context_prompt` / `group_prompt` / `tidy_prompt` / `extra_prompt` | 空 | 自定义提示词；留空用内置 |

### 匹配与路由

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `confidence_k` | 10 | 可信度下限参数；越大越需要原生样本 |
| `threshold_overall` | 0 | 群可信度门槛；0 = 不拦。单群可用 `/shit threshold` 覆盖 |
| `keyword_threshold` / `keyword_tolerance` | 0.75 / 0.10 | 文本关键词匹配的阈值与容差带 |
| `keyword_merge_threshold` | 0.6 | 相似关键词合并阈值（每次更新重合并整张口味表） |
| `tag_stopwords` | 空 | 追加的口味词停用词（逗号/空格/换行分隔）；内置已含时间戳碎片与“玩梗”等元词 |
| `tag_aliases` | 空 | 追加的别名表，每行 `规范词=别名1,别名2`（如 `人工智能=AI`） |
| `image_match_threshold` / `image_match_tolerance` | 0.6 / 0.12 | 图片标签独立门槛；文字匹配不能代替 |
| `llm_rerank` | true | 关键词粗筛后用 LLM 复核候选群 |
| `llm_candidates` | 3 | 进入 LLM 复核的前 N 个候选群 |
| `max_targets` | 0 | 单条屎最多搬几个群；0 = 不限 |

口味词进向量前会先做四步清洗：**停用词**（时间戳碎片、`玩梗` 这类只描述“是不是屎”的元词）、
**别名归一**（`AI` → `人工智能`）、**丢单字**（`站` 只是 `B站` 被旧滑窗切剩的碎片）、
**同根词合并**（`机器人` 吞并 `机器`）。前三步由本地规则即时完成，内置表可以直接用，
也可以在上面两个配置项里追加。

本地规则覆盖不到的同义关系（例如 `代码` / `编程`），用管理员指令 `/shit tidy [target]`
交给模型做一次归并。它只改口味表，不搬消息；模型返回的未知词会被忽略，不会无中生有。

### 搬运与模型

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `anon_source` | true | 不显示屎来自哪个群 |
| `source_label` | 【搬屎】 | 搬运消息前缀；留空回退默认值 |
| `dry_run` | false | 演练模式：只判定记录，不真发；强制搬出也遵守 |
| `analysis_provider_id` | 空 | 分析用模型；留空用 AstrBot 默认对话模型 |
| `embedding_provider_id` | 空 | 保留字段，当前未使用 |
| `bili_info` | true | 抓取 B 站视频标题 / UP 主 / 时长 / 播放量 |
| `proxy` | 空 | 抓 B 站信息时的可选代理，如 `http://127.0.0.1:7890` |

### 日志

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `log_level` | normal | `off` / `normal` / `full`，见下节 |


## 日志

所有日志前缀 `[shit]`，格式 `[shit] <阶段> | <内容>`，方便 `grep '\[shit\]'`。

| 级别 | 输出内容 |
| --- | --- |
| `off` | 只写最终搬运结果；`Trace` 不输出闸门与步骤 |
| `normal`（默认） | 每条消息是否被俘获；没被俘获的直接写出卡在哪道闸门；判定与投递结果 |
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

## 故障排查

### 图片没有被搬

按顺序检查：

1. 源群和目标群的图片开关：`/shit status [target]` 里的 `image:` 是否为 `on`；
2. `image_llm` 是否为 true，分析模型是否支持视觉；
3. 日志里是否有 `image tags: ...`；如果为空，检查模型是否按提示词返回了
   `image_tags` / `characters` / `copyrights`；
4. 目标群是否有图片口味样本：`/shit tags [target]`；
5. `/shit debug [target]` 看图片门槛和可信度；
6. 如果看到 `该群未启用图片搬运`，用 `/shit image on [target]` 打开。

### 抓到了但之后没有结果

大概率是 90 秒窗口没有收盘，或单条判定超时。见下节。

### 新群 / receive 群什么都不收

口味为空时匹配分为 0，这是防误投而不是故障。先给群喂口味样本：
`both` 运行一段时间，或管理员用 `/shit intake <target>` 播种。

### 日志里没有 `entry`

消息在插件之前就被 AstrBot 拦掉了。检查唤醒、白名单、平台适配器。

## 收盘循环的健壮性

「抓到了东西，但之后既没有 result 也没有错误」是这个插件最容易出现的故障。
原因通常是窗口没能收盘。`_ticker` 为此做了四重保护：

| 保护 | 日志 | 作用 |
| --- | --- | --- |
| 异常不退出循环 | `ticker \| CRASHED: ...` | 一次 `closed database` 不会永久废掉收盘机制 |
| 取消也留痕 | `ticker \| CANCELLED` | 被谁取消的不再是谜 |
| 上一轮未返回 | `ticker \| flush still running ...` | 长 LLM 重试导致的空转不再伪装成安静 |
| 单条超时 | `judge \| TIMEOUT after 180s` | 一条卡住不拖死整个循环 |

`/shit debug` 会直接给出 `ticker alive`、`flush blocked` 和 `open windows`，
不用猜。

## 依赖

- 必装：无（只用 AstrBot 自带能力）
- 可选：`astrbot_plugin_bilibili` —— 装了就走它的 `BiliClient` 取视频信息；
  没装则直连官方 `x/web-interface/view`；都失败时降级为只有 BV 号，不阻塞流程
- Python 包：无额外依赖（`aiohttp` 走 AstrBot 自带环境）

## 开发

```bash
python -m unittest discover -s tests -t . -v
```

当前 204 个测试。测试不需要 `astrbot` 包：发送组件用替身覆盖，
核心逻辑（向量、可信度、路由、存储、上下文窗口、流水线、图片门槛、
安全拦截、群图片开关、迁移）全部可在纯 Python 下测试。

目录：

```
main.py            # Star 入口：注册、事件监听、指令
metadata.yaml      # 插件元数据
_conf_schema.json  # WebUI 配置表单
core/
  models.py        # 数据结构（GroupProfile / ShitRecord / Message ...）
  storage.py       # SQLite：群、屎、搬运记录、设置、schema 迁移
  config.py        # 配置读取与阈值解析
  terms.py         # 口味词清洗：停用词、别名归一
  vector.py        # 关键词向量：分词、加权、余弦、容差、全表合并
  profile.py       # 口味更新与可信度公式
  similarity.py    # 关键词匹配与图片标签匹配
  detector.py      # 候选识别（转发/图片/B站/长文本）
  forward.py       # OneBot 合并转发展开
  bili.py          # B 站信息抓取（软依赖 + 直连 + 降级）
  accumulator.py   # 上下文窗口（同群多窗口）
  llm.py           # LLM 调用与 JSON 解析
  analyzer.py      # 判屎 / 上下文关系 / 群契合度 / 图片安全检查
  router.py        # 搬运路由（文本门、图片门、可信度、群开关）
  sender.py        # 跨群消息组装与发送
  targets.py       # <target> 解析（群号 / UMO / #序号 / 群名）
  tracelog.py      # 分阶段日志与 trail
  commands.py      # /shit 指令（纯 ASCII）
  pipeline.py      # 串起一切
tests/             # unittest
```
