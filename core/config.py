"""配置读取。

AstrBot 把 _conf_schema.json 渲染成 WebUI 表单，插件里以 dict 形式拿到配置。
这里把所有键收敛到一个带默认值的类，业务代码只读属性，不直接碰 dict。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULTS: dict[str, Any] = {
    # ---- 日志 ----
    # off    = 只写搬运结果
    # normal = 每条可疑消息的结论 + 被挡原因（默认）
    # full   = 每一层每一步，含模型输入摘要与原始输出
    "log_level": "normal",
    # ---- 触发 ----
    "enabled": True,
    "watch_only_registered": True,
    "text_min_chars": 60,
    "forward_min_nodes": 1,
    "expand_forward": True,
    "image_llm": True,
    "max_images_per_message": 4,
    "trigger_cooldown_sec": 30,
    # ---- 上下文 ----
    "history_max_messages": 40,
    "window_seconds": 90,
    "max_messages_per_shit": 20,
    # ---- 判断 ----
    # 0.5 = “明显抽象，值得跨群分享”的锚点下沿。原来的 0.6 在模型给保守分时
    # 会挡掉大量本该搬走的内容；配合提示词里的锚点，0.45 足够稳。
    "shit_min_intensity": 0.45,
    "shit_prompt": "",
    "context_prompt": "",
    "group_prompt": "",
    "extra_prompt": "",
    # ---- 信任与路由 ----
    "confidence_k": 10.0,
    "threshold_overall": 0.0,
    "keyword_threshold": 0.75,
    "keyword_tolerance": 0.10,
    "llm_rerank": True,
    "llm_candidates": 3,
    "max_targets": 0,
    # ---- 搬运 ----
    "with_context": True,
    "max_context_messages": 6,
    "anon_source": True,
    "source_label": "【搬屎】",
    "dry_run": False,
    # ---- 模型 ----
    "analysis_provider_id": "",
    "embedding_provider_id": "",
    # ---- 其他 ----
    "bili_info": True,
    "proxy": "",
}


@dataclass(slots=True)
class Settings:
    raw: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        value = self.raw.get(key, DEFAULTS.get(key))
        if value is None:
            return DEFAULTS.get(key, default)
        return value

    # 便捷读取器
    @property
    def watch_only_registered(self) -> bool:
        return bool(self.get("watch_only_registered"))

    @property
    def text_min_chars(self) -> int:
        return int(self.get("text_min_chars"))

    @property
    def forward_min_nodes(self) -> int:
        return max(1, int(self.get("forward_min_nodes")))

    @property
    def image_llm(self) -> bool:
        return bool(self.get("image_llm"))

    @property
    def max_images_per_message(self) -> int:
        return max(0, int(self.get("max_images_per_message")))

    @property
    def window_seconds(self) -> float:
        return max(5.0, float(self.get("window_seconds")))

    @property
    def max_messages_per_shit(self) -> int:
        return max(1, int(self.get("max_messages_per_shit")))

    @property
    def confidence_k(self) -> float:
        return max(1.0, float(self.get("confidence_k")))

    @property
    def keyword_threshold(self) -> float:
        return min(1.0, max(0.0, float(self.get("keyword_threshold"))))

    @property
    def keyword_tolerance(self) -> float:
        return min(1.0, max(0.0, float(self.get("keyword_tolerance"))))

    @property
    def extra_prompt(self) -> str:
        value = self.get("extra_prompt")
        return str(value or "").strip()

    # ---- 阈值 ----
    def global_threshold(self) -> float:
        return max(0.0, float(self.get("threshold_overall") or 0.0))

    def threshold_for(self, group_threshold: float) -> float:
        """单群阈值优先；单群未设置（或 <=0）时回落到全局阈值。"""
        if group_threshold and group_threshold > 0:
            return group_threshold
        return self.global_threshold()

    def parse_threshold(self, raw: str) -> float:
        """把群指令里的阈值文本解析成数值（支持 0.8 / 80%）。"""
        text = (raw or "").strip()
        if not text or text in ("-", "global", "默认"):
            return 0.0
        if text.endswith("%"):
            value = float(text[:-1]) / 100.0
        else:
            value = float(text)
        if value > 1.0:
            value = value / 100.0
        return min(1.0, max(0.0, value))
