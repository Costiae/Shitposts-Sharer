"""核心逻辑单元测试（不依赖 AstrBot 运行时，只用标准库 unittest）。

运行：``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import itertools
import sys
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.accumulator import ContextAccumulator  # noqa: E402
from core.bili import extract_bili_ids  # noqa: E402
from core.config import DEFAULTS, Settings  # noqa: E402
from core.analyzer import parse_analysis  # noqa: E402
from core.detector import (  # noqa: E402
    _component_type,
    is_there_a_shit,
    segment_from_component,
)
from core.llm import extract_json  # noqa: E402
from core.models import (  # noqa: E402
    Direction,
    GroupProfile,
    Kind,
    Message,
    Segment,
    ShitRecord,
)
from core.profile import (  # noqa: E402
    apply_term_rewrite,
    compact_intake,
    confidence_score,
    describe_intake,
    update_intake,
)
from core.router import Router, fingerprint_of  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.terms import TermRules  # noqa: E402
from core.vector import (  # noqa: E402
    blend_into,
    build_vector,
    cosine,
    keyword_score,
    keyword_similarity,
    merge_profile_terms,
    merge_similar_keywords,
    merge_similar_terms,
    tokenize,
)


def make_settings(**overrides) -> Settings:
    raw = dict(DEFAULTS)
    raw.update(overrides)
    return Settings(raw)


def make_group(
    key: str, direction: Direction = Direction.BOTH, native: int = 20
) -> GroupProfile:
    return GroupProfile(
        group_key=key,
        name=key,
        direction=direction,
        intake=build_vector(["抽象", "离谱"], "抽象离谱的玩梗内容", 0.8),
        native_message_count=native,
        native_char_count=native * 50,
    )


def make_shit(text: str, keywords: list[str], group_key: str = "g-source") -> ShitRecord:
    record = ShitRecord(
        shit_id="s1",
        source_group=group_key,
        kind=Kind.TEXT,
        text_blob=text,
        keywords=keywords,
        vector=build_vector(keywords, text, 0.8),
    )
    record.fingerprint = fingerprint_of(record)
    return record


def make_image_shit(url: str = "http://x/a.png") -> ShitRecord:
    record = ShitRecord(
        shit_id="img1",
        source_group="g-source",
        kind=Kind.IMAGE,
        text_blob="",
        keywords=["初音未来", "白丝"],
        image_tags=["初音未来", "白丝"],
        image_vector=build_vector(["初音未来", "白丝"], "", 0.8),
        payload=[{"kind": "image", "url": url}],
    )
    record.fingerprint = fingerprint_of(record)
    return record


class TestDirection(unittest.TestCase):
    def test_parse_and_capabilities(self):
        self.assertIs(Direction.parse("both"), Direction.BOTH)
        self.assertIs(Direction.parse("all"), Direction.BOTH)
        self.assertIs(Direction.parse("recv"), Direction.RECEIVE)
        self.assertTrue(Direction.RECEIVE.can_receive)
        self.assertFalse(Direction.RECEIVE.can_emit)
        self.assertTrue(Direction.EMIT.can_emit)
        self.assertFalse(Direction.EMIT.can_receive)
        self.assertTrue(Direction.BOTH.can_emit and Direction.BOTH.can_receive)
        with self.assertRaises(ValueError):
            Direction.parse("bogus")

    def test_old_and_chinese_names_are_gone(self):
        """语言统一：in/out 与中文别名都不再被 parse 接受。"""
        for token in ("in", "out", "收", "发", "进出"):
            with self.assertRaises(ValueError, msg=f"{token} should be rejected"):
                Direction.parse(token)

    def test_values_align_with_capability_names(self):
        """字段值与 can_receive/can_emit 对齐，避免 in/out 那种主语歧义。"""
        self.assertEqual(Direction.RECEIVE.value, "receive")
        self.assertEqual(Direction.EMIT.value, "emit")
        self.assertEqual(Direction.BOTH.value, "both")

    def test_describe_names_the_group_as_subject(self):
        for direction in Direction:
            self.assertIn("this group", direction.describe())


class TestVector(unittest.TestCase):
    def test_similarity_prefers_shared_topic(self):
        a = build_vector(["抽象", "玩梗"], "抽象玩梗高手", 0.8)
        b = build_vector(["抽象", "玩梗"], "抽象玩梗低手", 0.7)
        c = build_vector(["考研", "数学"], "考研数学加油", 0.1)
        self.assertGreater(cosine(a, b), cosine(a, c))
        self.assertAlmostEqual(cosine(a, a), 1.0, places=6)

    def test_tolerance_band(self):
        # 相近：共享主题词，余弦接近但略低于阈值 → 给及格分
        shit = build_vector(["抽象", "离谱", "逆天"], "抽象离谱的逆天发言内容", 0.5)
        close = build_vector(["抽象", "离谱", "复读"], "抽象离谱的复读内容", 0.5)
        far = build_vector(["考研", "数学"], "考研数学加油", 0.5)

        similar = cosine(shit, close)
        self.assertTrue(0.10 < similar < 0.75, f"夹具偏离预期: {similar:.3f}")

        near = keyword_score(shit, close, threshold=0.75, tolerance=0.10)
        self.assertGreater(near, 0.0)
        # 相似度一旦掉到容差带以下，就不该再给分
        strict = keyword_score(shit, close, threshold=similar + 0.20, tolerance=0.05)
        self.assertEqual(strict, 0.0)
        self.assertEqual(keyword_score(shit, far, threshold=0.75), 0.0)

    def test_keywords_stay_whole_terms(self):
        """LLM 给的是词组，不能再被 2 字滑窗切成碎片。"""
        vector = build_vector(["二次元", "机器人", "复读"], "", 1.0)
        self.assertIn("二次元", vector)
        self.assertIn("机器人", vector)
        self.assertIn("复读", vector)
        for fragment in ("二次", "次元", "机器"):
            self.assertNotIn(fragment, vector)

    def test_tags_are_whole_terms_and_split_on_separators(self):
        vector = build_vector([], "", 1.0, ["代码", "机器 复读/抽象"])
        for term in ("代码", "机器", "复读", "抽象"):
            self.assertIn(term, vector)
        self.assertNotIn("器复", vector)

    def test_body_does_not_re_add_keyword_fragments(self):
        vector = build_vector(["二次元"], "这是二次元机器人群聊", 0.5)
        self.assertIn("二次元", vector)
        self.assertNotIn("二次", vector)
        self.assertNotIn("次元", vector)

    def test_tokenize_drops_single_chars_and_timestamp_fragments(self):
        tokens = tokenize("B站 3小时前")
        for noise in ("站", "小时", "时前"):
            self.assertNotIn(noise, tokens)

    def test_similar_terms_merge_into_one_representative(self):
        merged = merge_similar_terms({"机器": 1.0, "机器人": 2.0, "抽象": 1.0})
        self.assertEqual(set(merged), {"机器人", "抽象"})
        self.assertAlmostEqual(merged["机器人"], 3.0)

    def test_similar_terms_merge_in_chains(self):
        merged = merge_similar_terms(
            {"二次元": 0.2, "二次元文化": 0.1, "二次元文化圈": 0.3}
        )
        self.assertEqual(set(merged), {"二次元文化圈"})
        self.assertAlmostEqual(merged["二次元文化圈"], 0.6)

    def test_single_char_terms_are_dropped(self):
        # “站”只是“B站”被旧滑窗切剩下的碎片，不该在口味表里占位
        merged = merge_similar_terms({"站": 1.0, "b站": 2.0})
        self.assertEqual(set(merged), {"b站"})
        self.assertEqual(merge_similar_terms({"草": 1.0}), {})

    def test_blend_into_merges_legacy_fragments(self):
        """旧版本存下的“二次/次元”遇到“二次元”时会自动并进去。"""
        merged = blend_into({"二次": 0.5, "次元": 0.5}, {"二次元": 1.0})
        self.assertIn("二次元", merged)
        self.assertNotIn("二次", merged)
        self.assertNotIn("次元", merged)

    def test_describe_intake_merges_similar_terms(self):
        group = GroupProfile(
            group_key="g", intake={"机器": 0.5, "机器人": 0.5, "抽象": 0.3}
        )
        text = describe_intake(group)
        self.assertIn("机器人", text)
        self.assertIn("抽象", text)
        self.assertNotIn("机器(", text)

    def test_stopwords_drop_meta_and_timestamp_terms(self):
        vector = build_vector(["玩梗", "小时", "代码"], "玩梗 3小时前 代码", 0.5)
        self.assertIn("代码", vector)
        for noise in ("玩梗", "小时", "时前", "梗内"):
            self.assertNotIn(noise, vector)

    def test_alias_merges_abbreviation_with_full_name(self):
        vector = build_vector(["AI", "人工智能"], "", 0.5)
        self.assertEqual(set(vector), {"人工智能"})
        merged = merge_similar_terms({"ai": 1.0, "人工智能": 2.0})
        self.assertEqual(set(merged), {"人工智能"})
        self.assertAlmostEqual(merged["人工智能"], 3.0)

    def test_custom_rules_extend_builtin_tables(self):
        rules = TermRules.build("缝合", "赛博朋克=cyberpunk")
        vector = build_vector(["缝合", "cyberpunk"], "", 0.5, rules=rules)
        self.assertEqual(set(vector), {"赛博朋克"})
        self.assertTrue(rules.is_stopped("缝合"))

    def test_settings_parse_tag_rules(self):
        settings = make_settings(tag_aliases="人工智能=AI,ai", tag_stopwords="缝合")
        self.assertEqual(settings.term_rules.canonical("AI"), "人工智能")
        self.assertTrue(settings.term_rules.is_stopped("缝合"))

    def test_alias_config_accepts_json_object(self):
        settings = make_settings(tag_aliases='{"人工智能": ["AI", "人工智慧"]}')
        self.assertEqual(settings.term_rules.canonical("人工智慧"), "人工智能")

    def test_describe_intake_hides_stopwords(self):
        group = GroupProfile(group_key="g", intake={"玩梗": 0.4, "代码": 0.3})
        text = describe_intake(group)
        self.assertIn("代码", text)
        self.assertNotIn("玩梗", text)

    def test_compact_intake_removes_legacy_fragments(self):
        compacted = compact_intake(
            {"站": 0.2, "机器": 0.2, "机器人": 0.2, "代码": 0.4}
        )
        self.assertEqual(set(compacted), {"机器人", "代码"})
        self.assertAlmostEqual(sum(v * v for v in compacted.values()), 1.0)

    def test_apply_term_rewrite_merges_and_drops(self):
        rewritten = apply_term_rewrite(
            {"ai": 0.3, "人工智能": 0.2, "站": 0.2, "代码": 0.3},
            {
                "drop": ["站"],
                "groups": [
                    {"canonical": "人工智能", "members": ["ai", "人工智能"]}
                ],
            },
        )
        self.assertEqual(set(rewritten), {"人工智能", "代码"})
        self.assertAlmostEqual(rewritten["人工智能"] / rewritten["代码"], 5 / 3)
        self.assertAlmostEqual(sum(v * v for v in rewritten.values()), 1.0)

    def test_apply_term_rewrite_can_create_canonical(self):
        # “二次”+“次元”合到一个原本不在表里的“二次元”
        rewritten = apply_term_rewrite(
            {"二次": 0.2, "次元": 0.2, "代码": 0.6},
            {
                "groups": [
                    {"canonical": "二次元", "members": ["二次", "次元"]}
                ]
            },
        )
        self.assertEqual(set(rewritten), {"二次元", "代码"})
        self.assertAlmostEqual(rewritten["代码"] / rewritten["二次元"], 1.5)

    def test_apply_term_rewrite_ignores_unknown_terms(self):
        rewritten = apply_term_rewrite(
            {"代码": 1.0},
            {
                "drop": ["不存在的词"],
                "groups": [
                    {"canonical": "量子计算", "members": ["不存在的词"]}
                ],
            },
        )
        self.assertEqual(set(rewritten), {"代码"})


class TestConfidence(unittest.TestCase):
    def test_grows_with_samples_and_falls_with_rejections(self):
        low = confidence_score(
            native_messages=1, native_chars=40, drift=0, accepted=1, rejected=0
        )
        high = confidence_score(
            native_messages=80, native_chars=3200, drift=0, accepted=160, rejected=0
        )
        rejected = confidence_score(
            native_messages=80, native_chars=3200, drift=0, accepted=1, rejected=200
        )
        self.assertLess(low, 0.2)
        self.assertGreater(high, 0.7)
        self.assertLess(rejected, high)

    def test_update_intake_accumulates(self):
        settings = make_settings()
        group = make_group("g1", native=0)
        before = dict(group.intake)
        shit = make_shit("抽象离谱，逆天发言", ["抽象", "逆天"])
        drift = update_intake(group, shit, settings, is_native=True)
        self.assertGreaterEqual(drift, 0.0)
        self.assertLessEqual(drift, 1.0)
        self.assertEqual(group.native_message_count, 1)
        self.assertTrue(set(group.intake) >= set(before) | {"逆天"})


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings()

    def test_forward_image_bili(self):
        forward = Message(
            group_key="g",
            segments=[
                Segment(
                    kind="forward", children=[Segment(kind="plain", text="哈哈哈")]
                )
            ],
        )
        self.assertTrue(is_there_a_shit(forward, self.settings)[0])

        image = Message(
            group_key="g", segments=[Segment(kind="image", url="http://x/y.png")]
        )
        self.assertTrue(is_there_a_shit(image, self.settings)[0])

        bili = Message(
            group_key="g", segments=[Segment(kind="plain", text="看这个 BV1xx411c7mD")]
        )
        self.assertTrue(is_there_a_shit(bili, self.settings)[0])

    def test_short_chatter_ignored(self):
        short = Message(group_key="g", segments=[Segment(kind="plain", text="哈哈")])
        self.assertEqual(is_there_a_shit(short, self.settings), (False, ""))

    def test_bili_extraction(self):
        refs = extract_bili_ids("看 https://www.bilibili.com/video/BV1xx411c7mD 和 BV1xx411c7mD")
        self.assertEqual(refs.bvids, ["BV1xx411c7mD"])

    # ---- 松闸门 ----
    def test_loose_marker_short_text_is_suspicious(self):
        message = Message(group_key="g", segments=[Segment(kind="plain", text="这也太逆天了吧")])
        interesting, reason = is_there_a_shit(message, self.settings)
        self.assertTrue(interesting)
        self.assertIn("玩梗短句", reason)
        self.assertIn("逆天", reason)

    def test_medium_text_is_suspicious(self):
        text = "今天群里发生了一件很怪的事情大家快来看看"  # 20 字，低于软阈值
        self.assertEqual(len(text), 20)
        message = Message(group_key="g", segments=[Segment(kind="plain", text=text)])
        self.assertFalse(is_there_a_shit(message, self.settings)[0])

        longer = text + "真的吗？"  # 24 字，达到软阈值
        message = Message(group_key="g", segments=[Segment(kind="plain", text=longer)])
        interesting, reason = is_there_a_shit(message, self.settings)
        self.assertTrue(interesting)
        self.assertIn("中等文本", reason)

    def test_media_and_link_are_suspicious(self):
        voice = Message(group_key="g", segments=[Segment(kind="record", url="http://x/a.mp3")])
        self.assertIn("媒体", is_there_a_shit(voice, self.settings)[1])

        link = Message(
            group_key="g",
            segments=[Segment(kind="plain", text="看 https://example.com/a 这个")],
        )
        self.assertIn("链接分享", is_there_a_shit(link, self.settings)[1])

    def test_loose_gate_can_be_disabled(self):
        settings = make_settings(loose_detect=False)
        marker = Message(group_key="g", segments=[Segment(kind="plain", text="太逆天了")])
        self.assertFalse(is_there_a_shit(marker, settings)[0])

        medium = Message(
            group_key="g",
            segments=[Segment(kind="plain", text="今天群里发生了一件很怪的事情大家快来看看真的吗？")],
        )
        self.assertFalse(is_there_a_shit(medium, settings)[0])

    def test_forward_min_nodes_is_honored(self):
        settings = make_settings(forward_min_nodes=3)
        thin = Message(
            group_key="g",
            segments=[
                Segment(
                    kind="forward", children=[Segment(kind="plain", text="只有一条")]
                )
            ],
        )
        self.assertFalse(is_there_a_shit(thin, settings)[0])

        thick = Message(
            group_key="g",
            segments=[
                Segment(
                    kind="forward",
                    children=[Segment(kind="plain", text=f"第{i}条") for i in range(3)],
                )
            ],
        )
        self.assertTrue(is_there_a_shit(thick, settings)[0])


class TestComponentType(unittest.TestCase):
    """AstrBot 的 ComponentType 是 Enum，必须取 .value，否则整条识别链断掉。"""

    def test_enum_value_is_unwrapped(self):
        class FakeEnum:
            value = "Plain"

            def __str__(self):  # 模拟非 StrEnum 的 str(ComponentType.Plain)
                return "ComponentType.Plain"

        class Comp:
            type = FakeEnum()

        self.assertEqual(_component_type(Comp()), "plain")

    def test_string_type_still_works(self):
        class Comp:
            type = "Image"

        self.assertEqual(_component_type(Comp()), "image")

    def test_segment_kind_survives_enum(self):
        class FakeEnum:
            value = "Image"

            def __str__(self):
                return "ComponentType.Image"

        class Comp:
            type = FakeEnum()
            url = "http://x/y.png"

        segment = segment_from_component(Comp())
        self.assertEqual(segment.kind, "image")
        self.assertEqual(segment.url, "http://x/y.png")


class TestKeywordMerge(unittest.TestCase):
    def test_similar_keywords_collapse_to_shortest(self):
        merged = merge_similar_keywords(["抽象话", "抽象", "抽象人"])
        self.assertEqual(merged, ["抽象"])

    def test_different_keywords_are_kept(self):
        merged = merge_similar_keywords(["考研", "数学", "玩梗"])
        self.assertEqual(merged, ["考研", "数学", "玩梗"])

    def test_similarity_threshold(self):
        self.assertGreater(keyword_similarity("抽象", "抽象话"), 0.6)
        self.assertEqual(keyword_similarity("抽象", "考研"), 0.0)

    def test_build_vector_does_not_double_count_variants(self):
        single = build_vector(["抽象"], "抽象发言", 0.8)
        variants = build_vector(["抽象", "抽象话", "抽象人"], "抽象发言", 0.8)
        self.assertEqual(single, variants)


class TestProfileMerge(unittest.TestCase):
    """口味表是长期累积的，老词条也要收敛，不能只合并新来的那一批。"""

    def test_merge_profile_terms_collapses_whole_vocabulary(self):
        merged = merge_profile_terms(
            {"抽象话": 3.0, "抽象人": 2.0, "抽象": 1.0, "离谱": 4.0}
        )
        self.assertIn("抽象", merged)
        self.assertIn("离谱", merged)
        self.assertEqual(len(merged), 2)
        # 相似词条的权重相加，再整体归一化
        self.assertGreater(merged["抽象"], merged["离谱"])
        self.assertAlmostEqual(sum(value * value for value in merged.values()), 1.0)

    def test_update_intake_merges_previously_registered_terms(self):
        settings = make_settings()
        group = GroupProfile(group_key="g", intake={"抽象话": 1.0, "抽象人": 1.0})
        shit = make_shit("抽象发言", ["抽象"])
        update_intake(group, shit, settings, is_native=True)
        self.assertIn("抽象", group.intake)
        self.assertNotIn("抽象话", group.intake)
        self.assertNotIn("抽象人", group.intake)

    def test_merge_is_transitive_and_order_independent(self):
        # “抽象话”和“抽象人”本身都不够像，但它们都能被“抽象”桥接；
        # 合并不应该依赖字典遍历顺序。
        forward = merge_profile_terms({"抽象话": 3.0, "抽象人": 2.0, "抽象": 1.0})
        backward = merge_profile_terms({"抽象": 1.0, "抽象人": 2.0, "抽象话": 3.0})
        self.assertEqual(forward, backward)
        self.assertEqual(list(forward), ["抽象"])


class TestAnalysisParsing(unittest.TestCase):
    def test_image_fields_are_parsed(self):
        analysis = parse_analysis(
            {
                "is_shit": True,
                "intensity": 0.8,
                "image_tags": ["初音未来", "白丝"],
                "characters": ["初音未来"],
                "copyrights": ["VOCALOID"],
                "nsfw": True,
                "minor_sexual": False,
            }
        )
        self.assertEqual(analysis.image_tags, ["初音未来", "白丝"])
        self.assertEqual(analysis.characters, ["初音未来"])
        self.assertEqual(analysis.copyrights, ["VOCALOID"])
        self.assertTrue(analysis.nsfw)
        self.assertFalse(analysis.minor_sexual)

    def test_missing_image_fields_default_to_empty(self):
        analysis = parse_analysis({"is_shit": True, "intensity": 0.7})
        self.assertEqual(analysis.image_tags, [])
        self.assertEqual(analysis.characters, [])
        self.assertEqual(analysis.copyrights, [])
        self.assertFalse(analysis.nsfw)
        self.assertFalse(analysis.minor_sexual)

    def test_minor_sexual_flag_is_parsed(self):
        analysis = parse_analysis(
            {"is_shit": False, "intensity": 0, "minor_sexual": True}
        )
        self.assertTrue(analysis.minor_sexual)

    def test_string_false_is_not_treated_as_true(self):
        # bool("false") == True 会误触发安全拦截，也会把普通消息当成屎。
        analysis = parse_analysis(
            {
                "is_shit": "false",
                "context_relevant": "false",
                "nsfw": "false",
                "minor_sexual": "false",
            }
        )
        self.assertFalse(analysis.is_shit)
        self.assertFalse(analysis.context_relevant)
        self.assertFalse(analysis.nsfw)
        self.assertFalse(analysis.minor_sexual)

    def test_string_true_is_treated_as_true(self):
        analysis = parse_analysis(
            {
                "is_shit": "true",
                "context_relevant": "yes",
                "nsfw": "1",
                "minor_sexual": "true",
            }
        )
        self.assertTrue(analysis.is_shit)
        self.assertTrue(analysis.context_relevant)
        self.assertTrue(analysis.nsfw)
        self.assertTrue(analysis.minor_sexual)


class TestConfidenceConsistency(unittest.TestCase):
    def test_group_confidence_matches_profile_formula(self):
        group = make_group("g", native=12)
        group.native_char_count = 900
        group.drift = 0.4
        group.accepted_count = 7
        group.rejected_count = 3
        self.assertAlmostEqual(
            group.confidence(10.0),
            confidence_score(
                native_messages=12,
                native_chars=900,
                drift=0.4,
                accepted=7,
                rejected=3,
                k=10.0,
            ),
            places=9,
        )

    def test_drift_lowers_confidence(self):
        stable = make_group("stable", native=80)
        unstable = make_group("unstable", native=80)
        unstable.drift = 0.9
        self.assertLess(unstable.confidence(10.0), stable.confidence(10.0))

    def test_update_intake_records_drift_after_topic_switch(self):
        settings = make_settings()
        group = GroupProfile(group_key="g")  # 空档案：第一条样本没有参照
        first = make_shit("抽象离谱的逆天发言", ["抽象", "逆天"])
        update_intake(group, first, settings, is_native=True)
        self.assertEqual(group.drift, 0.0)

        second = make_shit("考研数学复习进度", ["考研", "数学"])
        drift = update_intake(group, second, settings, is_native=True)
        self.assertGreater(drift, 0.0)
        self.assertGreater(group.drift, 0.0)


class TestRouter(unittest.TestCase):
    def setUp(self):
        # 全局阈值抬到 0.6，让“可信度”这道门真正起作用
        self.settings = make_settings(threshold_overall=0.6)
        self.router = Router(self.settings)
        self.shit = make_shit("抽象离谱到爆的内容", ["抽象", "离谱"])

    def test_direction_and_threshold(self):
        groups = [
            make_group("out-only", Direction.EMIT),
            make_group("in-only", Direction.RECEIVE, native=50),
            make_group("too-fresh", Direction.RECEIVE, native=0),
        ]
        plan = self.router.keyword_plan(self.shit, groups, exclude={"g-source"})
        keys = plan.group_keys()
        self.assertNotIn("out-only", keys)  # 只发的群不收货
        self.assertIn("in-only", keys)  # 口味对、样本足
        self.assertNotIn("too-fresh", keys)  # 口味对但没吃过屎，可信度不足

        reasons = dict(plan.rejected)
        self.assertIn("可信度不足", reasons["too-fresh"])

    def test_low_confidence_does_not_block_when_threshold_is_zero(self):
        settings = make_settings(threshold_overall=0.0)
        router = Router(settings)
        plan = router.keyword_plan(
            self.shit, [make_group("fresh", Direction.RECEIVE, native=0)]
        )
        # 新鲜群也能收（默认阈值 0 只防“口味不像”，不防“样本少”）
        self.assertIn("fresh", plan.group_keys())

    def test_no_repeat_delivery(self):
        groups = [make_group("in-only", Direction.RECEIVE)]
        plan = self.router.keyword_plan(self.shit, groups, delivered={"in-only"})
        self.assertEqual(plan.candidates, [])
        self.assertTrue(plan.rejected)


class TestImageGate(unittest.TestCase):
    """图片必须靠视觉标签过门槛，文字话题对上了也不能代替。"""

    def setUp(self):
        self.settings = make_settings(threshold_overall=0.0)
        self.router = Router(self.settings)

    def _group(self, key: str, tags: list[str]) -> GroupProfile:
        return GroupProfile(
            group_key=key,
            direction=Direction.RECEIVE,
            intake=build_vector(tags, "", 0.8),
            native_message_count=20,
            native_char_count=1000,
        )

    def test_image_is_routed_by_visual_tags(self):
        shit = make_image_shit()
        groups = [
            self._group("likes", ["初音未来", "白丝"]),
            self._group("other", ["考研", "数学"]),
        ]
        plan = self.router.keyword_plan(shit, groups, exclude={"g-source"})
        self.assertIn("likes", plan.group_keys())
        self.assertNotIn("other", plan.group_keys())
        reasons = dict(plan.rejected)
        self.assertIn("图片标签", reasons["other"])
        self.assertGreater(plan.candidates[0].image_score, 0.0)

    def test_text_match_cannot_bypass_image_gate(self):
        shit = make_image_shit()
        # 文字向量和该群高度一致，但视觉标签完全对不上
        shit.vector = build_vector(["抽象", "离谱"], "抽象离谱", 0.8)
        shit.image_vector = build_vector(["初音未来", "白丝"], "", 0.8)
        group = self._group("text-only", ["抽象", "离谱"])
        plan = self.router.keyword_plan(shit, [group], exclude={"g-source"})
        self.assertNotIn("text-only", plan.group_keys())
        self.assertIn("图片标签", dict(plan.rejected)["text-only"])

    def test_image_without_tags_is_rejected_when_tagging_on(self):
        shit = make_image_shit()
        shit.image_tags = []
        shit.image_vector = {}
        group = self._group("likes", ["初音未来", "白丝"])
        plan = self.router.keyword_plan(shit, [group], exclude={"g-source"})
        self.assertNotIn("likes", plan.group_keys())
        self.assertIn("缺少视觉标签", dict(plan.rejected)["likes"])

    def test_image_falls_back_to_text_when_tagging_is_disabled(self):
        settings = make_settings(
            threshold_overall=0.0, image_tagging=False
        )
        router = Router(settings)
        shit = make_image_shit()
        shit.image_tags = []
        shit.image_vector = {}
        shit.vector = build_vector(["初音未来", "白丝"], "", 0.8)
        group = self._group("likes", ["初音未来", "白丝"])
        plan = router.keyword_plan(shit, [group], exclude={"g-source"})
        self.assertIn("likes", plan.group_keys())

    def test_group_can_disable_image_relay(self):
        shit = make_image_shit()
        group = self._group("no-image", ["初音未来", "白丝"])
        group.image_enabled = False
        plan = self.router.keyword_plan(shit, [group], exclude={"g-source"})
        self.assertEqual(plan.candidates, [])
        self.assertIn("未启用图片", dict(plan.rejected)["no-image"])

    def test_text_relay_is_not_affected_by_image_switch(self):
        group = self._group("no-image", ["抽象", "离谱"])
        group.image_enabled = False
        shit = make_shit("抽象离谱的玩梗内容", ["抽象", "离谱"])
        plan = self.router.keyword_plan(shit, [group], exclude={"g-source"})
        self.assertIn("no-image", plan.group_keys())

    def test_different_images_have_different_fingerprints(self):
        first = make_image_shit("http://x/a.png")
        second = make_image_shit("http://x/b.png")
        same = make_image_shit("http://x/a.png")
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.fingerprint, same.fingerprint)


_WORKDIR = ROOT / "tests" / "_scratch"
_seq = itertools.count()


def scratch_path(name: str) -> Path:
    """在仓库内开一个隔离的数据库路径。

    不用 tempfile/TemporaryDirectory：Windows 沙箱会接管它创建的目录，
    进程随即失去 WRITE_DAC，连读都读不了，更没法清理。直接 makedirs 就行。
    """
    folder = _WORKDIR / f"{name}-{next(_seq)}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "t.sqlite3"


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.db_path = scratch_path("storage")
        self.storage = Storage(self.db_path)

    def tearDown(self):
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.db_path) + suffix).unlink(missing_ok=True)

    def test_roundtrip_and_delivery_dedup(self):
        group = make_group("g1")
        self.storage.upsert_group(group)
        self.assertIs(self.storage.get_group("g1").direction, Direction.BOTH)

        shit = make_shit("抽象离谱", ["抽象", "离谱"])
        self.storage.add_shit(shit)
        self.assertEqual(
            self.storage.find_by_fingerprint(shit.fingerprint).shit_id, shit.shit_id
        )

        self.assertTrue(
            self.storage.record_delivery(shit.shit_id, "g1", 0.9, 0.8, "keyword")
        )
        self.assertFalse(
            self.storage.record_delivery(shit.shit_id, "g1", 0.9, 0.8, "keyword")
        )
        self.assertTrue(self.storage.delete_group("g1"))
        self.assertIsNone(self.storage.get_group("g1"))

    def test_drift_is_persisted(self):
        group = make_group("g-drift")
        group.drift = 0.42
        self.storage.upsert_group(group)
        self.assertAlmostEqual(self.storage.get_group("g-drift").drift, 0.42)

    def test_image_switch_is_persisted(self):
        group = make_group("g-image-off")
        group.image_enabled = False
        self.storage.upsert_group(group)
        self.assertFalse(self.storage.get_group("g-image-off").image_enabled)

    def test_image_fields_are_persisted(self):
        image = make_image_shit()
        image.nsfw = True
        self.storage.add_shit(image)
        loaded = self.storage.get_shit(image.shit_id)
        self.assertEqual(loaded.image_tags, ["初音未来", "白丝"])
        self.assertTrue(loaded.nsfw)
        self.assertGreater(max(loaded.image_vector.values()), 0.0)


class TestJsonParsing(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(
            extract_json('好的，结果如下：{"is_shit": true, "intensity": 0.9} 谢谢'),
            {"is_shit": True, "intensity": 0.9},
        )
        self.assertIsNone(extract_json("没有 JSON"))


class TestAccumulator(unittest.TestCase):
    def test_window_close(self):
        settings = make_settings(window_seconds=10)
        acc = ContextAccumulator(settings)
        msg = Message(group_key="g", segments=[Segment(kind="plain", text="抽象大屎")])
        acc.observe(msg)
        acc.open(msg, "长文本")
        self.assertEqual(acc.due(now=time.time()), [])

        ready = acc.due(now=time.time() + 60)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0][0], "g")

    def test_multiple_open_windows_in_one_group_are_not_dropped(self):
        settings = make_settings(window_seconds=10)
        acc = ContextAccumulator(settings)
        first = Message(group_key="g", segments=[Segment(kind="plain", text="第一条")])
        second = Message(group_key="g", segments=[Segment(kind="plain", text="第二条")])
        acc.observe(first)
        one = acc.open(first, "长文本")
        acc.observe(second)
        two = acc.open(second, "图片")

        self.assertEqual(acc.pending(), 2)
        self.assertIn("第二条", acc.render_extra(one))
        ready = acc.due(now=time.time() + 60)
        self.assertEqual({id(item) for _, item in ready}, {id(one), id(two)})
        self.assertEqual(acc.pending(), 0)


if __name__ == "__main__":
    unittest.main()
