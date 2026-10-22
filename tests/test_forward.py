"""Forward expansion tests.

The bug these guard against: AstrBot's Forward component carries only an
``id`` (no node_list, no nodes), and NapCat ships with parseMultMsg=false.
So a merged-forward arrived as a hollow shell, the model was handed
"（无文本内容，只有媒体）", and every forward judged is_shit=false.

Instead of asking the user to flip a NapCat-wide setting (which affects all
plugins and may require a re-login), the plugin fetches the body itself via
``get_forward_msg``.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.detector import expand_forward, message_from_event  # noqa: E402
from core.forward import (  # noqa: E402
    fetch_forward_nodes,
    get_onebot_client,
    node_content,
    node_sender_name,
    onebot_segment_to_text,
)
from core.models import Message, Segment  # noqa: E402


class FakeClient:
    def __init__(self, payload=None, raises: Exception | None = None):
        self.payload = payload
        self.raises = raises
        self.calls: list[tuple[str, dict]] = []

    async def call_action(self, action: str, **kwargs):
        self.calls.append((action, kwargs))
        if self.raises:
            raise self.raises
        return self.payload


class FakePlatform:
    def __init__(self, client):
        self._client = client

    def get_client(self):
        return self._client


class FakePlatformManager:
    def __init__(self, platforms):
        self.platform_insts = platforms


class FakeContext:
    def __init__(self, client):
        self.platform_manager = FakePlatformManager([FakePlatform(client)])


# 真实 napcat get_forward_msg 返回形状
NAPCAT_PAYLOAD = {
    "status": "ok",
    "retcode": 0,
    "data": {
        "messages": [
            {
                "type": "node",
                "data": {
                    "sender": {"nickname": "张三", "user_id": "1001"},
                    "message": [
                        {"type": "text", "data": {"text": "你看这个"}},
                        {"type": "image", "data": {"file": "abc.jpg"}},
                    ],
                },
            },
            {
                "type": "node",
                "data": {
                    "sender": {"card": "李四"},
                    "message": [{"type": "text", "data": {"text": "太逆天了"}}],
                },
            },
        ]
    },
}


class TestNodeParsing(unittest.TestCase):
    def test_sender_name_prefers_card(self):
        node = NAPCAT_PAYLOAD["data"]["messages"][1]
        self.assertEqual(node_sender_name(node), "李四")

    def test_sender_name_falls_back_to_nickname(self):
        node = NAPCAT_PAYLOAD["data"]["messages"][0]
        self.assertEqual(node_sender_name(node), "张三")

    def test_node_content_from_message_key(self):
        node = NAPCAT_PAYLOAD["data"]["messages"][0]
        self.assertEqual(len(node_content(node)), 2)

    def test_segment_rendering(self):
        self.assertEqual(
            onebot_segment_to_text({"type": "text", "data": {"text": "hi"}}), "hi"
        )
        self.assertEqual(
            onebot_segment_to_text({"type": "image", "data": {"file": "x"}}), "[图片]"
        )


class TestFetch(unittest.TestCase):
    def test_extracts_nodes_from_napcat_shape(self):
        nodes = asyncio.run(fetch_forward_nodes(FakeContext(FakeClient(NAPCAT_PAYLOAD)), "777"))
        self.assertEqual(len(nodes), 2)

    def test_calls_get_forward_msg_with_id(self):
        client = FakeClient(NAPCAT_PAYLOAD)
        asyncio.run(fetch_forward_nodes(FakeContext(client), "777"))
        self.assertEqual(client.calls[0][0], "get_forward_msg")
        self.assertEqual(client.calls[0][1]["id"], "777")

    def test_empty_id_skips_call(self):
        client = FakeClient(NAPCAT_PAYLOAD)
        self.assertEqual(asyncio.run(fetch_forward_nodes(FakeContext(client), "")), [])
        self.assertEqual(client.calls, [])

    def test_api_failure_returns_empty_not_raises(self):
        context = FakeContext(FakeClient(raises=RuntimeError("boom")))
        self.assertEqual(asyncio.run(fetch_forward_nodes(context, "777")), [])

    def test_no_client_returns_empty(self):
        class Bare:
            platform_manager = FakePlatformManager([])

        self.assertEqual(asyncio.run(fetch_forward_nodes(Bare(), "777")), [])

    def test_get_onebot_client_rejects_non_onebot(self):
        class NoClientPlatform:
            def get_client(self):
                return None

        context = type(
            "C", (), {"platform_manager": FakePlatformManager([NoClientPlatform()])}
        )()
        self.assertIsNone(get_onebot_client(context))


class TestExpandForward(unittest.TestCase):
    def test_hollow_forward_gets_filled(self):
        message = Message(
            group_key="g",
            segments=[Segment(kind="forward", url="777")],
        )
        self.assertIn("未展开", message.text())

        asyncio.run(expand_forward(message, FakeContext(FakeClient(NAPCAT_PAYLOAD))))

        text = message.text()
        self.assertIn("张三", text)
        self.assertIn("你看这个", text)
        self.assertIn("太逆天了", text)
        self.assertNotIn("未展开", text)

    def test_already_populated_forward_is_untouched(self):
        message = Message(
            group_key="g",
            segments=[
                Segment(
                    kind="forward",
                    url="777",
                    children=[Segment(kind="plain", text="已有内容")],
                )
            ],
        )
        client = FakeClient(NAPCAT_PAYLOAD)
        asyncio.run(expand_forward(message, FakeContext(client)))
        self.assertEqual(client.calls, [])  # 不该重复请求
        self.assertIn("已有内容", message.text())

    def test_failed_fetch_keeps_hollow_state(self):
        message = Message(group_key="g", segments=[Segment(kind="forward", url="777")])
        asyncio.run(
            expand_forward(message, FakeContext(FakeClient(raises=RuntimeError("x"))))
        )
        self.assertIn("未展开", message.text())

    def test_no_forward_is_noop(self):
        message = Message(group_key="g", segments=[Segment(kind="plain", text="hi")])
        client = FakeClient(NAPCAT_PAYLOAD)
        asyncio.run(expand_forward(message, FakeContext(client)))
        self.assertEqual(client.calls, [])


class TestForwardIdCapture(unittest.TestCase):
    """Forward 组件只有 id —— 必须存进 Segment.url 才能回源。"""

    def test_forward_component_id_is_kept(self):
        class ForwardComp:
            type = "forward"
            id = "abc123"

        class Obj:
            message = [ForwardComp()]
            sender_name = "sender"
            message_id = "m1"
            time = 0

        class Event:
            message_obj = Obj()

            def get_sender_id(self):
                return "1"

        message = message_from_event(Event(), "g")
        self.assertEqual(len(message.segments), 1)
        self.assertEqual(message.segments[0].kind, "forward")
        self.assertEqual(message.segments[0].url, "abc123")


if __name__ == "__main__":
    unittest.main()
