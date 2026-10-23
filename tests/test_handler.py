"""Handler-level tests for main.py.

main.py imports astrbot, which is not installed on the dev box, so we stub the
handshake modules the plugin touches. The point of these tests is the failure
mode we actually hit in production: an exception escaping the command handler
makes AstrBot post a raw traceback into the group.
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_astrbot_stubs() -> None:
    """Register the minimal astrbot surface that main.py imports."""
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = dict
    api.logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                       warning=lambda *a, **k: None,
                                       error=lambda *a, **k: None)

    event = types.ModuleType("astrbot.api.event")

    class _FilterNamespace:
        EventMessageType = types.SimpleNamespace(ALL="all")

        @staticmethod
        def command(*args, **kwargs):
            def decorator(func):
                func._is_command = True
                return func

            return decorator

        @staticmethod
        def event_message_type(*args, **kwargs):
            def decorator(func):
                func._is_listener = True
                return func

            return decorator

    class AstrMessageEvent:  # noqa: D401 - stub
        pass

    event.AstrMessageEvent = AstrMessageEvent
    event.filter = _FilterNamespace()

    star = types.ModuleType("astrbot.api.star")

    class Context:
        pass

    class Star:
        def __init__(self, context):
            self.context = context

    def register(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    star.Context = Context
    star.Star = Star
    star.register = register

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.star": star,
        }
    )


_install_astrbot_stubs()

# main.py uses relative imports (it is normally loaded as a package by AstrBot),
# so register the repo root as a package namespace before importing it.
import importlib.util  # noqa: E402


def _load_plugin_module():
    package_name = "shitposts_sharer_under_test"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "main.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


plugin_main = _load_plugin_module()
from tests.test_command_helpers import FakeEvent  # noqa: E402


class TestHandlerSignature(unittest.TestCase):
    """Enforce AstrBot's parameter-parsing contract.

    AstrBot records every handler parameter after (self, event) and calls it
    as a converter. A trailing *args becomes inspect.Parameter.empty, which
    raises TypeError: _empty() takes no arguments - the command is swallowed
    by waking_check and never runs. Converting this class goes through exactly
    the code path that failed in production, so keep it.
    """

    # The exact coercion AstrBot's validate_and_convert_params performs:
    #     result[param_name] = param_type_or_default_val(params[i])
    @staticmethod
    def _astrbot_convert(params: dict, argv: list[str]) -> dict:
        result = {}
        for index, (name, type_or_default) in enumerate(params.items()):
            result[name] = type_or_default(argv[index])
        return result

    def _extra_params(self, func) -> dict:
        import inspect as _inspect

        params = {}
        for index, (name, param) in enumerate(
            _inspect.signature(func).parameters.items()
        ):
            if index < 2:  # self, event
                continue
            if param.default is _inspect.Parameter.empty:
                params[name] = param.annotation
            else:
                params[name] = param.default
        return params

    def test_command_handlers_declare_no_extra_params(self):
        plugin = plugin_main.ShitpostsSharer.__new__(plugin_main.ShitpostsSharer)
        for name in ("shit", "manual_shot"):
            handler = getattr(plugin_main.ShitpostsSharer, name)
            self.assertEqual(
                self._extra_params(handler),
                {},
                f"{name} declares parameters AstrBot would try to coerce",
            )

    def test_reproduces_astrbot_failure_with_star_args(self):
        import inspect as _inspect

        async def legacy(self, event, *args):
            yield None

        params = {}
        for index, (name, param) in enumerate(_inspect.signature(legacy).parameters.items()):
            if index < 2:
                continue
            params[name] = param.annotation if param.default is _inspect.Parameter.empty else param.default

        self.assertEqual(params, {"args": _inspect.Parameter.empty})
        with self.assertRaises(TypeError) as ctx:
            self._astrbot_convert(params, ["help"])
        self.assertIn("_empty() takes no arguments", str(ctx.exception))


class TestCommandHandler(unittest.TestCase):
    def setUp(self):
        self.plugin = plugin_main.ShitpostsSharer.__new__(
            plugin_main.ShitpostsSharer
        )
        self.plugin.settings = __import__(
            "core.config", fromlist=["Settings"]
        ).Settings({})

    def _run(self, event) -> list:
        async def _collect():
            return [item async for item in self.plugin.shit(event)]

        return asyncio.run(_collect())

    def test_command_args_strips_command_token(self):
        event = FakeEvent()
        event.message_str = "/shit register both"
        self.assertEqual(
            plugin_main._command_args(event, {"shit"}), ["register", "both"]
        )

    def test_command_args_strips_alias_token(self):
        event = FakeEvent()
        event.message_str = "/ss status"
        self.assertEqual(plugin_main._command_args(event, {"shit", "ss"}), ["status"])

    def test_command_args_strips_emoji_safe_alias(self):
        event = FakeEvent()
        event.message_str = "/sps status"
        self.assertEqual(
            plugin_main._command_args(event, {"shit", "sps"}), ["status"]
        )
        self.assertIn("sps", plugin_main.ALIASES)

    def test_exception_becomes_plain_text_not_traceback(self):
        class Boom:
            def dispatch(self, *args, **kwargs):
                raise RuntimeError("kaboom")

        self.plugin.commands = Boom()
        event = FakeEvent()
        event.message_str = "/shit help"

        outputs = self._run(event)
        self.assertEqual(len(outputs), 1)
        self.assertIn("plain", outputs[0])
        self.assertIn("Command failed", outputs[0]["plain"])
        self.assertNotIn("kaboom", outputs[0]["plain"])

    def test_empty_result_text_is_never_blank(self):
        class Empty:
            def dispatch(self, *args, **kwargs):
                return types.SimpleNamespace(text="")

        self.plugin.commands = Empty()
        event = FakeEvent()
        event.message_str = "/shit help"
        outputs = self._run(event)
        self.assertEqual(outputs[0]["plain"], "(empty)")


if __name__ == "__main__":
    unittest.main()
