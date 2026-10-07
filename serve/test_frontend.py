"""Tool-call quotation regressions, without a model or GPU.

    python -m unittest serve.test_frontend -v
"""
import json
import threading
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, OutputParser


class QuotedToolCalls(unittest.TestCase):
    CALL = "<tool_call><function=write><parameter=text>hello</parameter></function></tool_call>"
    TOOLS = [{"name": "write", "parameters": {"properties": {"text": {"type": "string"}}}}]

    def parses(self, text, thinking=False):
        """Whole output, single characters, and every two-chunk boundary must agree."""
        chunks = [[text], list(text)] + [[text[:i], text[i:]] for i in range(1, len(text))]
        for stream_tools in (False, True):
            for pieces in chunks:
                parser = OutputParser(thinking=thinking, tools=self.TOOLS, stream_tools=stream_tools)
                events = []
                for piece in pieces:
                    events.extend(parser.feed(piece))
                events.extend(parser.finish())
                yield events

    def assert_text(self, text):
        for events in self.parses(text):
            self.assertFalse([e for e in events if e.kind.startswith("tool_")])
            self.assertEqual("".join(e.text for e in events), text)

    def test_backtick_and_tilde_fences_are_text(self):
        for fence in ("```", "~~~~", "````"):
            self.assert_text(f"Example:\n{fence}xml\n{self.CALL}\n{fence}\nDone")

    def test_inline_code_delimiter_lengths_are_text(self):
        for before, after in (("`", "`"), ("``a ` b ", "``"), ("```", "```"), ("`first\n", "`")):
            self.assert_text(f"Example: {before}{self.CALL}{after} stays text")

    def test_shorter_or_nonclosing_fences_do_not_release_a_call(self):
        for middle in ("```", "```` still code", "~~~", "    ````", "\t````"):
            self.assert_text(f"````xml\n{middle}\n{self.CALL}\n````\nDone")

    def test_unclosed_code_stays_text(self):
        for prefix in ("```xml\n", "Example: `", "Example: ``"):
            self.assert_text(prefix + self.CALL)

    def test_a_prose_opener_is_not_a_malformed_call(self):
        for text in ("Use a <tool_call> block, closed by </tool_call>.",
                     "Use <tool_call>\n<funX> as an example.</tool_call>",
                     "A partial <tool_call>\n<fun", "A bare <tool_call>"):
            self.assert_text(text)

    def test_real_calls_after_quotes_still_work(self):
        prefixes = ("", "Calling now: ", "Example: `" + self.CALL + "`\n",
                    "Example: ``" + self.CALL + "``\n",
                    "```xml\n" + self.CALL + "\n```\n",
                    "~~~xml\n" + self.CALL + "\n~~~~\n",
                    "Escaped \\` delimiter\n", "Mention <tool_call> block\n",
                    "    ~~~\n    some code\n    ~~~\n\n")
        for prefix in prefixes:
            for events in self.parses(prefix + self.CALL + "\n" + self.CALL):
                calls = [e.call for e in events if e.kind == "tool_call"]
                self.assertEqual([(c.name, c.arguments) for c in calls],
                                 [("write", {"text": "hello"})] * 2)
                self.assertEqual("".join(e.text for e in events if e.kind == "content").rstrip(), prefix.rstrip())
                starts = [e.call for e in events if e.kind == "tool_start"]
                if starts:
                    self.assertEqual([c.id for c in starts], [c.id for c in calls])
                    for call in calls:
                        args = "".join(e.text for e in events if e.kind == "tool_args" and e.call.id == call.id)
                        self.assertEqual(json.loads(args), call.arguments)

    def test_reasoning_quotes_do_not_change_content_state(self):
        reasoning = "Quoted ` and ```\n" + self.CALL
        for events in self.parses(reasoning + "</think>\n\n" + self.CALL, thinking=True):
            self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), reasoning)
            self.assertEqual(len([e for e in events if e.kind == "tool_call"]), 1)

    def test_code_after_a_real_call_still_stays_text(self):
        quoted = f"~~~xml\n{self.CALL}\n~~~"
        for events in self.parses("Calling now: " + self.CALL + "\n" + quoted + "\n" + self.CALL):
            self.assertEqual(len([e for e in events if e.kind == "tool_call"]), 2)
            content = "".join(e.text for e in events if e.kind == "content")
            self.assertIn(quoted, content)

    def test_markdown_in_arguments_does_not_change_content_state(self):
        text = self.CALL.replace("hello", "```xml\n`example`") + self.CALL
        for events in self.parses(text):
            self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"],
                             [{"text": "```xml\n`example`"}, {"text": "hello"}])

    def test_quoted_examples_are_text_in_both_api_shapes(self):
        # Exercise the real service parser and API serializers, including their streaming output,
        # with the in-process mock engine. No HTTP port, model or external tool is started.
        from serve.server import (ByteTokenizer, MockEngine, Service, anthropic_collect, anthropic_events,
                                  openai_chunks, openai_collect)

        template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        for text in (f"```xml\n{self.CALL}\n```", f"Example: ``{self.CALL}``",
                     "Use a <tool_call> block, closed by </tool_call>."):
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, text), tok, template)
            args = (svc, {"model": "mock"}, [1], False, self.TOOLS, 1000, threading.Event())
            chunks = list(openai_chunks(*args))
            deltas = [c["choices"][0]["delta"] for c in chunks if c]
            self.assertFalse(any(d.get("tool_calls") for d in deltas))
            result = openai_collect(iter(chunks))["choices"][0]
            self.assertEqual(result["finish_reason"], "stop")
            self.assertEqual(result["message"]["content"], text)
            events = list(anthropic_events(*args))
            self.assertFalse(any(e and e[1].get("content_block", {}).get("type") == "tool_use" for e in events))
            result = anthropic_collect(iter(events))
            self.assertEqual(result["stop_reason"], "end_turn")
            self.assertEqual(result["content"], [{"type": "text", "text": text}])


if __name__ == "__main__":
    unittest.main()
