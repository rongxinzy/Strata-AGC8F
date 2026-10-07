"""Model discovery, on-demand loading and safe unloading over the actual HTTP API."""
import json
import io
import subprocess
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, EngineStuck, MockEngine, Service, StrataEngine, serve


class ResidentEngine(StrataEngine):
    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.loaded = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class RestartAdmissionState(unittest.TestCase):
    """Restart the real wrapper with in-memory pipes; no engine process or GPU is started."""

    def setUp(self):
        def process(*args, **kwargs):
            proc = mock.Mock()
            proc.stdin = io.StringIO()
            proc.stdout = io.StringIO("INFO batch_slots=2\nREADY 4096 stop\n")
            proc.poll.return_value = None
            proc.wait.side_effect = lambda **kw: setattr(proc.poll, "return_value", 0)
            return proc
        for patch in (mock.patch("serve.server.subprocess.Popen", side_effect=process),
                      mock.patch("serve.server.contain"), mock.patch("serve.server.threading.Thread")):
            patch.start()
            self.addCleanup(patch.stop)
        self.engine = StrataEngine("missing-executable", ["--batch", "2"])
        self.addCleanup(self.engine.close)

    def test_restart_keeps_admission_state_but_resets_process_state(self):
        engine = self.engine
        cv, ctl, wait_lens = engine.slot_cv, engine.ctl, engine.wait_lens
        engine.waiting = 2
        wait_lens.extend([[64], [256]])
        engine.ctl_epoch = 7
        engine._yielded = (1, 32)
        engine.slot_busy[1] = True
        engine.slot_held[1] = [10, 20]
        engine.slot_used[1] = 123.0
        engine.slot_live[1] = {"state": "decoding"}
        engine.lines.put("old control line")
        engine.slot_q[1].put("old slot line")
        proc, lines, slot_q, wlock = engine.proc, engine.lines, engine.slot_q, engine.wlock
        with ctl:
            engine.restart()
            self.assertIs(engine.slot_cv, cv)
            self.assertIs(engine.ctl, ctl)
            self.assertTrue(engine.ctl.locked())
        self.assertIs(engine.wait_lens, wait_lens)
        self.assertEqual(engine.waiting, 2)
        self.assertEqual(engine.wait_lens, [[64], [256]])
        self.assertEqual(engine.ctl_epoch, 7)
        self.assertIsNot(engine.proc, proc)
        self.assertIsNot(engine.lines, lines)
        self.assertTrue(engine.lines.empty())
        self.assertIsNot(engine.slot_q, slot_q)
        for new, old in zip(engine.slot_q, slot_q):
            self.assertIsNot(new, old)
            self.assertTrue(new.empty())
        self.assertEqual(engine.slot_busy, [False, False])
        self.assertEqual(engine.slot_held, [[], []])
        self.assertEqual(engine.slot_used, [0.0, 0.0])
        self.assertEqual(engine.slot_live, [None, None])
        self.assertIsNone(engine._yielded)
        self.assertIsNot(engine.wlock, wlock)
        self.assertTrue(engine.alive())

    def test_waiter_resumes_after_restart_without_losing_its_entry(self):
        engine = self.engine
        ctl = engine.ctl
        engine.ctl_epoch = 7
        waiter = engine._take_control(threading.Event(), 128)
        try:
            with ctl:
                # Stop at a real heartbeat while the control lock is held; no racing test threads or long sleep.
                with mock.patch("serve.server.time.monotonic", side_effect=[0.0, 10.0, 10.0]):
                    self.assertIsNone(next(waiter))
                self.assertEqual(engine.waiting, 1)
                self.assertEqual(engine.wait_lens, [[128]])
                engine.restart()
            with self.assertRaises(StopIteration) as done:
                next(waiter)
            self.assertTrue(done.exception.value)
            self.assertTrue(engine.ctl.locked())
            self.assertEqual(engine.waiting, 0)
            self.assertEqual(engine.wait_lens, [])
            self.assertEqual(engine.ctl_epoch, 8)
        finally:
            waiter.close()
            if engine.ctl.locked():
                engine.ctl.release()


class Lifecycle(unittest.TestCase):
    def test_close_sends_eof_before_waiting_for_windows_reader(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = lambda **kwargs: self.assertTrue(proc.stdin.closed)
        engine.proc = proc
        engine.close()
        self.assertIsNone(engine.proc)
        proc.kill.assert_not_called()

    def test_close_does_not_forget_a_process_still_exiting(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("engine", 2)
        engine.proc = proc
        with self.assertRaisesRegex(EngineStuck, "still releasing"):
            engine.close()
        self.assertIs(engine.proc, proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_close_terminates_before_it_kills(self):
        # an engine that does not end on QUIT is terminated first (as the unload always did); kill is the last resort
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = [subprocess.TimeoutExpired("engine", 20), 0]
        engine.proc = proc
        engine.close()
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        self.assertIsNone(engine.proc)

    def test_native_lazy_constructor_does_not_start_a_process(self):
        engine = StrataEngine("missing-executable", ["--max-context", "16384"], lazy=True)
        self.assertFalse(engine.alive())
        self.assertEqual(engine.max_context, 16384)
        self.assertIsNone(engine.exit_code())
        engine.close()

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_discover_chat_unload_and_stream_reload(self):
        self.assertEqual(self.request("/api/health")[1]["service"], "strata")
        self.assertFalse(self.request("/v1/status")[1]["loaded"])
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertTrue(self.request("/props")[1]["models_autoload"])
        self.assertEqual(self.engine.starts, 0)
        chat = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"}
        for _ in range(2):
            code, answer = self.request("/v1/chat/completions", chat)
            self.assertEqual(code, 200)
            self.assertEqual(answer["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(self.engine.starts, 1)
        self.assertFalse(self.request("/v1/unload", {})[1]["loaded"])
        code, stream = self.request("/v1/chat/completions", {**chat, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("data: [DONE]", stream)
        self.assertEqual(self.engine.starts, 2)

    def test_busy_auth_and_foreign_origin_cannot_unload(self):
        self.engine.loaded = True
        with self.svc.fifo:
            self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request("/v1/unload", {})[0], 401)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret",
                                                        "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.engine.closes, 0)
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret"})[0], 200)

    def test_an_engine_that_died_on_load_is_reported_as_such(self):
        # EngineDied is a RuntimeError: it keeps its own answer ("the next request restarts it")
        def died():
            raise EngineDied("the engine ended while it started")
        self.engine.restart = died
        code, body = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(code, 503)
        self.assertIn("the next request restarts it", body["error"]["message"])

    def test_explicit_load_and_model_matching(self):
        self.assertEqual(self.request("/v1/load", {"model": "other-model"})[0], 404)
        self.assertEqual(self.engine.starts, 0)
        before = time.time()
        code, status = self.request("/v1/load", {"model": self.svc.model})
        self.assertEqual(code, 200)
        self.assertTrue(status["loaded"])
        self.assertEqual(self.engine.starts, 1)
        self.assertGreaterEqual(self.svc.last_request_at, before)
        self.svc.status["queued"] = 1
        self.assertEqual(self.request("/v1/load", {})[0], 409)
        self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.assertEqual(self.engine.closes, 0)
        self.assertEqual(self.request("/v1/load", {}, {"Content-Type": "text/plain"})[0], 415)


if __name__ == "__main__":
    unittest.main()
