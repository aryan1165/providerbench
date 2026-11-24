"""Runner and report: prefix matching, cache metrics, per-target control."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providerbench import report  # noqa: E402
from providerbench.runner import lcp, render  # noqa: E402


class PrefixTests(unittest.TestCase):
    def test_lcp(self):
        self.assertEqual(lcp("abcdef", "abcxyz"), 3)
        self.assertEqual(lcp("", "abc"), 0)
        self.assertEqual(lcp("same", "same"), 4)
        a = "x" * 100000 + "a"
        self.assertEqual(lcp(a, "x" * 100000 + "b"), 100000)

    def test_render_prefix_grows_with_turns(self):
        t0 = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}]}
        t1 = {"messages": t0["messages"] + [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]}
        r0, r1 = render(t0), render(t1)
        self.assertTrue(r1.startswith(r0))
        other = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "zz"}]}
        self.assertLess(lcp(render(other), r1), len(r0))


def rec(turn, prompt, cached, possible, reported=True):
    return {"suite": "synthetic", "graded": False, "verdict": "pass", "turn": turn, "t": 1.0, "e2e_ms": 1000,
            "prompt_tokens": prompt, "cached_tokens": cached, "cache_possible_tokens": possible,
            "cache_reported": reported, "completion_tokens": 10, "ttft_ms": 100}


class CacheMetricTests(unittest.TestCase):
    def test_hit_possible_reuse(self):
        recs = [rec(0, 1000, 0, 0), rec(0, 1000, 900, 950), rec(1, 2000, 1000, 1900)]
        s = report.summarize(recs)
        self.assertAlmostEqual(s["cache_pct"], 1900 / 4000)
        self.assertAlmostEqual(s["cache_possible_pct"], 2850 / 4000)
        self.assertAlmostEqual(s["cache_reuse_pct"], 1900 / 2850)
        self.assertEqual(s["cache_by_turn"]["0"]["n"], 2)

    def test_reuse_capped_and_not_reported(self):
        s = report.summarize([rec(0, 1000, 1000, 500)])          # provider cached more than we could see
        self.assertEqual(s["cache_reuse_pct"], 1.0)
        s = report.summarize([rec(0, 1000, 0, 500, reported=False)])
        self.assertIsNone(s["cache_pct"])
        self.assertIsNone(s["cache_reuse_pct"])
        self.assertFalse(s["cache_reported"])

    def test_verdict_level_when_close(self):
        a = dict(report.summarize([dict(rec(0, 10, 0, 0), graded=True, verdict="pass")] * 20 +
                                  [dict(rec(0, 10, 0, 0), graded=True, verdict="fail")] * 6))
        b = dict(report.summarize([dict(rec(0, 10, 0, 0), graded=True, verdict="pass")] * 21 +
                                  [dict(rec(0, 10, 0, 0), graded=True, verdict="fail")] * 5))
        self.assertIn("level", report.verdict([("A", a), ("B", b)])[0])



class KeyStoreTests(unittest.TestCase):
    def test_save_and_delete(self):
        import os
        import stat
        import tempfile
        from providerbench import config
        old = config.ENV_FILE
        with tempfile.TemporaryDirectory() as d:
            config.ENV_FILE = Path(d) / ".env"
            try:
                config.save_keys({"ACME_API_KEY": "k-123", "OTHER_KEY": "o-456"})
                self.assertEqual(stat.S_IMODE(config.ENV_FILE.stat().st_mode), 0o600)
                self.assertTrue(config.delete_key("ACME_API_KEY"))
                self.assertNotIn("ACME_API_KEY", config.ENV_FILE.read_text())
                self.assertIn("OTHER_KEY=o-456", config.ENV_FILE.read_text())
                self.assertNotIn("ACME_API_KEY", os.environ)
                self.assertFalse(config.delete_key("ACME_API_KEY"))
            finally:
                config.ENV_FILE = old
                os.environ.pop("OTHER_KEY", None)


class SkippedTests(unittest.TestCase):
    def test_skipped_not_errors_and_not_in_cost(self):
        sent = [dict(rec(0, 100, 0, 0), cost_usd=0.01) for _ in range(4)]
        skipped = [{"suite": "synthetic", "graded": False, "verdict": "skipped", "turn": 0, "error": "stopped"}] * 6
        s = report.summarize(sent + skipped)
        self.assertEqual((s["errored"], s["skipped"]), (0, 6))
        self.assertAlmostEqual(s["cost_per_task_usd"], 0.01)


class DegenerationTests(unittest.TestCase):
    def test_flags_loop_and_skips_foreign_for_foreign_prompts(self):
        from providerbench.runner import Run
        from providerbench.suites.base import Task
        en = Task(id="a", suite="agentic", messages=[{"role": "user", "content": "Explain caching."}])
        hi = Task(id="b", suite="agentic", messages=[{"role": "user", "content": "कैशिंग समझाइए"}])
        self.assertIn("loop:content", Run._degeneration(en, {"content": "Let me read the full split logic. " * 40})["flags"])
        self.assertEqual(Run._degeneration(en, {"content": "Caching keeps recent data close."})["flags"], [])
        hindi = {"content": "कैशिंग हाल का डेटा पास रखती है ताकि अगली बार जल्दी मिले"}
        self.assertTrue(any(f.startswith("foreign") for f in Run._degeneration(en, hindi)["flags"]))
        self.assertEqual(Run._degeneration(hi, hindi)["flags"], [])
        self.assertIn("loop:reasoning", Run._degeneration(en, {"content": "ok", "reasoning": "wait, " * 300})["flags"])


class ExtraDegenTests(unittest.TestCase):
    def test_leak_entropy_exhausted(self):
        from providerbench.runner import degeneration
        flags = lambda c, r="", f="stop": (degeneration([], {"content": c, "reasoning": r}, f) or {}).get("flags", [])  # noqa: E731
        self.assertIn("leak:content", flags("Sure.<|im_end|> more"))
        self.assertIn("entropy:reasoning", flags("ok", "abcabcab" * 200))
        self.assertIn("exhausted:content", flags("", "thinking " + "x" * 50, "length"))
        self.assertEqual(flags("A normal, varied answer about caching and latency trade-offs."), [])


class SpanAndReviewTests(unittest.TestCase):
    def test_loop_span_covers_repeats_across_newlines(self):
        from providerbench.runner import _span
        text = "Intro. " + "red a\nred a " * 60 + "End."
        s, e = _span(text, "loop", "red a", 6.0)
        self.assertEqual(text[s:s + 5], "red a")
        self.assertTrue(text[e:].startswith(" End.") or text[e:].startswith("End.") or e > len(text) - 8)

    def test_missed_label_with_span(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            from providerbench import runner
            old = runner.RESULTS
            runner.RESULTS = Path(d)
            try:
                (Path(d) / "r1").mkdir()
                runner.set_review("r1", "x:y", "t1", "missed", "looks looped", [{"channel": "content", "start": 5, "end": 50}])
                doc = runner.reviews("r1")
                self.assertEqual(doc["x:y|t1"]["label"], "missed")
                self.assertEqual(doc["x:y|t1"]["spans"], [{"channel": "content", "start": 5, "end": 50}])
                with self.assertRaises(ValueError):
                    runner.set_review("r1", "x:y", "t1", "maybe")
            finally:
                runner.RESULTS = old


class DegenScopeTests(unittest.TestCase):
    def test_scope_real_prompts_only(self):
        from providerbench.runner import degeneration
        loop = {"content": "Let me read the full split logic. " * 40}
        self.assertIsNone(degeneration([], loop, "stop", "synthetic"))          # random prompts: off
        for suite in ("agentic", "custom", "math", "gpqa", "tools", "robustness"):
            self.assertIn("loop:content", degeneration([], loop, "stop", suite)["flags"], suite)


class SessionSuiteTests(unittest.TestCase):
    def test_closed_loop_messages_and_tool_answers(self):
        from providerbench.suites.session import closed_loop_messages
        prev = [{"role": "system", "content": "s"}, {"role": "user", "content": "task 1"}]
        reply = {"content": "Writing the fix.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "str_replace_editor", "arguments": '{"command": "create", "path": "/testbed/fix.py", "file_text": "x"}'}},
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pytest"}'}}]}
        msgs = closed_loop_messages(prev, reply, "task 2")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool", "tool", "user"])
        self.assertIn("created successfully", msgs[3]["content"])
        self.assertIn("Not available", msgs[4]["content"])
        self.assertEqual(msgs[-1]["content"], "task 2")

    def test_cut_at_clean_boundary(self):
        from providerbench.suites.session import cut_at
        msgs = [{"role": "user", "content": "u" * 3600},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "a", "content": "r" * 3600},
                {"role": "assistant", "content": "a" * 3600},
                {"role": "user", "content": "u" * 3600}]
        k = cut_at(msgs, 2600)
        self.assertIn(msgs[k - 1]["role"], ("tool", "assistant"))
        self.assertLessEqual(k, 4)


class HttpRetryTests(unittest.TestCase):
    """data.http_get waits out transient errors (Hugging Face 503s) and fails fast on a 404."""

    def _run(self, seq, url="https://example.invalid/x"):
        import io
        import urllib.error
        from unittest import mock
        from providerbench import data
        sleeps = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        def fake(req, timeout=0):
            c = seq.pop(0)
            if isinstance(c, int):
                raise urllib.error.HTTPError(req.full_url, c, "x", {}, None)
            return Resp(c)

        with mock.patch.object(data.urllib.request, "urlopen", fake), \
                mock.patch.object(data.time, "sleep", sleeps.append):
            return data.http_get(url, cache=False), sleeps

    def test_retries_503_then_succeeds(self):
        body, sleeps = self._run([503, 503, b"ok"])
        self.assertEqual(body, "ok")
        self.assertEqual(sleeps, [2, 4])

    def test_404_fails_at_once(self):
        with self.assertRaises(RuntimeError) as e:
            self._run([404])
        self.assertNotIn("try again", str(e.exception))

    def test_gives_up_with_hint(self):
        with self.assertRaises(RuntimeError) as e:
            self._run([503] * 7)
        self.assertIn("try again in a few minutes", str(e.exception))


class CustomProviderTests(unittest.TestCase):
    def test_add_private_no_key_and_remove(self):
        import tempfile
        import threading
        from providerbench import config, mock, probe
        from providerbench.server import LocalServer
        srv = LocalServer(("127.0.0.1", 0), mock.Handler)          # only GET /models is used
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d/v1" % srv.server_address[1]
        old = config.USER_PROVIDERS_DIR
        with tempfile.TemporaryDirectory() as d:
            config.USER_PROVIDERS_DIR = Path(d)
            try:
                probe.add("my-vllm", url, name="My vLLM", private=True, no_key=True,
                          run_probes=False, log=lambda *_: None)
                prov = config.load_providers()["my-vllm"]
                self.assertTrue(prov.custom and prov.has_key())
                self.assertEqual(prov.models["glm-5.3-flash"], "glm-5.3-flash")
                self.assertEqual(config.resolve("my-vllm:glm-5.3-flash").model_id, "glm-5.3-flash")
                with self.assertRaises(ValueError):
                    probe.add("fireworks", url, private=True, no_key=True, run_probes=False)
                probe.remove("my-vllm")
                self.assertNotIn("my-vllm", config.load_providers())
            finally:
                config.USER_PROVIDERS_DIR = old
                srv.shutdown()
                srv.server_close()

    def test_placeholder_host(self):
        import os
        from providerbench import config
        p = config.Provider(id="bedrock", name="Bedrock", base_url="https://bedrock-mantle.{AWS_REGION}.api.aws/v1")
        os.environ["AWS_REGION"] = "eu-west-1"
        try:
            self.assertEqual(p.host(), "bedrock-mantle.eu-west-1.api.aws")
        finally:
            os.environ.pop("AWS_REGION")


if __name__ == "__main__":
    unittest.main()
