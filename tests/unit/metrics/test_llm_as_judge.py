# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from types import SimpleNamespace

import pytest

from lighteval.metrics.utils.llm_as_judge import JudgeLM


def _fake_completion_response(text: str = "judgment"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _make_judge(**kwargs):
    return JudgeLM(
        model="openai/dummy-model",
        templates=lambda question, answer, options=None, gold=None, **kw: [{"role": "user", "content": question}],
        process_judge_response=lambda response: response,
        judge_backend="litellm",
        # `increase_max_tokens_for_reasoning` would rewrite max_tokens for reasoning
        # models; disable it so the test asserts on the value the caller passed in.
        backend_options={"caching": False, "increase_max_tokens_for_reasoning": False},
        **kwargs,
    )


@pytest.mark.parametrize("max_tokens", [64, 512])
def test_litellm_judge_passes_max_tokens_as_int(monkeypatch, max_tokens):
    """`max_tokens` must reach litellm as an int, not a one-element sequence.

    Sending a sequence makes any spec-compliant OpenAI-compatible server reject the
    request with a 400, after which the judge returns its error string and that string
    gets scored as a real judgment.
    """
    litellm = pytest.importorskip("litellm")

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_completion_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    judge = _make_judge(max_tokens=max_tokens)
    _, _, responses = judge.evaluate_answer_batch(
        questions=["Why is 2 + 2 = 4?"], answers=["4"], options=[None], golds=["4"]
    )

    assert captured["max_tokens"] == max_tokens
    assert isinstance(captured["max_tokens"], int)
    assert not isinstance(captured["max_tokens"], (list, tuple))
    assert responses == ["judgment"]


def test_litellm_judge_omits_max_tokens_when_unset(monkeypatch):
    """No `max_tokens` on the judge means no `max_tokens` key in the litellm call."""
    litellm = pytest.importorskip("litellm")

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_completion_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    judge = _make_judge()
    judge.evaluate_answer_batch(questions=["Why is 2 + 2 = 4?"], answers=["4"], options=[None], golds=["4"])

    assert "max_tokens" not in captured
