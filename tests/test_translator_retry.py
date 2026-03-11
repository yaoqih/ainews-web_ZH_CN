import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("LLM_API_KEY", "test-key")

import translator


def _response_with_content(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TranslateTextChunkRetryTests(unittest.TestCase):
    def test_retries_until_success(self):
        calls = []

        def create(**_kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("temporary failure")
            return _response_with_content("翻译成功")

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        with mock.patch.object(translator, "client", fake_client):
            result = translator.translate_text_chunk("hello world")

        self.assertEqual(result, "翻译成功")
        self.assertEqual(len(calls), 3)

    def test_stops_after_three_attempts_and_falls_back_to_source_text(self):
        calls = []

        def create(**_kwargs):
            calls.append(1)
            raise RuntimeError("still failing")

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        with mock.patch.object(translator, "client", fake_client):
            result = translator.translate_text_chunk("original text")

        self.assertEqual(result, "original text")
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
