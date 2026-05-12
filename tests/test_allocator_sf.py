"""Tests for StyleShield allocator chunking and allocation logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from scripts.allocator_sf import (
    Chunk,
    chunk_by_sentence,
    chunk_by_paragraph,
    chunk_text,
    weighted_chunk_rate,
)


class TestChunkBySentence:
    def test_basic_split(self):
        text = "这是第一句话。这是第二句话。这是第三句话。" * 20
        chunks = chunk_by_sentence(text, target_size=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_all_content_preserved(self):
        text = "句子一。句子二！句子三？句子四；"
        chunks = chunk_by_sentence(text, target_size=1000)
        joined = "".join(chunks)
        assert "句子一" in joined
        assert "句子四" in joined


class TestChunkByParagraph:
    def test_basic_split(self):
        text = "第一段内容很长很长很长。" * 10 + "\n\n" + "第二段也很长很长。" * 10
        chunks = chunk_by_paragraph(text, min_chars=50)
        assert len(chunks) == 2

    def test_empty(self):
        chunks = chunk_by_paragraph("", min_chars=100)
        assert len(chunks) == 0


class TestChunkText:
    def test_dispatch_sentence(self):
        text = "句子。" * 100
        chunks = chunk_text(text, method="sentence", chunk_size=50)
        assert len(chunks) > 1

    def test_dispatch_paragraph(self):
        text = "段一" * 50 + "\n\n" + "段二" * 50
        chunks = chunk_text(text, method="paragraph")
        assert len(chunks) >= 1

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            chunk_text("hello", method="invalid")


class TestWeightedRate:
    def test_all_original(self):
        chunks = [Chunk(index=0, text="a" * 100), Chunk(index=1, text="b" * 100)]
        chunks[0].p_ai_original = 0.8
        chunks[1].p_ai_original = 0.6
        rate = weighted_chunk_rate(chunks)
        assert abs(rate - 0.7) < 1e-6

    def test_with_rewrite(self):
        chunks = [Chunk(index=0, text="a" * 100), Chunk(index=1, text="b" * 100)]
        chunks[0].p_ai_original = 0.9
        chunks[0].selected_for_rewrite = True
        chunks[0].p_ai_rewritten = 0.1
        chunks[1].p_ai_original = 0.9
        rate = weighted_chunk_rate(chunks)
        assert abs(rate - 0.5) < 1e-6

    def test_weighted_by_length(self):
        chunks = [Chunk(index=0, text="a" * 300), Chunk(index=1, text="b" * 100)]
        chunks[0].p_ai_original = 0.8
        chunks[1].p_ai_original = 0.4
        rate = weighted_chunk_rate(chunks)
        expected = (0.8 * 300 + 0.4 * 100) / 400
        assert abs(rate - expected) < 1e-6

    def test_empty(self):
        rate = weighted_chunk_rate([])
        assert rate == 0.0


class TestChunk:
    def test_final_text_original(self):
        c = Chunk(index=0, text="原始文本")
        assert c.final_text == "原始文本"

    def test_final_text_rewritten(self):
        c = Chunk(index=0, text="原始文本")
        c.selected_for_rewrite = True
        c.rewritten_text = "改写文本"
        assert c.final_text == "改写文本"

    def test_char_count(self):
        c = Chunk(index=0, text="你好世界")
        assert c.char_count == 4

    def test_not_selected_returns_original(self):
        c = Chunk(index=0, text="原始")
        c.rewritten_text = "改写"
        c.selected_for_rewrite = False
        assert c.final_text == "原始"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
