"""Tests for content_blocks encode/decode utilities."""
from __future__ import annotations

import json

from llming_models.content_blocks import (
    encode_content_block,
    extract_content_blocks,
    strip_content_blocks,
)


class TestEncodeContentBlock:
    def test_basic_dict(self):
        result = encode_content_block("plotly", {"data": [1, 2, 3]})
        assert result == '\n\n```plotly\n{"data": [1, 2, 3]}\n```'

    def test_list_data(self):
        result = encode_content_block("rich_mcp", [{"x": 1}])
        assert result.startswith("\n\n```rich_mcp\n")
        assert result.endswith("\n```")
        blocks = extract_content_blocks(result)
        assert len(blocks) == 1
        assert json.loads(blocks[0][1]) == [{"x": 1}]

    def test_unicode(self):
        result = encode_content_block("test", {"text": "Hallo Welt"})
        assert "Hallo Welt" in result  # ensure_ascii=False


class TestExtractContentBlocks:
    def test_no_blocks(self):
        assert extract_content_blocks("Hello world") == []

    def test_single_block(self):
        content = 'Some text\n\n```plotly\n{"data": [1]}\n```'
        blocks = extract_content_blocks(content)
        assert len(blocks) == 1
        assert blocks[0][0] == "plotly"
        assert json.loads(blocks[0][1]) == {"data": [1]}

    def test_multiple_blocks(self):
        content = (
            "Text before\n\n"
            '```rich_mcp\n{"id": "a"}\n```'
            "\n\nMiddle text\n\n"
            '```plotly\n{"traces": []}\n```'
        )
        blocks = extract_content_blocks(content)
        assert len(blocks) == 2
        assert blocks[0][0] == "rich_mcp"
        assert blocks[1][0] == "plotly"

    def test_round_trip(self):
        data = {"key": "value", "nested": {"a": 1}}
        encoded = encode_content_block("mytype", data)
        blocks = extract_content_blocks(encoded)
        assert len(blocks) == 1
        assert blocks[0][0] == "mytype"
        assert json.loads(blocks[0][1]) == data

    def test_multiline_json(self):
        """Blocks with multiline JSON content should extract correctly."""
        content = '```test\n{\n  "a": 1,\n  "b": 2\n}\n```'
        blocks = extract_content_blocks(content)
        assert len(blocks) == 1
        assert json.loads(blocks[0][1]) == {"a": 1, "b": 2}


class TestStripContentBlocks:
    def test_no_blocks(self):
        assert strip_content_blocks("Hello world") == "Hello world"

    def test_strip_single_block(self):
        content = 'Hello\n\n```plotly\n{"data": []}\n```'
        assert strip_content_blocks(content) == "Hello"

    def test_strip_preserves_text(self):
        content = (
            "Before\n\n"
            '```rich_mcp\n{"id": "x"}\n```'
            "\n\nAfter"
        )
        stripped = strip_content_blocks(content)
        assert "Before" in stripped
        assert "After" in stripped
        assert "rich_mcp" not in stripped

    def test_strip_multiple_blocks(self):
        content = '```a\n{}\n```\ntext\n```b\n{}\n```'
        stripped = strip_content_blocks(content)
        assert "text" in stripped
        assert "```" not in stripped
