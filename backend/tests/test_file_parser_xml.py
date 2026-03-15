"""Unit tests for XML parsing in FileParser."""

import os
import tempfile
import textwrap
import unittest

from app.utils.file_parser import FileParser


class TestExtractFromGenericXml(unittest.TestCase):
    def _write(self, content: str, suffix=".xml") -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        return f.name

    def tearDown(self):
        # nothing to clean up – each test removes its own file
        pass

    def test_simple_elements(self):
        path = self._write("<root><a>Hello</a><b>World</b></root>")
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        self.assertIn("Hello", result)
        self.assertIn("World", result)

    def test_nested_elements(self):
        path = self._write("<root><outer><inner>Deep text</inner></outer></root>")
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        self.assertIn("Deep text", result)

    def test_tail_text(self):
        path = self._write("<root><a>A</a>tail<b>B</b></root>")
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        self.assertIn("tail", result)

    def test_empty_xml(self):
        path = self._write("<root></root>")
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        self.assertEqual(result.strip(), "")

    def test_invalid_xml_raises(self):
        path = self._write("<root><unclosed>")
        try:
            with self.assertRaises(ValueError):
                FileParser.extract_text(path)
        finally:
            os.unlink(path)


class TestExtractFromMediawikiXml(unittest.TestCase):
    MEDIAWIKI_TEMPLATE = textwrap.dedent("""\
        <mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
          <page>
            <title>{title1}</title>
            <revision>
              <text>{text1}</text>
            </revision>
          </page>
          <page>
            <title>{title2}</title>
            <revision>
              <text>{text2}</text>
            </revision>
          </page>
        </mediawiki>
    """)

    def _write_mediawiki(self, title1="Art1", text1="Content one",
                         title2="Art2", text2="Content two") -> str:
        xml = self.MEDIAWIKI_TEMPLATE.format(
            title1=title1, text1=text1, title2=title2, text2=text2
        )
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8")
        f.write(xml)
        f.close()
        return f.name

    def test_both_articles_extracted(self):
        path = self._write_mediawiki()
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        self.assertIn("Art1", result)
        self.assertIn("Content one", result)
        self.assertIn("Art2", result)
        self.assertIn("Content two", result)

    def test_title_used_as_section_header(self):
        path = self._write_mediawiki(title1="Python (language)")
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        self.assertIn("=== Python (language) ===", result)

    def test_empty_text_page_skipped(self):
        path = self._write_mediawiki(title2="EmptyPage", text2="")
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        self.assertNotIn("EmptyPage", result)

    def test_multiple_pages_separated(self):
        path = self._write_mediawiki(
            title1="A", text1="alpha",
            title2="B", text2="beta",
        )
        try:
            result = FileParser.extract_text(path)
        finally:
            os.unlink(path)
        idx_a = result.index("alpha")
        idx_b = result.index("beta")
        self.assertLess(idx_a, idx_b)


class TestXmlInSupportedExtensions(unittest.TestCase):
    def test_xml_in_supported_extensions(self):
        self.assertIn(".xml", FileParser.SUPPORTED_EXTENSIONS)

    def test_unsupported_extension_raises(self):
        f = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        f.close()
        try:
            with self.assertRaises(ValueError):
                FileParser.extract_text(f.name)
        finally:
            os.unlink(f.name)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            FileParser.extract_text("/nonexistent/path/file.xml")


if __name__ == "__main__":
    unittest.main()
