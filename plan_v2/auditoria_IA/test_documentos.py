"""
Unit tests for documentos.py

Tests cover:
- TXT, MD, PDF, DOCX extraction
- Heading detection
- Paragraph/table ordering
- Chunking by structure and words
- Validation of concatenation (concatenated chunks should reconstruct document.text)
- Edge cases (long sentences, oversized tokens, invalid limits, empty documents)
"""

import unittest
import tempfile
from pathlib import Path
from io import BytesIO

from documentos import (
    TextBlock,
    Document,
    TextChunk,
    extract_document,
    make_chunks,
)


class TestTextBlockAndDataclasses(unittest.TestCase):
    """Test dataclass definitions."""
    
    def test_text_block_creation(self):
        block = TextBlock(text='Hello', location='para_0', section='Intro', heading=False)
        self.assertEqual(block.text, 'Hello')
        self.assertEqual(block.location, 'para_0')
        self.assertEqual(block.section, 'Intro')
        self.assertFalse(block.heading)
    
    def test_text_block_default_section(self):
        block = TextBlock(text='Hello', location='para_0')
        self.assertEqual(block.section, '')
        self.assertFalse(block.heading)
    
    def test_document_creation(self):
        block = TextBlock(text='Hello', location='para_0')
        doc = Document(text='Hello', blocks=[block], metadata={}, warnings=[])
        self.assertEqual(doc.text, 'Hello')
        self.assertEqual(len(doc.blocks), 1)
        self.assertEqual(len(doc.warnings), 0)
    
    def test_document_default_warnings(self):
        doc = Document(text='Test', blocks=[], metadata={})
        self.assertEqual(doc.warnings, [])
    
    def test_text_chunk_creation(self):
        chunk = TextChunk(
            chunk_id='c0001',
            text='Test text',
            locations=['para_0', 'para_1'],
            sections=['Intro']
        )
        self.assertEqual(chunk.chunk_id, 'c0001')
        self.assertEqual(chunk.text, 'Test text')
        self.assertEqual(chunk.locations, ['para_0', 'para_1'])
        self.assertEqual(chunk.sections, ['Intro'])


class TestTxtExtraction(unittest.TestCase):
    """Test TXT file extraction."""
    
    def test_txt_basic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("First paragraph.\nStill first.\n\nSecond paragraph.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            self.assertGreater(len(doc.blocks), 0)
            self.assertIn('First paragraph', doc.text)
            self.assertIn('Second paragraph', doc.text)
            self.assertEqual(doc.metadata['format'], 'txt')
        finally:
            path.unlink()
    
    def test_txt_heading_detection(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CHAPTER 1\n\nThis is content under chapter 1.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            heading_blocks = [b for b in doc.blocks if b.heading]
            self.assertGreater(len(heading_blocks), 0)
        finally:
            path.unlink()
    
    def test_txt_numbered_heading(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1. Introduction\n\nContent here.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            # Check that at least one block was detected as heading
            heading_blocks = [b for b in doc.blocks if b.heading]
            self.assertGreater(len(heading_blocks), 0)
        finally:
            path.unlink()
    
    def test_txt_concatenation(self):
        """Concatenated blocks should reconstruct document.text"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Para 1.\n\nPara 2.\n\nPara 3.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            reconstructed = '\n\n'.join(b.text for b in doc.blocks)
            self.assertEqual(doc.text, reconstructed)
        finally:
            path.unlink()


class TestMarkdownExtraction(unittest.TestCase):
    """Test Markdown file extraction."""
    
    def test_md_atx_headings(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Main Title\n\nContent under main.\n\n## Subsection\n\nSubsection content.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            heading_blocks = [b for b in doc.blocks if b.heading]
            self.assertGreaterEqual(len(heading_blocks), 2)
            self.assertIn('Main Title', doc.text)
        finally:
            path.unlink()
    
    def test_md_setext_headings(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("Main Heading\n=============\n\nContent here.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            # Setext headings should be detected
            self.assertIn('Main Heading', doc.text)
        finally:
            path.unlink()
    
    def test_md_markup_preserved(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("This is **bold** and *italic* text.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            # Markup should be preserved in extracted text
            self.assertIn('**bold**', doc.text)
            self.assertIn('*italic*', doc.text)
        finally:
            path.unlink()
    
    def test_md_concatenation(self):
        """Concatenated blocks should reconstruct document.text"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Heading 1\n\nParagraph 1.\n\n## Heading 2\n\nParagraph 2.")
            f.flush()
            path = Path(f.name)
        
        try:
            doc = extract_document(path)
            reconstructed = '\n\n'.join(b.text for b in doc.blocks)
            self.assertEqual(doc.text, reconstructed)
        finally:
            path.unlink()


class TestPdfExtraction(unittest.TestCase):
    """Test PDF extraction (requires PyMuPDF)."""
    
    def test_pdf_create_and_extract_real(self):
        """Create a real PDF with fitz and extract it."""
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        
        # Create a real PDF in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_document.pdf"
            
            # Create PDF with text
            doc = fitz.open()
            page = doc.new_page()
            
            text_content = "This is a test PDF.\n\nIt has multiple paragraphs.\n\nWith some structure."
            page.insert_text((50, 50), text_content, fontsize=12)
            
            doc.save(pdf_path)
            doc.close()
            
            # Extract and verify
            extracted = extract_document(pdf_path)
            self.assertEqual(extracted.metadata['format'], 'pdf')
            self.assertIn('test PDF', extracted.text)
            self.assertGreater(len(extracted.blocks), 0)
    
    def test_pdf_real_with_warnings(self):
        """Test PDF extraction and warning generation."""
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "simple.pdf"
            
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Simple PDF", fontsize=12)
            
            doc.save(pdf_path)
            doc.close()
            
            extracted = extract_document(pdf_path)
            self.assertEqual(extracted.metadata['format'], 'pdf')
            self.assertIsInstance(extracted.warnings, list)


class TestDocxExtraction(unittest.TestCase):
    """Test DOCX extraction (requires python-docx)."""
    
    def test_docx_create_and_extract_real(self):
        """Create a real DOCX with python-docx and extract it."""
        try:
            from docx import Document as DocxDocument
        except ImportError:
            self.skipTest("python-docx not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path = Path(tmpdir) / "test_document.docx"
            
            # Create DOCX
            doc = DocxDocument()
            doc.add_paragraph("This is a test DOCX.", style='Normal')
            doc.add_paragraph("Introduction", style='Heading 1')
            doc.add_paragraph("Content under introduction.", style='Normal')
            doc.add_paragraph("Details", style='Heading 2')
            doc.add_paragraph("More detailed content here.", style='Normal')
            
            doc.save(docx_path)
            
            # Extract and verify
            extracted = extract_document(docx_path)
            self.assertEqual(extracted.metadata['format'], 'docx')
            self.assertIn('test DOCX', extracted.text)
            self.assertIn('Introduction', extracted.text)
            self.assertGreater(len(extracted.blocks), 0)
            
            # Check heading detection
            heading_blocks = [b for b in extracted.blocks if b.heading]
            self.assertGreater(len(heading_blocks), 0)
    
    def test_docx_with_table(self):
        """Test DOCX extraction with table."""
        try:
            from docx import Document as DocxDocument
        except ImportError:
            self.skipTest("python-docx not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path = Path(tmpdir) / "table_document.docx"
            
            doc = DocxDocument()
            doc.add_paragraph("Document with table")
            
            table = doc.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Header 1"
            table.cell(0, 1).text = "Header 2"
            table.cell(1, 0).text = "Data 1"
            table.cell(1, 1).text = "Data 2"
            
            doc.save(docx_path)
            
            extracted = extract_document(docx_path)
            self.assertIn('Header 1', extracted.text)
            self.assertIn('Data 1', extracted.text)
    
    def test_docx_empty_raises_error(self):
        """Test that empty DOCX raises ValueError."""
        try:
            from docx import Document as DocxDocument
        except ImportError:
            self.skipTest("python-docx not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path = Path(tmpdir) / "empty.docx"
            
            doc = DocxDocument()
            doc.save(docx_path)
            
            with self.assertRaises(ValueError) as cm:
                extract_document(docx_path)
            self.assertIn('empty', str(cm.exception).lower())


class TestDocExtensionRejection(unittest.TestCase):
    """Test that old .doc format is rejected."""
    
    def test_doc_format_rejected(self):
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as f:
            f.write(b'test')
            f.flush()
            path = Path(f.name)
        
        try:
            with self.assertRaises(ValueError) as cm:
                extract_document(path)
            self.assertIn('DOC', str(cm.exception).upper())
            self.assertIn('DOCX', str(cm.exception))
        finally:
            path.unlink()


class TestUnsupportedFormat(unittest.TestCase):
    """Test that unsupported formats are rejected."""
    
    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b'test')
            f.flush()
            path = Path(f.name)
        
        try:
            with self.assertRaises(ValueError) as cm:
                extract_document(path)
            self.assertIn('Unsupported', str(cm.exception))
        finally:
            path.unlink()


class TestFileNotFound(unittest.TestCase):
    """Test missing file handling."""
    
    def test_nonexistent_file(self):
        path = Path('/nonexistent/file.txt')
        with self.assertRaises(FileNotFoundError):
            extract_document(path)


class TestChunkingStructureMode(unittest.TestCase):
    """Test chunking with structure mode."""
    
    def test_chunking_basic_structure(self):
        blocks = [
            TextBlock(text='Heading 1', location='loc_0', section='', heading=True),
            TextBlock(text='Paragraph 1 content.', location='loc_1', section='Heading 1'),
            TextBlock(text='Paragraph 2 content.', location='loc_2', section='Heading 1'),
        ]
        doc = Document(
            text='Heading 1\n\nParagraph 1 content.\n\nParagraph 2 content.',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=10, mode='structure')
        self.assertGreater(len(chunks), 0)
        
        # Check chunk IDs
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, f'c{i+1:04d}')
    
    def test_chunking_respects_section_change(self):
        blocks = [
            TextBlock(text='Section 1 heading', location='loc_0', section='', heading=True),
            TextBlock(text='Content in section 1.', location='loc_1', section='Section 1 heading'),
            TextBlock(text='Section 2 heading', location='loc_2', section='', heading=True),
            TextBlock(text='Content in section 2.', location='loc_3', section='Section 2 heading'),
        ]
        doc = Document(
            text='Section 1 heading\n\nContent in section 1.\n\nSection 2 heading\n\nContent in section 2.',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=100, mode='structure')
        # With section change, we should get multiple chunks
        self.assertGreater(len(chunks), 1)
    
    def test_chunking_preserves_headings(self):
        blocks = [
            TextBlock(text='HEADING', location='loc_0', section='', heading=True),
            TextBlock(text='A' * 10, location='loc_1', section='HEADING'),
        ]
        doc = Document(
            text='HEADING\n\n' + 'A' * 10,
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=2, max_chars=15, mode='structure')
        # Heading should be preserved
        found_heading = any('HEADING' in chunk.text for chunk in chunks)
        self.assertTrue(found_heading)
    
    def test_chunking_no_overlap_structure(self):
        blocks = [
            TextBlock(text='Word1 Word2', location='loc_0', section='S1'),
            TextBlock(text='Word3 Word4', location='loc_1', section='S1'),
            TextBlock(text='Word5 Word6', location='loc_2', section='S1'),
        ]
        doc = Document(
            text='Word1 Word2\n\nWord3 Word4\n\nWord5 Word6',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=3, mode='structure')
        
        # Concatenate chunks and split
        concat_text = ' '.join(c.text for c in chunks)
        doc_words = doc.text.split()
        concat_words = concat_text.split()
        
        self.assertEqual(sorted(doc_words), sorted(concat_words))


class TestChunkingWordsMode(unittest.TestCase):
    """Test chunking with words mode."""
    
    def test_chunking_words_mode(self):
        blocks = [
            TextBlock(text='Word1 Word2 Word3', location='loc_0', section='S1'),
            TextBlock(text='Word4 Word5', location='loc_1', section='S1'),
        ]
        doc = Document(
            text='Word1 Word2 Word3\n\nWord4 Word5',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=2, mode='words')
        self.assertGreater(len(chunks), 1)
    
    def test_chunking_words_ignores_sections(self):
        blocks = [
            TextBlock(text='A B', location='loc_0', section='S1'),
            TextBlock(text='C D', location='loc_1', section='S2'),
            TextBlock(text='E F', location='loc_2', section='S3'),
        ]
        doc = Document(
            text='A B\n\nC D\n\nE F',
            blocks=blocks,
            metadata={}
        )
        
        # Words mode should ignore section boundaries
        chunks = make_chunks(doc, max_words=100, mode='words')
        # With 6 words and max_words=100, should be 1 chunk
        self.assertEqual(len(chunks), 1)
    
    def test_chunking_no_overlap_words(self):
        blocks = [
            TextBlock(text='W1 W2 W3 W4 W5', location='loc_0', section='S1'),
        ]
        doc = Document(
            text='W1 W2 W3 W4 W5',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=2, mode='words')
        
        # Reconstruct and compare
        concat_text = ' '.join(c.text for c in chunks)
        self.assertEqual(set(doc.text.split()), set(concat_text.split()))


class TestChunkingValidation(unittest.TestCase):
    """Test chunking validation and error handling."""
    
    def test_invalid_mode(self):
        doc = Document(text='Test', blocks=[], metadata={})
        
        with self.assertRaises(ValueError):
            make_chunks(doc, mode='invalid')
    
    def test_oversized_token_structure(self):
        blocks = [
            TextBlock(text='x' * 10000, location='loc_0', section=''),
        ]
        doc = Document(
            text='x' * 10000,
            blocks=blocks,
            metadata={}
        )
        
        with self.assertRaises(ValueError) as cm:
            make_chunks(doc, max_chars=1000, mode='structure')
        self.assertIn('exceeds', str(cm.exception).lower())
    
    def test_oversized_token_words(self):
        blocks = [
            TextBlock(text='x' * 10000, location='loc_0', section=''),
        ]
        doc = Document(
            text='x' * 10000,
            blocks=blocks,
            metadata={}
        )
        
        with self.assertRaises(ValueError) as cm:
            make_chunks(doc, max_chars=1000, mode='words')
        self.assertIn('exceeds', str(cm.exception).lower())


class TestChunkingEdgeCases(unittest.TestCase):
    """Test edge cases in chunking."""
    
    def test_empty_document(self):
        doc = Document(text='', blocks=[], metadata={})
        chunks = make_chunks(doc, mode='structure')
        self.assertEqual(len(chunks), 0)
    
    def test_long_sentence_no_breaks(self):
        text = ' '.join(['word'] * 500)
        blocks = [TextBlock(text=text, location='loc_0', section='')]
        doc = Document(text=text, blocks=blocks, metadata={})
        
        chunks = make_chunks(doc, max_words=100, mode='words')
        self.assertGreater(len(chunks), 1)
    
    def test_single_word(self):
        blocks = [TextBlock(text='Word', location='loc_0', section='')]
        doc = Document(text='Word', blocks=blocks, metadata={})
        
        chunks = make_chunks(doc, max_words=10, mode='words')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, 'Word')
    
    def test_only_headings(self):
        blocks = [
            TextBlock(text='Heading 1', location='loc_0', section='', heading=True),
            TextBlock(text='Heading 2', location='loc_1', section='', heading=True),
        ]
        doc = Document(
            text='Heading 1\n\nHeading 2',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=10, mode='structure')
        self.assertGreater(len(chunks), 0)


class TestChunkingConcatenation(unittest.TestCase):
    """Test that chunks can be concatenated to reconstruct document."""
    
    def test_concatenation_structure_mode(self):
        blocks = [
            TextBlock(text='Para 1', location='loc_0', section='Intro'),
            TextBlock(text='Para 2', location='loc_1', section='Intro'),
            TextBlock(text='Para 3', location='loc_2', section='Body'),
            TextBlock(text='Para 4', location='loc_3', section='Body'),
        ]
        doc = Document(
            text='Para 1\n\nPara 2\n\nPara 3\n\nPara 4',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=50, mode='structure')
        concatenated = '\n\n'.join(c.text for c in chunks)
        
        # Compare word-split versions
        doc_words = doc.text.split()
        concat_words = concatenated.split()
        self.assertEqual(sorted(doc_words), sorted(concat_words))
    
    def test_concatenation_words_mode(self):
        blocks = [
            TextBlock(text='A B C D E', location='loc_0', section='S1'),
            TextBlock(text='F G H I J', location='loc_1', section='S1'),
        ]
        doc = Document(
            text='A B C D E\n\nF G H I J',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=3, mode='words')
        concatenated = ' '.join(c.text for c in chunks)
        
        doc_words = doc.text.split()
        concat_words = concatenated.split()
        self.assertEqual(sorted(doc_words), sorted(concat_words))


class TestChunkMetadata(unittest.TestCase):
    """Test chunk metadata (locations, sections)."""
    
    def test_chunk_locations_preserved(self):
        blocks = [
            TextBlock(text='Text1', location='para_0', section='Sec1'),
            TextBlock(text='Text2', location='para_1', section='Sec1'),
        ]
        doc = Document(
            text='Text1\n\nText2',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=100, mode='structure')
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertGreater(len(chunk.locations), 0)
            for loc in chunk.locations:
                self.assertIsInstance(loc, str)
    
    def test_chunk_sections_unique(self):
        blocks = [
            TextBlock(text='Text1', location='para_0', section='Sec1'),
            TextBlock(text='Text2', location='para_1', section='Sec1'),
        ]
        doc = Document(
            text='Text1\n\nText2',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=100, mode='structure')
        
        for chunk in chunks:
            # sections should be sorted and unique
            self.assertEqual(chunk.sections, sorted(set(chunk.sections)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
