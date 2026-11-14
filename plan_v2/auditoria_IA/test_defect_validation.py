"""
Focused regression tests to validate known defects before fixing.
"""

import unittest
from pathlib import Path
import tempfile

from documentos import (
    TextBlock,
    Document,
    extract_document,
    make_chunks,
)


class TestDefectStructureModeBlockSplitting(unittest.TestCase):
    """Test that structure mode splits oversized blocks exceeding max_words."""
    
    def test_oversized_block_exceeding_max_words_gets_split(self):
        """A single block with 500+ words should be split across chunks."""
        # Create one block with 501 words
        text_words = ' '.join(['word'] * 501)
        blocks = [
            TextBlock(text=text_words, location='para_0', section='Section1'),
        ]
        doc = Document(
            text=text_words,
            blocks=blocks,
            metadata={}
        )
        
        # With max_words=400, a 501-word block should be split
        chunks = make_chunks(doc, max_words=400, mode='structure')
        
        # Should produce at least 2 chunks
        self.assertGreaterEqual(len(chunks), 2, 
            "Block with 501 words should be split with max_words=400")
        
        # Verify word order is preserved exactly
        doc_words = doc.text.split()
        chunk_words = ' '.join(c.text for c in chunks).split()
        self.assertEqual(doc_words, chunk_words,
            "Word order must be preserved exactly (not sorted/deduplicated)")


class TestDefectWordsModeLoci(unittest.TestCase):
    """Test that words mode correctly tracks block locations."""
    
    def test_words_mode_locations_map_to_blocks(self):
        """Words mode chunks should track which blocks contributed."""
        blocks = [
            TextBlock(text='alpha beta', location='para_0', section='S1'),
            TextBlock(text='gamma delta', location='para_1', section='S2'),
            TextBlock(text='epsilon zeta', location='para_2', section='S3'),
        ]
        doc = Document(
            text='alpha beta\n\ngamma delta\n\nepsilon zeta',
            blocks=blocks,
            metadata={}
        )
        
        # With max_words=2, should split at block boundaries
        chunks = make_chunks(doc, max_words=2, mode='words')
        
        # Each chunk should have locations tied to contributing blocks
        for chunk in chunks:
            # locations should be non-empty and refer to actual blocks
            self.assertGreater(len(chunk.locations), 0,
                "Chunk must have non-empty locations list")
            for loc in chunk.locations:
                # Verify location is from original blocks
                self.assertIn(loc, [b.location for b in blocks])


class TestDefectHeadingBoundary(unittest.TestCase):
    """Test heading boundary behavior—headings should start fresh chunks."""
    
    def test_heading_forces_new_chunk_and_resets_section(self):
        """A heading should force a new chunk and establish a new section."""
        blocks = [
            TextBlock(text='content1 content2', location='para_0', section='OldSection'),
            TextBlock(text='NEW HEADING', location='para_1', section='', heading=True),
            TextBlock(text='content3', location='para_2', section='NEW HEADING'),
        ]
        doc = Document(
            text='content1 content2\n\nNEW HEADING\n\ncontent3',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=100, mode='structure')
        
        # Find chunk with heading
        heading_chunk = None
        for chunk in chunks:
            if 'NEW HEADING' in chunk.text:
                heading_chunk = chunk
                break
        
        self.assertIsNotNone(heading_chunk, "Should find chunk with heading")
        # Heading chunk should NOT contain 'content1 content2'
        self.assertNotIn('content1 content2', heading_chunk.text,
            "Heading should start new chunk, not group with previous section")


class TestDefectPdfSectionCarry(unittest.TestCase):
    """Test PDF section tracking across pages."""
    
    def test_pdf_section_carries_across_pages(self):
        """PDF should maintain section across page boundaries."""
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "multipage.pdf"
            
            doc = fitz.open()
            
            # Page 1: heading
            page1 = doc.new_page()
            page1.insert_text((50, 50), "SECTION A", fontsize=12)
            page1.insert_text((50, 100), "Page 1 content", fontsize=10)
            
            # Page 2: more content (should still be in SECTION A)
            page2 = doc.new_page()
            page2.insert_text((50, 50), "Page 2 content", fontsize=10)
            
            doc.save(pdf_path)
            doc.close()
            
            extracted = extract_document(pdf_path)
            
            # Find which blocks are on page 2
            page2_blocks = [b for b in extracted.blocks if 'page_2' in b.location]
            
            # At least some page 2 blocks should inherit SECTION A from page 1
            has_inherited_section = any(b.section == 'SECTION A' for b in page2_blocks)
            self.assertTrue(has_inherited_section or len(page2_blocks) == 0,
                "PDF should carry section across pages, not reset per page")


class TestDefectWordAndMultiplicityConservation(unittest.TestCase):
    """Test exact word order and multiplicity conservation."""
    
    def test_word_order_and_multiplicity_structure_mode(self):
        """Concatenated chunks must preserve exact word order and count."""
        blocks = [
            TextBlock(text='the cat sat on the mat', location='para_0', section='S1'),
            TextBlock(text='the dog ran away', location='para_1', section='S1'),
        ]
        doc = Document(
            text='the cat sat on the mat\n\nthe dog ran away',
            blocks=blocks,
            metadata={}
        )
        
        chunks = make_chunks(doc, max_words=5, mode='structure')
        concatenated = '\n\n'.join(c.text for c in chunks)
        
        # Must preserve EXACT order and multiplicity (not sorted, not deduplicated)
        self.assertEqual(doc.text.split(), concatenated.split(),
            "Word order must be preserved exactly")
    
    def test_word_order_and_multiplicity_words_mode(self):
        """Words mode must also preserve exact order and multiplicity."""
        blocks = [
            TextBlock(text='a b a c a', location='para_0', section='S1'),
        ]
        doc = Document(text='a b a c a', blocks=blocks, metadata={})
        
        chunks = make_chunks(doc, max_words=2, mode='words')
        concatenated = ' '.join(c.text for c in chunks)
        
        # Must preserve 'a' appears 3 times, in order
        self.assertEqual(doc.text, concatenated,
            "Words mode must preserve exact order and multiplicity")


class TestDefectCrossPageSections(unittest.TestCase):
    """Test section tracking across PDF pages."""
    
    def test_section_persists_across_pdf_pages(self):
        """Blocks on page N should inherit section from page N-1 if no new heading."""
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sections.pdf"
            
            doc = fitz.open()
            
            # Page 1: heading + content
            page1 = doc.new_page()
            page1.insert_text((50, 50), "MY SECTION", fontsize=12)
            page1.insert_text((50, 100), "Content 1", fontsize=10)
            
            # Page 2: content (no new heading)
            page2 = doc.new_page()
            page2.insert_text((50, 50), "Content 2", fontsize=10)
            
            # Page 3: new heading
            page3 = doc.new_page()
            page3.insert_text((50, 50), "NEW SECTION", fontsize=12)
            
            doc.save(pdf_path)
            doc.close()
            
            extracted = extract_document(pdf_path)
            
            # Count blocks by section
            my_section_blocks = [b for b in extracted.blocks if b.section == 'MY SECTION']
            new_section_blocks = [b for b in extracted.blocks if b.section == 'NEW SECTION']
            
            # MY SECTION should have blocks from both page 1 and page 2
            self.assertGreater(len(my_section_blocks), 1,
                "Section should persist across pages")


class TestDefectLongParagraphBothModes(unittest.TestCase):
    """Test 500+ word paragraph splits in both modes."""
    
    def test_500word_para_structure_mode(self):
        """500+ word paragraph should split in structure mode."""
        # 501 words in single block
        text_words = ' '.join(['word'] * 501)
        blocks = [TextBlock(text=text_words, location='para_0', section='')]
        doc = Document(text=text_words, blocks=blocks, metadata={})
        
        chunks = make_chunks(doc, max_words=400, mode='structure')
        self.assertGreaterEqual(len(chunks), 2)
    
    def test_500word_para_words_mode(self):
        """500+ word paragraph should split in words mode."""
        text_words = ' '.join(['word'] * 501)
        blocks = [TextBlock(text=text_words, location='para_0', section='')]
        doc = Document(text=text_words, blocks=blocks, metadata={})
        
        chunks = make_chunks(doc, max_words=400, mode='words')
        self.assertGreaterEqual(len(chunks), 2)


class TestDefectDocxTableOrder(unittest.TestCase):
    """Test DOCX table and paragraph ordering."""
    
    def test_docx_para_table_order_preservation(self):
        """DOCX should preserve paragraph/table order."""
        try:
            from docx import Document as DocxDocument
        except ImportError:
            self.skipTest("python-docx not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path = Path(tmpdir) / "ordered.docx"
            
            doc_docx = DocxDocument()
            doc_docx.add_paragraph("Para 1")
            
            table = doc_docx.add_table(rows=1, cols=1)
            table.cell(0, 0).text = "Table Cell"
            
            doc_docx.add_paragraph("Para 2")
            
            doc_docx.save(docx_path)
            
            extracted = extract_document(docx_path)
            
            # Check order: Para 1 before Table before Para 2
            text = extracted.text
            pos_para1 = text.find('Para 1')
            pos_table = text.find('Table Cell')
            pos_para2 = text.find('Para 2')
            
            self.assertGreater(pos_table, pos_para1, "Table should come after Para 1")
            self.assertGreater(pos_para2, pos_table, "Para 2 should come after Table")


if __name__ == '__main__':
    unittest.main(verbosity=2)
