"""
Document extraction and chunking module.

Contratos:
- TextBlock(text: str, location: str, section: str = '', heading: bool = False)
- Document(text: str, blocks: list[TextBlock], metadata: dict, warnings: list[str])
- TextChunk(chunk_id: str, text: str, locations: list[str], sections: list[str])

extract_document(path: Path) -> Document
make_chunks(document: Document, max_words: int = 400, max_chars: int = 5000, mode: str = 'structure') -> list[TextChunk]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import re


@dataclass
class TextBlock:
    """Represents a block of text extracted from a document."""
    text: str
    location: str
    section: str = ''
    heading: bool = False


@dataclass
class Document:
    """Represents an extracted document with all its blocks and metadata."""
    text: str
    blocks: List[TextBlock]
    metadata: Dict
    warnings: List[str] = field(default_factory=list)


@dataclass
class TextChunk:
    """Represents a chunk of text split from a document."""
    chunk_id: str
    text: str
    locations: List[str]
    sections: List[str]


def extract_document(path: Path) -> Document:
    """
    Extract text from a document file.
    
    Supported formats:
    - TXT (.txt, .text)
    - Markdown (.md, .markdown)
    - PDF (.pdf) with text layer
    - Word (.docx)
    
    Args:
        path: Path to the document file
        
    Returns:
        Document object with extracted text, blocks, metadata, and warnings
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is not supported or file is unsupported binary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix in {'.txt', '.text'}:
        return _extract_txt(path)
    elif suffix in {'.md', '.markdown'}:
        return _extract_markdown(path)
    elif suffix == '.pdf':
        return _extract_pdf(path)
    elif suffix == '.docx':
        return _extract_docx(path)
    elif suffix == '.doc':
        raise ValueError(
            f"Old DOC format not supported. Please convert to DOCX using Word or LibreOffice."
        )
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _extract_txt(path: Path) -> Document:
    """Extract text from a TXT file."""
    try:
        text = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding='utf-16')
        except UnicodeDecodeError:
            raise ValueError(
                f"Cannot decode {path.name} with UTF-8, UTF-8-BOM, or UTF-16. "
                f"Please save the file as UTF-8 (with or without BOM) or UTF-16."
            )
    
    blocks = []
    warnings = []
    
    # Split into paragraphs (consecutive non-empty lines)
    current_section = ''
    para_idx = 0
    
    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        
        # Simple heuristic: lines that are all uppercase or start with numbers might be headings
        lines = para.split('\n')
        first_line = lines[0]
        is_heading = (
            len(first_line) < 80 and (
                first_line.isupper() or 
                re.match(r'^\d+\.?\s+', first_line)
            )
        )
        
        if is_heading:
            current_section = first_line
        
        location = f"para_{para_idx}"
        block = TextBlock(
            text=para,
            location=location,
            section=current_section,
            heading=is_heading
        )
        blocks.append(block)
        para_idx += 1
    
    metadata = {
        'source': str(path),
        'format': 'txt',
        'total_blocks': len(blocks),
    }
    
    full_text = '\n\n'.join(b.text for b in blocks)
    return Document(text=full_text, blocks=blocks, metadata=metadata, warnings=warnings)


def _extract_markdown(path: Path) -> Document:
    """Extract text from a Markdown file."""
    try:
        text = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding='utf-16')
        except UnicodeDecodeError:
            raise ValueError(
                f"Cannot decode {path.name} with UTF-8, UTF-8-BOM, or UTF-16. "
                f"Please save the file as UTF-8 (with or without BOM) or UTF-16."
            )
    
    blocks = []
    warnings = []
    
    # Parse markdown with ATX and setext headings
    lines = text.split('\n')
    current_section = ''
    current_block = []
    block_idx = 0
    in_fenced_code = False
    fence_marker = None
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for fenced code blocks (``` or ~~~)
        fence_match = re.match(r'^(```|~~~)', line)
        if fence_match:
            if not in_fenced_code:
                in_fenced_code = True
                fence_marker = fence_match.group(1)
            elif fence_marker == fence_match.group(1):
                in_fenced_code = False
                fence_marker = None
            current_block.append(line)
            i += 1
            continue
        
        # Skip content inside fenced code blocks (don't parse as headings)
        if in_fenced_code:
            current_block.append(line)
            i += 1
            continue
        
        # Check for ATX headings (# ## ### etc.)
        atx_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if atx_match:
            # Save previous block if any
            if current_block:
                para_text = '\n'.join(current_block).strip()
                if para_text:
                    location = f"para_{block_idx}"
                    block = TextBlock(
                        text=para_text,
                        location=location,
                        section=current_section,
                        heading=False
                    )
                    blocks.append(block)
                    block_idx += 1
                current_block = []
            
            # Current line is a heading
            current_section = atx_match.group(2)
            location = f"para_{block_idx}"
            block = TextBlock(
                text=line,
                location=location,
                section='',
                heading=True
            )
            blocks.append(block)
            block_idx += 1
            i += 1
            
        # Check for setext headings (underline with === or ---)
        elif i + 1 < len(lines):
            next_line = lines[i + 1]
            is_setext_h1 = (
                line.strip() and
                re.match(r'^=+$', next_line.strip())
            )
            is_setext_h2 = (
                line.strip() and
                re.match(r'^-+$', next_line.strip())
            )
            
            if is_setext_h1 or is_setext_h2:
                # Save previous non-heading block if any
                if current_block:
                    para_text = '\n'.join(current_block).strip()
                    if para_text:
                        location = f"para_{block_idx}"
                        block = TextBlock(
                            text=para_text,
                            location=location,
                            section=current_section,
                            heading=False
                        )
                        blocks.append(block)
                        block_idx += 1
                    current_block = []
                
                # Current line is setext heading, next line is underline
                current_section = line.strip()
                location = f"para_{block_idx}"
                block = TextBlock(
                    text=line.strip(),
                    location=location,
                    section='',
                    heading=True
                )
                blocks.append(block)
                block_idx += 1
                
                # Skip underline
                i += 2
            elif line.strip():
                current_block.append(line)
                i += 1
            else:
                # Empty line - potential paragraph separator
                if current_block:
                    para_text = '\n'.join(current_block).strip()
                    if para_text:
                        location = f"para_{block_idx}"
                        block = TextBlock(
                            text=para_text,
                            location=location,
                            section=current_section,
                            heading=False
                        )
                        blocks.append(block)
                        block_idx += 1
                    current_block = []
                i += 1
        else:
            if line.strip():
                current_block.append(line)
            else:
                # Empty line at end
                if current_block:
                    para_text = '\n'.join(current_block).strip()
                    if para_text:
                        location = f"para_{block_idx}"
                        block = TextBlock(
                            text=para_text,
                            location=location,
                            section=current_section,
                            heading=False
                        )
                        blocks.append(block)
                        block_idx += 1
                    current_block = []
            i += 1
    
    # Save last block
    if current_block:
        para_text = '\n'.join(current_block).strip()
        if para_text:
            location = f"para_{block_idx}"
            block = TextBlock(
                text=para_text,
                location=location,
                section=current_section,
                heading=False
            )
            blocks.append(block)
    
    metadata = {
        'source': str(path),
        'format': 'markdown',
        'total_blocks': len(blocks),
    }
    
    full_text = '\n\n'.join(b.text for b in blocks)
    return Document(text=full_text, blocks=blocks, metadata=metadata, warnings=warnings)


def _extract_pdf(path: Path) -> Document:
    """Extract text from a PDF file.
    
    Maintains section tracking across pages (doesn't reset per-page).
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF (fitz) required for PDF parsing. Install with: pip install PyMuPDF")
    
    doc = fitz.open(path)
    try:
        total_pages = len(doc)
        blocks = []
        warnings = []
        current_section = ''  # Carry section across pages
        block_idx = 0
        
        # Process pages in order
        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text()
            
            if not page_text.strip():
                warnings.append(f"Page {page_num + 1} has no text layer (may require OCR)")
                continue
            
            # Check for tables or complex layouts
            blocks_on_page = page.get_text("blocks")
            if any(b for b in blocks_on_page if b[6] == 1):  # Table blocks (type 1 = table)
                warnings.append(f"Page {page_num + 1} contains tables/complex layouts (review extraction)")
            
            # Sort blocks by vertical position (top to bottom), then left to right
            sorted_blocks = sorted(blocks_on_page, key=lambda b: (b[1], b[0]))
            
            for block_data in sorted_blocks:
                if block_data[6] == 1:  # Skip tables for now
                    continue
                
                block_text = block_data[4].strip()
                if not block_text:
                    continue
                
                # Detect headings conservatively:
                # - All uppercase with reasonable length
                # - Starts with digit(s) followed by dot/space
                # - Very short compared to normal paragraphs
                lines = block_text.split('\n')
                first_line = lines[0]
                is_heading = (
                    (len(first_line) < 100 and first_line.isupper() and len(first_line) > 3) or
                    re.match(r'^\d+\.?\s+', first_line)
                )
                
                if is_heading:
                    current_section = first_line
                
                location = f"page_{page_num + 1}_block_{block_idx}"
                block = TextBlock(
                    text=block_text,
                    location=location,
                    section=current_section,
                    heading=is_heading
                )
                blocks.append(block)
                block_idx += 1
        
        metadata = {
            'source': str(path),
            'format': 'pdf',
            'total_pages': total_pages,
            'total_blocks': len(blocks),
        }
        
        full_text = '\n\n'.join(b.text for b in blocks)
        return Document(text=full_text, blocks=blocks, metadata=metadata, warnings=warnings)
    finally:
        doc.close()


def _extract_docx(path: Path) -> Document:
    """Extract text from a DOCX file."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise ImportError("python-docx required for DOCX parsing. Install with: pip install python-docx")
    
    try:
        doc = DocxDocument(path)
    except Exception as e:
        raise ValueError(
            f"Cannot open {path.name} as DOCX. File may be corrupted, "
            f"in old .doc format, or scanned/empty. Error: {e}"
        )
    
    blocks = []
    warnings = []
    current_section = ''
    block_idx = 0
    
    from docx.table import Table
    
    # Iterate through document body elements in order
    for elem in doc.element.body:
        # Handle paragraphs (p tag)
        if elem.tag.endswith('}p'):
            para_text = elem.text.strip()
            if not para_text:
                continue
            
            # Reconstruct paragraph from runs to preserve formatting
            para_obj = None
            for p in doc.paragraphs:
                if p._element == elem:
                    para_obj = p
                    break
            
            if para_obj:
                para_text = para_obj.text.strip()
                if not para_text:
                    continue
                
                # Check if paragraph has heading style
                is_heading = para_obj.style.name.startswith('Heading')
                
                if is_heading:
                    current_section = para_text
                
                location = f"para_{block_idx}"
                block = TextBlock(
                    text=para_text,
                    location=location,
                    section=current_section,
                    heading=is_heading
                )
                blocks.append(block)
                block_idx += 1
        
        # Handle tables (tbl tag)
        elif elem.tag.endswith('}tbl'):
            table = Table(elem, doc)
            for row_idx, row in enumerate(table.rows):
                row_cells = []
                for cell in row.cells:
                    cell_text = ' '.join(p.text for p in cell.paragraphs).strip()
                    if cell_text:
                        row_cells.append(cell_text)
                
                if row_cells:
                    table_row_text = ' | '.join(row_cells)
                    location = f"table_row_{block_idx}"
                    block = TextBlock(
                        text=table_row_text,
                        location=location,
                        section=current_section,
                        heading=False
                    )
                    blocks.append(block)
                    block_idx += 1
    
    if not blocks:
        raise ValueError(f"{path.name} appears to be empty or scanned PDF saved as DOCX with no text layer")
    
    metadata = {
        'source': str(path),
        'format': 'docx',
        'total_blocks': len(blocks),
    }
    
    full_text = '\n\n'.join(b.text for b in blocks)
    return Document(text=full_text, blocks=blocks, metadata=metadata, warnings=warnings)


def make_chunks(
    document: Document,
    max_words: int = 400,
    max_chars: int = 5000,
    mode: str = 'structure'
) -> List[TextChunk]:
    """
    Split a document into chunks.
    
    Args:
        document: Document to split
        max_words: Maximum words per chunk (mode='words')
        max_chars: Hard limit on characters per chunk
        mode: 'structure' (respects section changes) or 'words' (ignores sections)
        
    Returns:
        List of TextChunk objects with stable IDs and no overlaps
        
    Raises:
        ValueError: If a single token exceeds max_chars or invalid parameters
    """
    if mode not in {'structure', 'words'}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'structure' or 'words'")
    
    if not isinstance(max_words, int) or max_words <= 0:
        raise ValueError(f"max_words must be positive integer, got {max_words}")
    
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError(f"max_chars must be positive integer, got {max_chars}")
    
    chunks = []
    chunk_counter = 1
    
    if mode == 'structure':
        chunks = _chunk_by_structure(document, max_words, max_chars, chunk_counter)
    elif mode == 'words':
        chunks = _chunk_by_words(document, max_words, max_chars, chunk_counter)
    
    return chunks


def _chunk_by_structure(
    document: Document,
    max_words: int,
    max_chars: int,
    start_counter: int
) -> List[TextChunk]:
    """Chunk respecting section boundaries, enforcing both word and char limits.
    
    Splits blocks that exceed max_words across multiple chunks.
    Headings always start new chunks.
    """
    chunks = []
    chunk_id_counter = start_counter
    
    current_chunk_text = []
    current_chunk_locations = []
    current_chunk_sections = set()
    current_chunk_chars = 0
    current_chunk_words = 0
    
    def save_current_chunk():
        """Save and reset current chunk if it has content."""
        nonlocal chunk_id_counter, current_chunk_text, current_chunk_locations, current_chunk_sections, current_chunk_chars, current_chunk_words
        
        if current_chunk_text:
            chunk_text = '\n\n'.join(current_chunk_text)
            chunk_sections = sorted(current_chunk_sections - {''})
            chunk = TextChunk(
                chunk_id=f'c{chunk_id_counter:04d}',
                text=chunk_text,
                locations=current_chunk_locations,
                sections=chunk_sections
            )
            chunks.append(chunk)
            chunk_id_counter += 1
        
        current_chunk_text = []
        current_chunk_locations = []
        current_chunk_sections = set()
        current_chunk_chars = 0
        current_chunk_words = 0
    
    for block in document.blocks:
        # If block alone exceeds max_chars, raise error
        if len(block.text) > max_chars:
            raise ValueError(
                f"Single token exceeds max_chars ({len(block.text)} > {max_chars}): {block.text[:100]}..."
            )
        
        block_word_count = len(block.text.split())
        
        # Headings always start new chunks
        if block.heading and current_chunk_text:
            save_current_chunk()
        
        # If block exceeds max_words, split it
        if block_word_count > max_words:
            # Save current chunk if any
            save_current_chunk()
            
            # Split block into word chunks
            words = block.text.split()
            word_group = []
            for word in words:
                word_group.append(word)
                if len(word_group) >= max_words:
                    # Create chunk from word group
                    chunk_text = ' '.join(word_group)
                    if len(chunk_text) <= max_chars:
                        chunk = TextChunk(
                            chunk_id=f'c{chunk_id_counter:04d}',
                            text=chunk_text,
                            locations=[block.location],
                            sections=[block.section] if block.section else []
                        )
                        chunks.append(chunk)
                        chunk_id_counter += 1
                        word_group = []
                    else:
                        raise ValueError(
                            f"Single token exceeds max_chars when splitting block: {block.location}"
                        )
            
            # Handle remaining words
            if word_group:
                chunk_text = ' '.join(word_group)
                if len(chunk_text) <= max_chars:
                    chunk = TextChunk(
                        chunk_id=f'c{chunk_id_counter:04d}',
                        text=chunk_text,
                        locations=[block.location],
                        sections=[block.section] if block.section else []
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
        else:
            # Block is within max_words, check if it fits in current chunk
            would_exceed_chars = (current_chunk_chars + len(block.text) + 2 > max_chars)  # +2 for \n\n
            would_exceed_words = (current_chunk_words + block_word_count > max_words) and current_chunk_text
            section_change = (
                block.section and 
                current_chunk_sections and 
                block.section not in current_chunk_sections and
                not block.heading
            )
            
            # Decide if we should start a new chunk
            if would_exceed_chars or would_exceed_words or (section_change and current_chunk_text):
                save_current_chunk()
            
            # Add block to current chunk
            current_chunk_text.append(block.text)
            current_chunk_locations.append(block.location)
            if block.section:
                current_chunk_sections.add(block.section)
            current_chunk_chars += len(block.text) + 2
            current_chunk_words += block_word_count
    
    # Save last chunk
    save_current_chunk()
    
    return chunks


def _chunk_by_words(
    document: Document,
    max_words: int,
    max_chars: int,
    start_counter: int
) -> List[TextChunk]:
    """Chunk by word count, ignoring section boundaries.
    
    Properly tracks which blocks contributed words to each chunk.
    """
    chunks = []
    chunk_id_counter = start_counter
    
    # Build a map of each word to its contributing block
    word_to_blocks = {}  # word_index -> (block_location, block_section)
    word_index = 0
    
    for block in document.blocks:
        block_words = block.text.split()
        for _ in block_words:
            word_to_blocks[word_index] = (block.location, block.section)
            word_index += 1
    
    # Flatten all text
    words = document.text.split()
    
    current_chunk_words = []
    current_chunk_chars = 0
    current_chunk_locations = set()
    current_chunk_sections = set()
    start_word_idx = 0
    
    for word_idx, word in enumerate(words):
        word_with_space = word + ' '
        
        if len(word) > max_chars:
            raise ValueError(
                f"Single token exceeds max_chars ({len(word)} > {max_chars}): {word}"
            )
        
        # Check if adding this word would exceed limits
        potential_chunk_chars = current_chunk_chars + len(word_with_space)
        would_exceed_chars = potential_chunk_chars > max_chars
        would_exceed_words = len(current_chunk_words) >= max_words and current_chunk_words
        
        if would_exceed_chars or would_exceed_words:
            # Save current chunk
            if current_chunk_words:
                chunk_text = ' '.join(current_chunk_words)
                chunk_sections = sorted(current_chunk_sections - {''})
                chunk_locations = sorted(current_chunk_locations)
                
                chunk = TextChunk(
                    chunk_id=f'c{chunk_id_counter:04d}',
                    text=chunk_text,
                    locations=chunk_locations,
                    sections=chunk_sections
                )
                chunks.append(chunk)
                chunk_id_counter += 1
            
            current_chunk_words = [word]
            current_chunk_chars = len(word)
            current_chunk_locations = set()
            current_chunk_sections = set()
            start_word_idx = word_idx
        else:
            current_chunk_words.append(word)
            current_chunk_chars += len(word_with_space)
        
        # Track location and section for this word
        if word_idx in word_to_blocks:
            loc, sec = word_to_blocks[word_idx]
            current_chunk_locations.add(loc)
            if sec:
                current_chunk_sections.add(sec)
    
    # Save last chunk
    if current_chunk_words:
        chunk_text = ' '.join(current_chunk_words)
        chunk_sections = sorted(current_chunk_sections - {''})
        chunk_locations = sorted(current_chunk_locations)
        
        chunk = TextChunk(
            chunk_id=f'c{chunk_id_counter:04d}',
            text=chunk_text,
            locations=chunk_locations,
            sections=chunk_sections
        )
        chunks.append(chunk)
    
    return chunks
