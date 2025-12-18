from typing import List, Dict, Optional
import re

class Chunk:
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap.
    Simple recursive-like splitting on newlines and spaces.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # Try to find a nice break point
        # Prioritize double newline, then newline, then space
        boundary = -1
        
        # Look for double newline within the overlap area
        search_start = max(start, end - chunk_overlap)
        
        double_newline_pos = text.rfind('\n\n', search_start, end)
        if double_newline_pos != -1:
            boundary = double_newline_pos + 2
        else:
            newline_pos = text.rfind('\n', search_start, end)
            if newline_pos != -1:
                boundary = newline_pos + 1
            else:
                space_pos = text.rfind(' ', search_start, end)
                if space_pos != -1:
                    boundary = space_pos + 1
        
        if boundary != -1:
            chunks.append(text[start:boundary])
            start = boundary
        else:
            # Force cut
            chunks.append(text[start:end])
            start = end - chunk_overlap  # Backtrack only if forced cut, or just continue?
            # Actually standard sliding window logic:
            # If we couldn't find a delimiter, we cut at 'end'.
            # To respect overlap, next chunk should start at end - overlap.
            start = max(start, end - chunk_overlap)

    return chunks

def extract_sections(text: str) -> List[Dict]:
    """
    Extract high-level sections based on markdown headers.
    Returns: [{'title': '...', 'content': '...', 'level': 1}, ...]
    """
    lines = text.split('\n')
    sections = []
    current_section = {"title": "Introduction", "content": [], "level": 0}
    
    for line in lines:
        match = re.match(r'^(#+)\s+(.*)', line)
        if match:
            # Save previous section
            if current_section["content"]:
                sections.append({
                    "title": current_section["title"],
                    "content": '\n'.join(current_section["content"]).strip(),
                    "level": current_section["level"]
                })
            
            level = len(match.group(1))
            title = match.group(2).strip()
            current_section = {"title": title, "content": [], "level": level}
        else:
            current_section["content"].append(line)
            
    # Append last
    if current_section["content"]:
         sections.append({
            "title": current_section["title"],
            "content": '\n'.join(current_section["content"]).strip(),
            "level": current_section["level"]
        })
        
    return sections

def create_chunks(text: str, metadata: Dict, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
    """
    Process text into Chunks with metadata.
    Tries to respect sections.
    """
    sections = extract_sections(text)
    all_chunks = []
    
    for section in sections:
        section_text = section['content']
        if not section_text:
            continue
            
        # Add section title context to the text or metadata? 
        # Ideally prepended to text for better retrieval context.
        # But we also store it in metadata.
        
        raw_chunks = split_text(section_text, chunk_size, chunk_overlap)
        
        for i, rc in enumerate(raw_chunks):
            # Prepend section title for context if it's not the main intro
            contextualized_content = rc
            if section['title'] != 'Introduction':
               contextualized_content = f"Section: {section['title']}\n{rc}"
            
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "section_title": section['title'],
                "chunk_id": f"{metadata.get('doc_id', 'unknown')}_{section['title'][:10]}_{i}",
                "original_text": rc # Store original for precise citation if needed, or just use content
            })
            
            all_chunks.append(Chunk(content=contextualized_content, metadata=chunk_meta))
            
    return all_chunks
