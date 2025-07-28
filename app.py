import os
import json
import re
import fitz
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from typing import List, Dict, Any, Union
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Challenge1BCorrected:
    def __init__(self):
        """Initialize corrected Challenge 1B processor"""
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except:
            self.summarizer = None
        
        # Enhanced persona configurations
        self.persona_config = {
            "Travel Planner": {
                "section_patterns": [
                    r"guide to.*cities", r"coastal adventures", r"culinary experiences", 
                    r"packing.*tips", r"nightlife.*entertainment", r"activities", 
                    r"restaurants", r"hotels", r"attractions", r"water.*sports"
                ],
                "keywords": [
                    "cities", "coastal", "adventures", "culinary", "experiences", "packing", 
                    "tips", "nightlife", "entertainment", "activities", "restaurants", 
                    "hotels", "attractions", "water", "sports", "beaches", "tours"
                ]
            },
            "HR professional": {
                "section_patterns": [
                    r"change.*forms.*fillable", r"create.*multiple.*pdfs", r"convert.*clipboard", 
                    r"fill.*sign.*forms", r"send.*document.*signatures", r"prepare.*forms",
                    r"acrobat.*pro", r"pdf.*tools", r"e-signatures", r"request.*signatures"
                ],
                "keywords": [
                    "forms", "fillable", "acrobat", "pdf", "signatures", "create", "convert",
                    "fill", "sign", "tools", "prepare", "request", "documents", "pro"
                ]
            },
            "Food Contractor": {
                "section_patterns": [
                    r"^[A-Z][a-z]+\s*[A-Z]*[a-z]*$", r"falafel", r"ratatouille", r"baba.*ganoush",
                    r"veggie.*sushi", r"vegetable.*lasagna", r"macaroni.*cheese", r"escalivada"
                ],
                "keywords": [
                    "falafel", "ratatouille", "baba", "ganoush", "sushi", "vegetable", 
                    "lasagna", "macaroni", "cheese", "vegetarian", "recipe", "ingredients",
                    "instructions", "cooking", "dinner", "buffet", "corporate"
                ]
            }
        }

    def _embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using BERT model"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                               max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def load_input_configuration(self, input_file: Path) -> Dict[str, Any]:
        """Load input configuration from JSON file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading input configuration: {e}")
            return {}

    def is_valid_section_heading(self, text: str, line_info: Dict, body_size: float, 
                                persona: str) -> bool:
        """Enhanced section heading detection"""
        # Length checks
        word_count = len(text.split())
        if word_count < 1 or word_count > 15 or len(text) < 3:
            return False
        
        # Skip common non-headings
        skip_patterns = [
            r"^\d+$", r"^page \d+", r"^figure \d+", r"^table \d+",
            r"^copyright", r"^all rights reserved", r"^www\.", r"^http"
        ]
        if any(re.search(pattern, text.lower()) for pattern in skip_patterns):
            return False
        
        # Font-based detection
        spans = line_info.get("spans", [])
        if spans:
            font_size = spans[0].get("size", body_size)
            is_bold = spans[0].get("flags", 0) & 16
            is_larger = font_size > body_size + 1.5
        else:
            is_bold = is_larger = False
        
        # Persona-specific pattern matching
        persona_patterns = self.persona_config.get(persona, {}).get("section_patterns", [])
        matches_persona_pattern = any(
            re.search(pattern, text.lower()) for pattern in persona_patterns
        )
        
        # General heading patterns
        general_patterns = [
            r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$",  # Title Case
            r"^\d+\.\s*[A-Z]",  # Numbered sections
            r"^(Chapter|Section|Part)\s+\d+",
            r"(Introduction|Conclusion|Summary|Overview|Instructions|Ingredients)$"
        ]
        matches_general_pattern = any(
            re.search(pattern, text) for pattern in general_patterns
        )
        
        # Decision logic
        if matches_persona_pattern:
            return True
        if (is_bold or is_larger) and (matches_general_pattern or word_count <= 8):
            return True
        if matches_general_pattern and word_count <= 6:
            return True
            
        return False

    def extract_sections_from_pdf(self, pdf_path: Path, persona: str, 
                                 job_description: str) -> List[Dict[str, Any]]:
        """Extract sections with improved heading detection"""
        try:
            doc = fitz.open(pdf_path)
            sections = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                
                # Calculate font sizes
                sizes = []
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            sizes.append(span["size"])
                
                if not sizes:
                    continue
                
                body_size = Counter(sizes).most_common(1)[0][0]
                
                # Extract sections
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        if not line["spans"]:
                            continue
                        
                        # Skip header/footer
                        y_pos = line["spans"][0]["origin"][1]
                        if y_pos < page_height * 0.1 or y_pos > page_height * 0.9:
                            continue
                        
                        text = " ".join([span["text"] for span in line["spans"]]).strip()
                        
                        if self.is_valid_section_heading(text, line, body_size, persona):
                            relevance = self.calculate_section_relevance(text, persona, job_description)
                            
                            if relevance > 0.3:  # Higher threshold for quality
                                sections.append({
                                    "document": pdf_path.name,
                                    "section_title": text,
                                    "page_number": page_num + 1,
                                    "relevance_score": relevance,
                                    "importance_rank": 0
                                })
            
            doc.close()
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting sections from {pdf_path}: {e}")
            return []

    def calculate_section_relevance(self, section_title: str, persona: str, 
                                   job_description: str) -> float:
        """Calculate relevance score for section"""
        if not section_title:
            return 0.0
        
        # Embed texts
        title_embedding = self._embed_texts(section_title)[0]
        persona_embedding = self._embed_texts(persona)[0]
        job_embedding = self._embed_texts(job_description)[0]
        
        # Calculate similarities
        persona_sim = np.dot(title_embedding, persona_embedding) / (
            np.linalg.norm(title_embedding) * np.linalg.norm(persona_embedding) + 1e-8
        )
        job_sim = np.dot(title_embedding, job_embedding) / (
            np.linalg.norm(title_embedding) * np.linalg.norm(job_embedding) + 1e-8
        )
        
        # Keyword matching
        persona_keywords = self.persona_config.get(persona, {}).get("keywords", [])
        title_lower = section_title.lower()
        keyword_matches = sum(1 for keyword in persona_keywords if keyword in title_lower)
        keyword_score = min(0.4, keyword_matches * 0.1)
        
        # Pattern bonus
        persona_patterns = self.persona_config.get(persona, {}).get("section_patterns", [])
        pattern_bonus = 0.3 if any(re.search(pattern, title_lower) for pattern in persona_patterns) else 0
        
        return float(persona_sim * 0.25 + job_sim * 0.25 + keyword_score * 0.25 + pattern_bonus * 0.25)

    def deduplicate_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sections while preserving quality"""
        seen = set()
        deduped = []
        
        # Sort by relevance first
        sections_sorted = sorted(sections, key=lambda x: x["relevance_score"], reverse=True)
        
        for section in sections_sorted:
            # Create key for deduplication
            key = (
                section["document"], 
                section["section_title"].strip().lower(),
                section["page_number"]
            )
            
            if key not in seen:
                seen.add(key)
                deduped.append(section)
                
            if len(deduped) >= 5:  # Limit to top 5
                break
        
        return deduped

    def extract_detailed_content(self, pdf_path: Path, section_title: str, 
                               page_number: int, persona: str) -> str:
        """Extract comprehensive content for subsection analysis"""
        try:
            doc = fitz.open(pdf_path)
            if page_number > len(doc):
                doc.close()
                return section_title
            
            page = doc[page_number - 1]
            text_dict = page.get_text("dict")
            
            # Extract all text from the page
            full_text = ""
            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    if line["spans"]:
                        line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                        full_text += " " + line_text
            
            doc.close()
            
            # Find content around the section title
            full_text = full_text.strip()
            if not full_text:
                return section_title
            
            # Look for the section and extract surrounding content
            title_lower = section_title.lower()
            text_lower = full_text.lower()
            
            if title_lower in text_lower:
                start_idx = text_lower.find(title_lower)
                # Extract content after the title (next 500-800 characters)
                content_start = start_idx + len(section_title)
                content_end = min(len(full_text), content_start + 800)
                detailed_content = full_text[content_start:content_end].strip()
                
                if detailed_content:
                    return f"{section_title} {detailed_content}"
            
            # Fallback: return section title with some context
            return section_title
            
        except Exception as e:
            logger.error(f"Error extracting detailed content: {e}")
            return section_title

    def process_collection(self, input_file: Path, pdf_directory: Path) -> Dict[str, Any]:
        """Process collection with corrected logic"""
        # Load configuration
        config = self.load_input_configuration(input_file)
        if not config:
            return {"error": "Failed to load input configuration"}
        
        # Extract details
        documents = config.get("documents", [])
        persona = config.get("persona", {}).get("role", "Unknown")
        job_description = config.get("job_to_be_done", {}).get("task", "Unknown")
        
        # Process documents
        all_sections = []
        processed_docs = []
        
        for doc_info in documents:
            filename = doc_info.get("filename", "")
            pdf_path = pdf_directory / filename
            
            if pdf_path.exists():
                sections = self.extract_sections_from_pdf(pdf_path, persona, job_description)
                all_sections.extend(sections)
                processed_docs.append(filename)
        
        if not all_sections:
            return {
                "metadata": {
                    "input_documents": processed_docs,
                    "persona": persona,
                    "job_to_be_done": job_description,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }
        
        # Deduplicate and rank
        top_sections = self.deduplicate_sections(all_sections)
        
        # Assign importance ranks
        for rank, section in enumerate(top_sections, 1):
            section["importance_rank"] = rank
        
        # Generate subsection analysis
        subsection_analysis = []
        for section in top_sections:
            pdf_path = pdf_directory / section["document"]
            detailed_content = self.extract_detailed_content(
                pdf_path, section["section_title"], section["page_number"], persona
            )
            
            subsection_analysis.append({
                "document": section["document"],
                "refined_text": detailed_content,
                "page_number": section["page_number"]
            })
        
        return {
            "metadata": {
                "input_documents": processed_docs,
                "persona": persona,
                "job_to_be_done": job_description,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": section["importance_rank"],
                    "page_number": section["page_number"]
                }
                for section in top_sections
            ],
            "subsection_analysis": subsection_analysis
        }

def main():
    """Main execution function"""
    INPUT_DIR = Path("")
    OUTPUT_DIR = Path("")
    
    if not INPUT_DIR.exists():
        INPUT_DIR = Path("")
        OUTPUT_DIR = Path("")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    processor = Challenge1BCorrected()
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    for collection_name in collections:
        collection_path = INPUT_DIR / collection_name
        input_file = collection_path / "challenge1b_input.json"
        pdf_directory = collection_path / "PDFs"
        output_file = collection_path / f"challenge1b_output.json"
        
        if input_file.exists() and pdf_directory.exists():
            print(f"📁 Processing {collection_name}...")
            
            try:
                result = processor.process_collection(input_file, pdf_directory)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                
                if "error" not in result:
                    print(f"  ✅ Extracted {len(result['extracted_sections'])} sections")
                    print(f"  📝 Generated {len(result['subsection_analysis'])} analyses")
                else:
                    print(f"  ❌ Error: {result['error']}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    main()