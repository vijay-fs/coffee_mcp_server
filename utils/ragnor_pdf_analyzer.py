"""
PDF analyzer utilities for Ragnor document extraction API.
Provides functions to determine if PDF pages are text-based or scanned.
"""
import io
import pdfplumber
from typing import Dict, List, Tuple


def is_scanned_page(page_content: Dict) -> bool:
    """
    Determine if a page is likely a scanned image rather than text-based.
    
    Args:
        page_content: A pdfplumber page extracted content
        
    Returns:
        bool: True if the page is likely scanned, False if likely text-based
    """
    # Check if there's text content
    has_text = bool(page_content.get('chars', []))
    
    # If page has no text at all, it's likely a scanned page
    if not has_text:
        return True
    
    # Calculate text area coverage - text-based PDFs typically have many character objects
    # with clear positioning, while OCR'd PDFs have fewer positioned text elements
    text_count = len(page_content.get('chars', []))
    
    # Threshold for minimum number of characters to consider a page text-based
    # This is a heuristic that may need adjustment
    return text_count < 10  # If less than 10 characters, likely scanned


def extract_text_with_pdfplumber(page) -> str:
    """
    Extract text from a text-based PDF page using pdfplumber.
    
    Args:
        page: A pdfplumber page object
        
    Returns:
        str: The extracted text from the page
    """
    return page.extract_text() or ""


def analyze_pdf(file_content: bytes) -> List[bool]:
    """
    Analyze a PDF to determine which pages are scanned vs text-based.
    
    Args:
        file_content: The PDF file content as bytes
        
    Returns:
        List[bool]: List of boolean values where True indicates a scanned page
                   and False indicates a text-based page
    """
    pdf_io = io.BytesIO(file_content)
    
    try:
        with pdfplumber.open(pdf_io) as pdf:
            # Initialize results list
            is_scanned_list = []
            
            # Check each page
            for page in pdf.pages:
                page_content = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                    horizontal_ltr=True,
                    vertical_ttb=True,
                    extra_attrs=["size", "fontname"]
                )
                
                # Determine if page is scanned based on extracted content
                is_scanned = is_scanned_page({"chars": page_content})
                is_scanned_list.append(is_scanned)
                
            return is_scanned_list
    except Exception as e:
        # If analysis fails, assume all pages are scanned to fall back to OCR
        print(f"Error analyzing PDF: {str(e)}")
        # Try to get page count at least
        try:
            with pdfplumber.open(pdf_io) as pdf:
                return [True] * len(pdf.pages)
        except:
            return [True]  # Assume at least one scanned page


def extract_pdf_text(file_content: bytes) -> List[Tuple[str, bool]]:
    """
    Extract text from a PDF using pdfplumber for text-based pages.
    
    Args:
        file_content: The PDF file content as bytes
        
    Returns:
        List[Tuple[str, bool]]: List of tuples where each tuple contains:
                               - extracted text for the page
                               - boolean indicating if the page was scanned
    """
    pdf_io = io.BytesIO(file_content)
    results = []
    
    try:
        with pdfplumber.open(pdf_io) as pdf:
            for page in pdf.pages:
                try:
                    page_content = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=False
                    )
                    
                    is_scanned = is_scanned_page({"chars": page_content})
                    
                    if is_scanned:
                        # For scanned pages, return empty text and let OCR handle it
                        results.append(("", True))
                    else:
                        # For text-based pages, extract text with pdfplumber
                        text = extract_text_with_pdfplumber(page)
                        results.append((text, False))
                except Exception as e:
                    print(f"Error processing page: {str(e)}")
                    # If extraction fails, assume it's a scanned page
                    results.append(("", True))
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        # If extraction completely fails, return empty result
        results = [("", True)]
        
    return results