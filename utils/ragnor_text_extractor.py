"""
Text extraction and OCR utilities for Ragnor document extraction API.
"""
import io
import os
import time
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
from dataclasses import dataclass


@dataclass
class TextLine:
    """Represents a single line of extracted text with position and confidence."""
    text: str
    confidence: float
    bbox: List[int]  # [x, y, width, height]
    polygon: Optional[List[List[int]]] = None  # [[x1, y1], [x2, y2], ...]


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text_lines: List[TextLine]
    languages: List[str]
    # [x, y, width, height] of the processed image
    image_bbox: Optional[List[int]] = None


class RagnorTextExtractor:
    """Text extraction using pytesseract OCR."""

    def __init__(self, config=None):
        """Initialize text extractor with configuration."""
        self.config = config or {}

        # Set Tesseract executable path explicitly
        tesseract_path = "/opt/homebrew/bin/tesseract"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"Set Tesseract executable path to: {tesseract_path}")
        else:
            print(
                f"Warning: Tesseract executable not found at {tesseract_path}")

        # Set default pytesseract configuration options
        self.tesseract_config = '--oem 3 --psm 6'

        # Configure OCR parameters based on document types
        # OEM modes:
        # 0 = Original Tesseract only
        # 1 = Neural nets LSTM only
        # 2 = Tesseract + LSTM
        # 3 = Default, based on what is available
        #
        # PSM modes:
        # 0 = Orientation and script detection (OSD) only
        # 1 = Automatic page segmentation with OSD
        # 3 = Fully automatic page segmentation, but no OSD (default)
        # 4 = Assume a single column of text of variable sizes
        # 6 = Assume a single uniform block of text
        # 11 = Sparse text. Find as much text as possible in no particular order
        # 12 = Sparse text with OSD

        # Optimized configurations for different document types
        self.format_configs = {
            'pdf': '--oem 1 --psm 3 -l eng --dpi 300',  # LSTM only, auto page segmentation
            'docx': '--oem 1 --psm 3 -l eng --dpi 300',
            'xlsx': '--oem 1 --psm 3 -l eng -c preserve_interword_spaces=1 --dpi 300',
            'pptx': '--oem 1 --psm 3 -l eng --dpi 300',
            'png': '--oem 1 --psm 3 -l eng --dpi 300',
            'jpg': '--oem 1 --psm 3 -l eng --dpi 300',
            'jpeg': '--oem 1 --psm 3 -l eng --dpi 300',
            'image': '--oem 1 --psm 3 -l eng --dpi 300',
            'default': '--oem 1 --psm 3 -l eng --dpi 300'  # LSTM with auto page segmentation
        }

        print(f"Initialized RagnorTextExtractor with Tesseract OCR configurations")

    def extract_text(self, image: Image.Image, doc_format: str = None) -> OCRResult:
        """Extract text from an image using OCR."""
        try:
            # Print diagnostic info
            print(f"\n======== OCR EXTRACTION BEGIN ========")
            print(f"Starting OCR extraction for format: {doc_format if doc_format else 'unknown'}")
            print(f"Image info: Type={type(image)}, Mode={image.mode if hasattr(image, 'mode') else 'unknown'}, " +
                  f"Size={image.size if hasattr(image, 'size') else 'unknown'}")
            
            # Check Tesseract installation
            tesseract_path = pytesseract.pytesseract.tesseract_cmd
            print(f"Tesseract path is set to: {tesseract_path}")
            if not os.path.exists(tesseract_path):
                print(f"ERROR: Tesseract not found at {tesseract_path}")
                if os.path.exists('/opt/homebrew/bin/tesseract'):
                    print(f"Setting tesseract path to /opt/homebrew/bin/tesseract")
                    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
                    tesseract_path = '/opt/homebrew/bin/tesseract'
            print(f"Checking Tesseract version with direct command...")
            try:
                import subprocess
                version = subprocess.check_output([tesseract_path, '--version'], stderr=subprocess.STDOUT)
                print(f"Tesseract version: {version.decode('utf-8')}")
            except Exception as ver_err:
                print(f"Error checking Tesseract version: {str(ver_err)}")

            # Ensure image is in correct mode for OCR
            if image.mode not in ['RGB', 'L']:
                print(f"Converting image from {image.mode} to RGB mode")
                image = image.convert('RGB')

            # Select the appropriate tesseract configuration based on document format
            if doc_format and doc_format in self.format_configs:
                config = self.format_configs[doc_format]
                print(f"Using format-specific OCR config for {doc_format}: {config}")
            else:
                config = self.format_configs['default']
                print(f"Using default OCR config: {config}")

            # Debug: Save a copy of the image being sent to OCR
            try:
                debug_dir = os.path.join(
                    os.path.dirname(__file__), "debug_images")
                os.makedirs(debug_dir, exist_ok=True)

                # Generate a unique filename
                import hashlib
                img_hash = hashlib.md5(
                    str(time.time()).encode()).hexdigest()[:8]
                debug_path = os.path.join(
                    debug_dir, f"ocr_input_{img_hash}.png")

                # Save the image
                image.save(debug_path)
                print(f"Saved OCR input image to {debug_path}")
            except Exception as debug_err:
                print(f"Warning: Failed to save debug image: {str(debug_err)}")

            # Force OCR to run with Tesseract directly
            print("\n==== RUNNING REAL OCR EXTRACTION ====\n")
            print("Saving image to temporary file for OCR...")
            
            # Save image to a temporary file first
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                temp_path = temp.name
                image.save(temp_path)
                print(f"Saved image to {temp_path}")
            
            # Run Tesseract directly using subprocess for maximum transparency
            print("Running Tesseract directly via subprocess...")
            try:
                import subprocess
                import tempfile
                
                # Create a temporary file for the output
                with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as out_temp:
                    out_path = out_temp.name
                
                # Build the command with all options
                tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
                
                # Split the config string into separate arguments
                config_args = config.split()
                cmd = [tesseract_cmd, temp_path, out_path.replace('.txt', '')]
                cmd.extend(config_args)
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Execute Tesseract
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(f"Tesseract return code: {result.returncode}")
                if result.stderr:
                    print(f"Tesseract stderr: {result.stderr}")
                
                # Read the output file
                with open(out_path, 'r') as f:
                    extracted_text = f.read()
                
                print(f"Successfully extracted {len(extracted_text)} characters of text")
                print(f"First 100 chars: {extracted_text[:100]}...")
                
                # Clean up temporary files
                os.unlink(temp_path)
                os.unlink(out_path)
                
                # Create TextLine objects from the extracted text
                width, height = image.size
                text_lines = []
                
                if extracted_text.strip():
                    # Split text by lines and create a TextLine for each
                    for i, line in enumerate(extracted_text.strip().split('\n')):
                        if line.strip():
                            # Create a simple TextLine with the extracted text
                            text_lines.append(TextLine(
                                text=line.strip(),
                                confidence=90.0,  # Assumed confidence since we don't have actual values
                                # Simple positioning - one line below another
                                bbox=[0, i*20, width, 20]
                            ))
                
                print(f"Created {len(text_lines)} text lines from extraction")
                print("\n==== COMPLETED REAL OCR EXTRACTION ====\n")
            except Exception as ocr_err:
                print(f"ERROR in direct Tesseract execution: {str(ocr_err)}")
                # Log full details and raise the error
                print(f"OCR failed with error: {str(ocr_err)}")
                raise
                
                # For debugging, check if test mode is enabled, which might be causing issues
                is_test_mode = os.environ.get('RAGNOR_TEST_MODE', 'false').lower() == 'true'
                if is_test_mode:
                    print("WARNING: RAGNOR_TEST_MODE is enabled! This forces mock data generation.")
                    print("Set this environment variable to 'false' to use real OCR.")
                
                # Try to troubleshoot common Tesseract issues
                tesseract_path = pytesseract.pytesseract.tesseract_cmd
                print(f"Tesseract path is set to: {tesseract_path}")
                print(f"Checking if Tesseract exists at this path: {os.path.exists(tesseract_path)}")
                
                # Try a basic command to see if Tesseract is working
                try:
                    import subprocess
                    version = subprocess.check_output([tesseract_path, '--version'], stderr=subprocess.STDOUT)
                    print(f"Tesseract version: {version.decode('utf-8')}")
                except Exception as ver_err:
                    print(f"Error checking Tesseract version: {str(ver_err)}")
                
                # Raise the original error to make it visible in logs
                raise ocr_err

            # Get image dimensions for reference
            width, height = image.size
            image_bbox = [0, 0, width, height]

            # Attempt language detection, but don't worry if it fails
            try:
                lang_data = pytesseract.image_to_osd(image, config="--psm 0")
                languages = ['Latin']  # Default to Latin
            except Exception as lang_err:
                print(f"Warning: Could not detect language: {str(lang_err)}")
                languages = ['Latin']  # Fallback to Latin script

            return OCRResult(
                text_lines=text_lines,
                languages=languages,
                image_bbox=image_bbox
            )

        except Exception as e:
            # If OCR fails, return empty result
            print(f"OCR processing failed: {e}")
            return OCRResult(text_lines=[], languages=[], image_bbox=None)

    def extract_from_base64(self, base64_image: str) -> OCRResult:
        """Extract text from a base64-encoded image."""
        # Check if the input includes the data URL prefix
        if ',' in base64_image:
            # Extract the base64 data part
            _, base64_data = base64_image.split(',', 1)
        else:
            base64_data = base64_image

        # Decode base64 to binary
        image_data = base64.b64decode(base64_data)

        # Open as PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Process with OCR
        return self.extract_text(image)

    # Mock text generation has been completely removed to ensure only real OCR is used

    def serialize_result(self, result: OCRResult) -> Dict[str, Any]:
        """Convert OCR result to a serializable dictionary."""
        return {
            "text_lines": [
                {
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": line.bbox,
                    "polygon": line.polygon
                } for line in result.text_lines
            ],
            "languages": result.languages,
            "image_bbox": result.image_bbox
        }
