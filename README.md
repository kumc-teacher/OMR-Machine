# MC Marking

Desktop application for recognizing multiple-choice answer sheets, verifying them against an answer key, and producing per-page scoring summaries.

## Features
- Load answer key from image or PDF files and auto-detect the answer table.
- Manual rectangle selection tool when automatic table detection fails.
- OCR-powered extraction of question numbers and answers using Tesseract.
- Batch processing of student submissions (images or PDFs) with per-page scoring breakdowns.
- PyQt6 desktop interface with live preview and results tables.

## Getting Started
1. Create a virtual environment (already configured in `.venv` if you used the provided tooling).
2. Install the dependencies:
	```powershell
	C:/Users/JCMKEC/Desktop/python/MC_marking/.venv/Scripts/python.exe -m pip install -r requirements.txt
	```
3. Launch the application:
	```powershell
	C:/Users/JCMKEC/Desktop/python/MC_marking/.venv/Scripts/python.exe main.py
	```

## Usage Tips
- Ensure Tesseract OCR is installed on your system and available on the PATH.
- Poppler is required for PDF support. Install it (e.g., from https://github.com/oschwartz10612/poppler-windows/releases/) and add the `bin` folder to your PATH.
- Use **Set Poppler Path** to point the app at Poppler's `bin` folder if it is not on PATH, and **Set Tesseract Path** to choose the `tesseract.exe` location when needed.
- When automatic detection fails, use the preview pane to draw a rectangle around the table and click **Use Selection**.
- The results table lists every processed page with counts of correct, incorrect, and unanswered responses.

## Known Limitations
- Table detection relies on reasonably high-contrast scans; low-quality images may require manual assistance.
- OCR accuracy depends on the clarity of printed text and the language packs installed in Tesseract.

## License
This project currently does not specify a license.
