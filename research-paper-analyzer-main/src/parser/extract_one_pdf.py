import fitz  # PyMuPDF

pdf_path = "data/raw/pdfs/1803.01837v1.pdf"   # replace with one actual pdf name from your folder

doc = fitz.open(pdf_path)

print("Number of pages:", len(doc))

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()

    print(f"\n--- PAGE {page_num + 1} ---")
    print(text[:1000])   # print first 1000 characters of each page