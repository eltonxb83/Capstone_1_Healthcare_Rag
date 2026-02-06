#---------------------------------------
# Date          : 20 Dec 25
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : pdfplumber (uses pdfminer.six and pypdfium2 internally)
#Purpose        : This script serve as text extraction tool from previously downloaded clinical pdf guidelines.
#Output         : Clean text files for processing (ie embedding / FIASS)
#Notes          :
#   - Images, diagrams and graphs are not extracted
#   - Output files may need additional cleaning
#----------------------------------------

import pdfplumber
from pathlib import Path

raw_pdf_dir = Path("data/raw")
output_dir = Path("data/extracted")

output_dir.mkdir(parents = True, exist_ok = True)

def extract_text_from_pdf(pdf_path: Path) -> str: #"-> str" means this function is intended to return a string
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_pages.append(text)
    return "\n\n".join(text_pages)

'''
def extract_text_from_pdf(pdf_path: Path) -> str:
user defined function with pdf source pathfilename as arugment
pdf_path: Path -> the function expects a Path object representing the PDF location
-> str -> the function is intended to return a string (the extracted text)

text_pages =[]
assign empty list (to store text from each page) to text_pages
this list will accumulate strings, each representing one page

with pdfplumber.open(pdf_path) as pdf:
open the pdf file for reaching in safe context ("with" ensures it is closed automically)
pdf here is not the raw file, but a pdfplumber PDF object you can work with

for page in pdf.pages:
text = page.extract_text()
for each page in identified pages (pdf.pages), perform text extraction
Returns None if the page has no extractable text (like only images/diagrams)
Returns a string otherwise

"\n\n".join(text_pages)
if there is extracted text, append to text_pages
returns all text in one string, separated by double newlines
'''

def main():
    for pdf_file in raw_pdf_dir.glob("*.pdf"):
        print(f"Extracting: {pdf_file.name}")
        
        extracted_text = extract_text_from_pdf(pdf_file)

        output_file = output_dir/f"{pdf_file.stem}.txt"
        output_file.write_text(extracted_text, encoding ="utf-8")

        print(f"Saved to: {output_file}")
'''
for pdf_file in raw_pdf_fir.glob("*.pdf")
print(f"Extracting: {pdf_file.name}")
run throuhg all detected pdf formatted files
display the current file the working on
.glob("*.pdf") does not recurse into subfolders unless you use **/*.pdf

extracted_text = extract_text_from_pdf(pdf_file)
assign output string from extract_text_from_pdf(pdf_file) to extracted_file
returns a single string containing the entire PDF's text
extracted_text now holds all the page content

output_file = output_dir/f"{pdf_file.stem}.txt"
output_file.write_text(extracted_text, encoding = "utf-8")
create a output file in defined output_dir with the original file name (.stem) in text format
write the output into newly created file in utf-8 text encooding
Note - 
    . Overwrites existing file silently
    . Encoding ensures Unicode support (UTF-8)
    . ALL PDF pages are combined into a single text file, not one per page

Note - UTF-8 Text Encoding
    . Computer store text as bytes , not characters
    . Encoding is the rule that maps characters ->bytes

print(f"Saved to: {output_file}")
indicate progress
'''


if __name__ == "__main__":
    main()

'''
__name__ == "__main__"
this file is being executed directly by Python, not imported by another script
'''