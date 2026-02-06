#---------------------------------------
# Date          : 22 Dec 25
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : regex
#Purpose        : Removing excessive / leading / trailing whitespaces and footer with objective to improve chunking efficiency
#Output         : Cleaned text for chunking, embedding, retrieval
#Notes          :
#   - Extracted text underwent initial curating (removal of preceding pages such as TOCs, revision history etc, usage instructions, Overview, intended audience details)
#   - Script aims at removing broken line breaks, extra whitespace, repeated headers/footers, page numbers mid-sentence, unicode oddities
#----------------------------------------
from pathlib import Path
import re

raw_text_dir = Path("data/extracted")
clean_text_dir = Path("data/cleaned")

clean_text_dir.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    #remove excessive whitespace by performing regex substitution
    #text argument -> input string variable on which the operation is performed, and the result
    #is then assigned back to the text varaible, effevtively modifying the original string in place 
    text = re.sub(r'\s+',' ', text)

    '''
    Caution - use ' ' instead of '' for substitution.
    Using '' will remove all spaces in text, which will break sentences and chunking.
    '''

    #Remove page numbers
    footer = re.compile(r'NICE.*?Page\s+\d+.*?rights.*?\d+', re.DOTALL | re.IGNORECASE)
    text = footer.sub('', text)

    '''
    Normally in Python regex, the .(dot) matches any character except newline(\n)
    re.DOTALL changes this behavior so that . matches every character including newlines

    The | here is bitwise OR , not logical OR.
    It combines flags, so the regex uses both behavoirs simultaneously
    When use bitwise OR, Python comines the bits of both flags:
    
    0b100 (DOTALL)
    | 0b001 (IGNORECASE)
    = 0b101
    
    The regex engine sees both bits set -> it applies both DOTALL and IGNORECASE
    '''

    #Strip leading/trailing spaces
    return text.strip()

def main():
    for txt_file in raw_text_dir.glob("*.txt"):
        print(f"Cleaning: {txt_file.name}")

        raw_text = txt_file.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)

        output_file = clean_text_dir/txt_file.name
        output_file.write_text(cleaned_text, encoding ="utf-8")

        print(f"Saved cleaed file to : {output_file}")

if __name__ == "__main__":
    main()
