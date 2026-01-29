import argparse
import PyPDF2
import pdfplumber
import json


def parse_args():
    parser = argparse.ArgumentParser(description='PDF file processor for MCM problem')
    parser.add_argument('--input', type=str, required=True, help='Input PDF file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path for extracted text')
    return parser.parse_args()


def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本"""
    text = ""
    try:
        # 使用pdfplumber提取文本
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        # 备用：使用PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e2:
            print(f"Error extracting text: {e2}")
    return text


def main():
    args = parse_args()
    
    # 提取文本
    text = extract_text_from_pdf(args.input)
    
    # 保存提取的文本
    result = {
        'extracted_text': text,
        'file_path': args.input
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"PDF text extracted successfully and saved to {args.output}")


if __name__ == "__main__":
    main()