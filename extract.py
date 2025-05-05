import os
import json
from typing import Optional, List
import easyocr
import pandas as pd
import cv2
import csv
from litellm import completion
from pdf2image import convert_from_bytes, convert_from_path
import numpy as np
from io import BytesIO

def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    """Convert a PDF file (from a path) to a list of high-quality OpenCV images (numpy arrays)"""
    images = convert_from_path(pdf_path, dpi=600)
    return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]

def image_to_csv(image: np.ndarray) -> str:
    """Process image directly to CSV text without file operations"""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    
    data = []
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        data.append({
            'text': text,
            'left': tl[0],
            'top': tl[1],
            'right': br[0],
            'bottom': br[1]
        })
    
    df = pd.DataFrame(data)
    df['row'] = (df['top'] // 20).astype(int)
    table_data = df.groupby('row')['text'].apply(lambda x: '|'.join(x)).reset_index()
    
    # Convert DataFrame to CSV string
    csv_buffer = BytesIO()
    table_data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue().decode('utf-8')

    with open("output.txt", 'a', encoding='utf-8') as f:
        f.write(csv_content)

    return csv_content

def csv_to_text(csv_data: str) -> str:
    """Convert CSV string to formatted text"""
    lines = []
    reader = csv.reader(csv_data.splitlines())
    for row in reader:
        lines.append(', '.join(row))
    return ' '.join(line.replace('\n', '') for line in lines)


def extract_bank_statement_to_json(text_data):

    FORMAT = """
    {
        "SckBaIfscCo": "IFSC code from statement, fifth character is always a '0' and following characters are numbers",
        "CompanyCode": "Customer ID from statement",
        "Currency": "Currency from statement",
        "BankAcc": "Account number from statement",
        "OpeningBal": "Opening balance amount",
        "ClosingBal": "Closing balance amount",
        "StatementDate": "Start date of statement period in DD.MM.YYYY format",
        "Transactions": [
            {
            "TransactionDate": "Date of transaction in DD.MM.YYYY format, eg: 01.10.2024",
            "ValueDate": "Value date in DD.MM.YYYY format, eg: 01.10.2024, not '01-10-2024'",
            "Narration": "Description of transaction, eg: 'o and m exp_2024093089011188', if garbage values fix them like: 'RTGSISBINR [2024101454684396/202421707120242[7074/STA' to 'RTGS/SBINR 12024101154684396/20242170712024217071/STATE BANK OF INDIA//INB20242170712024217071'",
            "ChequeNo": "Cheque number if available, else empty string",
            "TransAmountDr": "Debit amount if DR, else empty string",
            "TransAmountCr": "Credit amount if CR, else empty string",
            "UtrNumber": "Extract string starting with 'R' followed by numbers until '/' from 'Narration', if present in narration e.g.:'R620240147793200' from 'RTGSIEAJUTIBR62024[0147793[200/SBI', else empty string
            "Balance": "Balance after transaction"
            }
        ]
        }
        """

    prompt = f"""
        Convert the following bank statement text into a structured JSON format. 
        Pay close attention to all details and handle any variations in the input format.

        BANK STATEMENT TEXT:
        {text_data}

        REQUIRED JSON FORMAT:
        {FORMAT}

        INSTRUCTIONS:
        1. Extract all fields accurately from the text, the fields are separated by '|'
        2. Add 'R' infront of UTR numbers in the start, if not present already. UTR must start with 'R' e.g.: 'R6202410147793200' from '"RTGSIEAJUTIBR6202410147793200/SBI' in narration
        3. Handle any variations in the input format
        4. Start extracting transactions after the opening balance line and stop before closing balance line
        5. Fix obvious typos or mistakes in the 'Narration' like: 'PA YMENT' instead of 'PAYMENT' or '/SBI MUTUAL FUNDIHDFC BANKI' instead of '/SBI MUTUAL FUND/HDFC BANK/'
        6. Remove garbage characters from 'Narration' like : 'RTGSISBINR [202410145[4684396' to 'RTGSISBINR2024101454684396'
        7. Ensure amounts are strings with 2 decimal places
        8. If cheque no. is very small like two digits, it is from the digits from Narration field, join them to the end in narration and put empty string
        """

    
    response = completion(
        model = os.getenv("LLM"),
        temperature = 0.1,
        api_key=os.getenv("GEMINI_API_KEY"),
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract the JSON content from the response
    json_output = response.choices[0].message.content
    
    # Clean the output (remove markdown code blocks if present)
    if '```json' in json_output:
        json_output = json_output.split('```json')[1].split('```')[0].strip()
    elif '```' in json_output:
        json_output = json_output.split('```')[1].strip()
    
    def correct_bank_statement_json(generated_json):
        print('Correcting..')
        prompt = f"""
        Correct this bank statement JSON with these rules:
        1. Add 'R' infront of UTR numbers if not already at the start of the string
        2. Remove garbage characters from 'Narration' field like 'RTGSISBINR [202410145[4684396' to 'RTGSISBINR2024101454684396'
        3. Ensure amounts are strings with 2 decimal places
        4. If cheque no. is very small like two digits it is from the digits from Narration field, add them at end in narration and put empty string
        
        JSON to correct:
        {json.dumps(generated_json, indent=2)}
        
        Return ONLY the corrected JSON.
        """

        response = completion(
                model = os.getenv("LLM"),
                temperature = 0.4,
                api_key=os.getenv("GEMINI_API_KEY"),
                messages=[{"role": "user", "content": prompt}]
        )

        try:
            return json.loads(response.choices[0].message.content)
        except:
            return None

    # Parse the JSON to validate it
    try:
        parsed_json = json.loads(json_output)
        # corrected_json = correct_bank_statement_json(parsed_json)
        return parsed_json
    except json.JSONDecodeError as e:
        print("Error parsing JSON output:", e)
        print("Raw output:", json_output)
        return None

if __name__ == '__main__':
    images = pdf_to_images("bank.pdf")
    csv_text=""
    for image in images:
        csv_text+=image_to_csv(image)

    # csv_text = image_to_csv("page_1.png")

    # txt_file = "output-1.txt"

    # with open(csv_file, 'r', newline='', encoding='utf-8') as infile, \
    #     open(txt_file, 'w', encoding='utf-8') as outfile:
    #     reader = csv.reader(infile)
    #     for row in reader:
    #         outfile.write(', '.join(row) + '\n')

    # with open('output.txt', 'r', encoding='utf-8') as f:
    #     text = f.read()
    #     text = text.replace("\n", "")

    result = extract_bank_statement_to_json(csv_text)
    with open("output-1.json", "w") as f:
        json.dump(result, f, indent=4)