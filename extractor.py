from pdf2image import convert_from_bytes
import cv2
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import json
import tempfile
from PIL import Image, ImageEnhance
from typing import List, Dict, Any
from litellm import completion
from io import BytesIO
import base64
import os
from PIL import Image


def preprocess_image(img):
    """Enhance image for better OCR results"""
    img = img.convert('L')  # Convert to grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)  # Boost contrast
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_pretext(pil_image):
    """Extract text from the header portion of the statement (memory-only version)"""
    # Crop top 30% of the image (where header usually is)
    pil_image = Image.fromarray(pil_image)
    width, height = pil_image.size
    header_img = pil_image.crop((0, 0, width, int(height * 0.3)))
    
    # Convert to numpy array (OpenCV format)
    header_cv = np.array(header_img)
    header_cv = cv2.cvtColor(header_cv, cv2.COLOR_RGB2BGR)
    
    # Run OCR on the numpy array
    result = ocr.ocr(header_cv, cls=True)
    
    # Extract all text
    texts = [line[1][0] for line in result[0]] if result else []
    return '\n'.join(texts)

def extract_tables(img):
    """Detect and extract tabular regions"""
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > img.shape[1]*0.5 and h > img.shape[0]*0.1:  # Filter for table-like regions
            tables.append(img[y:y+h, x:x+w])
    return tables

def ocr_to_table_data(result):
    """Convert OCR result to structured table data by detecting complete cells"""
    if not result:
        return []

    # Collect all bounding boxes and text
    boxes = [item[0] for item in result]
    texts = [item[1][0] for item in result]

    # Calculate average character height for row grouping
    char_heights = [box[2][1] - box[0][1] for box in boxes]
    avg_char_height = sum(char_heights) / len(char_heights) if char_heights else 20

    # Cluster rows based on vertical positions with tighter grouping
    y_centers = [(box[0][1] + box[2][1])/2 for box in boxes]
    row_groups = {}
    for i, y in enumerate(y_centers):
        # Group within 1.5x character height
        row_key = round(y / (avg_char_height * 1.5)) * (avg_char_height * 1.5)
        if row_key not in row_groups:
            row_groups[row_key] = []
        row_groups[row_key].append(i)

    # Cluster columns based on horizontal positions
    x_centers = [(box[0][0] + box[2][0])/2 for box in boxes]
    col_thresholds = sorted(list(set([round(x/50)*50 for x in x_centers])))  # 50px column groups

    # Build the table structure
    table = []
    for row_key in sorted(row_groups.keys()):
        row_indices = row_groups[row_key]
        
        # Get all elements in this row
        row_elements = []
        for idx in row_indices:
            x1, y1, x2, y2 = boxes[idx][0][0], boxes[idx][0][1], boxes[idx][2][0], boxes[idx][2][1]
            row_elements.append({
                'x1': x1,
                'x2': x2,
                'text': texts[idx],
                'idx': idx
            })

        # Sort elements left to right
        row_elements.sort(key=lambda x: x['x1'])

        # Group into columns
        current_row = {}
        for elem in row_elements:
            # Find which column this belongs to
            col_pos = 0
            center_x = (elem['x1'] + elem['x2']) / 2
            for i, threshold in enumerate(col_thresholds):
                if center_x < threshold:
                    col_pos = i
                    break
            else:
                col_pos = len(col_thresholds)

            # If column already has text, append with space
            if col_pos in current_row:
                current_row[col_pos] += " " + elem['text']
            else:
                current_row[col_pos] = elem['text']

        # Convert to list format (fill empty columns)
        max_col = max(current_row.keys()) if current_row else 0
        ordered_row = []
        for col in range(max_col + 1):
            ordered_row.append(current_row.get(col, ""))
        
        table.append(ordered_row)

    return table


def image_to_json(img, output_json="table_data.json"):
    """Extract table data and save as JSON"""
    all_tables = []
    tables = extract_tables(img)
    
    for table_img in tables:
        result = ocr.ocr(table_img, cls=True)
        table_data = ocr_to_table_data(result[0])
        all_tables.append(table_data)
    
    ## Save to JSON
    # with open(output_json, 'w') as f:
    #     json.dump(all_tables, f, indent=2)
    # print(f"Table data saved to {output_json}")

    return all_tables

def extract_bank_statement_data(text_statement, transactions_json):
    """
    Extracts structured data from bank statement text and transactions JSON using Gemini via LiteLLM.
    
    Args:
        text_statement (str): The text content of the bank statement
        transactions_json (dict): The JSON containing transaction data
        
    Returns:
        dict: Structured data in the required format
    """
    # Prepare the prompt
    prompt = f"""
    Analyze the following bank statement header text and transaction data to extract structured information.
    
    BANK STATEMENT HEADER TEXT:
    {text_statement}
    
    TRANSACTIONS DATA:
    {json.dumps(transactions_json, indent=2)}
    
    Extract the following information in JSON format:
    {{
        "SckBaIfscCo": "IFSC code from statement",
        "CompanyCode": "Customer ID from statement",
        "Currency": "Currency from statement (e.g., INR)",
        "BankAcc": "Account number from statement",
        "OpeningBal": "Opening balance amount",
        "ClosingBal": "Closing balance amount",
        "StatementDate": "Start date of statement period in DD.MM.YYYY format",
        "Transactions": [
            {{
                "TransactionDate": "Date in DD.MM.YYYY",
                "ValueDate": "Date in DD.MM.YYYY",
                "Narration": "Narration of transaction e.g.: 'statutory payment_2024101889005634' , 'RTGS/HDFCR52024102856122601/SBI MUTUAL FUND/HDFC', etc.",
                "ChequeNo": "Cheque number or empty string",
                "TransAmountDr": "Debit amount or empty string",
                "TransAmountCr": "Credit amount or empty string",
                "UtrNumber": "UTR from narration or empty string, always starts with 'R', for e.g.:'R12024101154684396' from '/SBINR12024101154684396/' ",
                "Balance": "Balance after transaction"
            }}
        ]
    }}
    
    Instructions:
    1. UTR must start with 'R' followed by numbers until '/' e.g.:'R62024101477931200' from '/UTIBR62024101477931200'.
    2. Format all dates as DD.MM.YYYY.
    3. All amounts have two decimal places.
    4. Sometimes narrations may get split across, so merge them into one.
    5. Narration usually ends with numbers, examples: 'RTGS/EA/UTIBR6202...nt 20241014' or 'o and m expens_2024100989007301' or 'statutory payment_2024101889005634'
    """
    
    try:
        ## Call Gemini via LiteLLM
        # response = completion(
        #     model="gemini/gemini-2.0-flash",
        #     api_key=os.getenv('GEMINI_API_KEY'),
        #     temperature = 0.4,
        #     messages=[{"content": prompt, "role": "user"}]
        # )
        

        ## Call a local model
        # response = completion(
        #     model="ollama/phi3:mini",
        #     api_base= "http://localhost:11434",
        #     temperature = 0,
        #     messages=[{"content": prompt, "role": "user"}]
        # )

        ## Call openrouter models
        response = completion(
            model="openrouter/tngtech/deepseek-r1t-chimera:free",
            api_key=os.getenv('OPENROUTER_API_KEY'),
            temperature = 0.2,
            messages=[{"content": prompt, "role": "user"}]
        )

        # Extract the JSON from the response
        result = response.choices[0].message.content
        
        # Sometimes the response might include markdown code blocks
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0]
        elif '```' in result:
            result = result.split('```')[1]
            
        return json.loads(result)
        
    except Exception as e:
        print(f"Error processing statement: {str(e)}")
        return None


if __name__ == "__main__":
    
    pdf_content = open("bank.pdf", "rb").read()
    images = convert_from_bytes(
        pdf_content,
        dpi=600,
        fmt='png'
    )
    transactions_json = []
    for i in range(len(images)):
        images[i]= preprocess_image(images[i])
        transactions_json.extend(image_to_json(images[i]))
    pre_table_text = extract_pretext(images[0])

    ## Save the transactions JSON to a file
    # with open("table_data.json", 'w') as f:
    #     json.dump(transactions_json, f, indent=2)
    
    result = extract_bank_statement_data(pre_table_text, transactions_json)
    with open("output.json", "w") as f:
        json.dump(result, f, indent=4)