# Bank Statement Extractor

This project is a Python-based tool designed to extract structured data from bank statements in PDF format. It uses OCR (Optical Character Recognition) to process the PDF, extract tabular and textual data, and convert it into a structured JSON format. The project supports both terminal-based usage and a FastAPI-based web interface. Additionally, it allows using **local models or open-source models** from OpenRouter to address security concerns.

---

## Features

- Converts bank statement PDFs into structured JSON data.
- Extracts header information (e.g., IFSC code, account number, etc.).
- Processes tabular transaction data.
- Supports both terminal-based and FastAPI-based usage.
- Allows integration with local models for enhanced security.
- Supports open-source models from OpenRouter for flexibility and transparency.

---

### Potential Use Cases

# Personal Finance Management
Automatically parse your own bank statements into structured data for use in budgeting apps, spreadsheets, or dashboards.

# Accounting and Bookkeeping
Eliminate manual data entry by converting bank statements into structured JSON for easy integration with accounting software.

# Fintech Applications
Integrate this tool into fintech platforms to securely extract transactional data from uploaded bank statements, enabling credit analysis, financial planning, or loan evaluation.

# Compliance and Auditing
Help audit teams or compliance tools convert and analyze financial documents for anomalies, suspicious patterns, or regulatory reporting.

---

## Setup Instructions

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Set Up a Virtual Environment
# On Windows
```bash
python -m venv venv
venv\Scripts\activate
```

# On macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
# Install the required Python packages using requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Create and Configure `.env` File
Some configurations are required for the project to run properly. Create a `.env` file in the project directory and add the following variables:

```env

# API keys for external models
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional: Uncomment and configure if needed
# DEBUG_MODE=true
# OUTPUT_DIR=/path/to/output/directory

# Local model configuration (if applicable)
# LOCAL_MODEL_API_BASE=http://localhost:11434
```

- Replace `/path/to/tesseract` with the path to your Tesseract OCR executable.
- Replace `your_gemini_api_key` and `your_openrouter_api_key` with the respective API keys.
- Uncomment and configure `DEBUG_MODE` and `OUTPUT_DIR` if you want to enable debugging or specify a custom output directory.
- If using a local model, uncomment and set `LOCAL_MODEL_API_BASE` to the base URL of your local model API.

---

## Usage Instructions

### 1. Terminal-Based Usage
You can run the project directly from the terminal without using FastAPI.

**Steps:**
1. Place your bank statement PDF in the project directory and name it `bank.pdf`.
2. Run the script `extractor.py`:
   ```bash
   python extractor.py
   ```
3. The extracted JSON data will be saved in a file named `output.json` in the project directory.
4. You can also get the extracted transation statements to debug or check the output by uncommenting the code block: "Save the transactions JSON to a file"

---

### 2. FastAPI-Based Usage
You can also use the FastAPI web interface to upload PDFs and get the extracted JSON data.

**Steps:**
1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
2. Open your browser or use a tool like Postman to access the API at:
   ```
   http://127.0.0.1:8000
   ```
3. Use the `/docs` endpoint to access the Swagger UI interface:
   ```
   http://127.0.0.1:8000/docs
   ```
   ![FastAPI swagger UI](https://github.com/user-attachments/assets/b00b431c-c0b1-43a3-a32e-36cf8fe31c6f)

4. Use the `/extract-bank-statement/` endpoint in Swagger UI to upload a PDF file.Click on 'try it out'.
   ![Uploadding pdf and execution](https://github.com/user-attachments/assets/896d1ff8-9860-4ca3-876b-5b14c9391ace)

5. Upload your bank statement pdf and click on execute. The API will return the extracted JSON of the bank statement.

---

## Contributions and Suggestions

You are welcome to make contributions to improve this project! Or if have some feedback or ideas feel free to reach out via email at **aum832003@gmail.com** for any feedback .

---

