from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import pdf2image
import json
from extractor import (
    preprocess_image,
    extract_pretext,
    image_to_json,
    extract_bank_statement_data
)
import numpy as np

app = FastAPI()

@app.post("/extract-bank-statement/")
async def extract_bank_statement(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        pdf_content = await file.read()
        print("read pdf")

        images = pdf2image.convert_from_bytes(
            pdf_content,
            dpi=600,
            fmt='png'
        )
        print("converted to images")
        
        transactions_json = []
        for i in range(len(images)):
            images[i]= preprocess_image(images[i])
            transactions_json.extend(image_to_json(images[i]))
        pre_table_text = extract_pretext(images[0])

        result = extract_bank_statement_data(pre_table_text, transactions_json)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)