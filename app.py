from openai import OpenAI
import PyPDF2
import json
from pathlib import Path
from typing import List, Dict, Tuple

client = OpenAI(api_key=open("openai_key.txt", "r").read())

def read_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def read_requirements(requirements_path: str) -> List[str]:
    """Read requirements from text file."""
    with open(requirements_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def analyze_requirement(requirement: str, document_text: str) -> Dict:
    """Analyze a single requirement against the document using GPT."""
    prompt = f"""
    Analyze if the following requirement is met in the building document text.
    Requirement: {requirement}
    
    Document text: {document_text}
    
    Respond in JSON format:
    {{
        "status": "met" | "not_met" | "not_addressed",
        "explanation": "detailed explanation",
        "error": "error details if not met, otherwise null"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.choices[0].message.content)

def main():
    # Get file paths from user
    pdf_path = input("Enter the path to the building PDF file: ")
    requirements_path = input("Enter the path to the requirements text file: ")
    
    # Validate file paths
    if not Path(pdf_path).exists() or not Path(requirements_path).exists():
        print("Error: One or both files do not exist!")
        return
    
    # Read files
    try:
        document_text = read_pdf(pdf_path)
        requirements = read_requirements(requirements_path)
    except Exception as e:
        print(f"Error reading files: {str(e)}")
        return
    
    # Analyze each requirement
    results = []
    for requirement in requirements:
        print(f"\nAnalyzing requirement: {requirement}")
        analysis = analyze_requirement(requirement, document_text)
        results.append({"requirement": requirement, "analysis": analysis})
    
    # Display results
    print("\n=== Analysis Results ===")
    for result in results:
        req = result["requirement"]
        analysis = result["analysis"]
        print(f"\nRequirement: {req}")
        print(f"Status: {analysis['status']}")
        print(f"Explanation: {analysis['explanation']}")
        if analysis['error']:
            print(f"Error: {analysis['error']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
