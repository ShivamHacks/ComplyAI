from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from pathlib import Path
import PyPDF2
from typing import List, Dict

# Initialize OpenAI
llm = ChatOpenAI(api_key=open("openai_key.txt", "r").read(), model="gpt-4")

class DocumentAnalysisTool(BaseTool):
    name: str = "document_analysis"
    description: str = "Analyzes if a building document meets specified requirements"
    
    def _read_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() for page in reader.pages)
            
    def _read_requirements(self, req_path: str) -> List[str]:
        """Read requirements from file."""
        with open(req_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    
    def _run(self, query: str) -> str:
        """Run the analysis."""
        try:
            # Get paths
            pdf_path = "data/DesignCorrect.pdf"
            req_path = "data/requirements.txt"
            
            # Validate paths
            if not all(Path(p).exists() for p in [pdf_path, req_path]):
                return "Error: Required files not found in data directory"
                
            # Read documents
            doc_text = self._read_pdf(pdf_path)
            requirements = self._read_requirements(req_path)
            
            # Analyze requirements
            results = []
            for req in requirements:
                analysis = f"Analyzing requirement: {req}\n"
                analysis += f"Document text excerpt: {doc_text[:500]}...\n"
                analysis += "Is this requirement met? Please explain."
                
                response = llm.invoke(analysis)
                results.append(f"Requirement: {req}\nAnalysis: {response.content}\n")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"

# Create tool list
tools = [DocumentAnalysisTool()]

# Create agent prompt
prompt = PromptTemplate.from_template("""
You are a building requirements analyst. Your task is to analyze building documents 
and determine if they meet specified requirements.

You have access to the following tools:
{tools}
                                      
Tool Names: {tool_names}

To analyze the documents, follow these steps:
1. Use the document_analysis tool to read and analyze the building document
2. Review the requirements and determine if they are met
3. Provide a clear explanation for each requirement

You must respond in this format:
Thought: Consider what tool to use and why
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
""")

# Create and run agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

def analyze_building():
    """Run the building analysis agent."""
    return agent_executor.invoke({"input": "Analyze the building document and tell me if it meets all requirements."})

if __name__ == "__main__":
    result = analyze_building()
    print(result["output"])
