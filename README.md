# BIM LLM Code Agent

This is BIM (Building Information Modeling) LLM Code Agent for the purpose of checking LLM Agent Performance as the viewpoint of complicated model like IFC and publishing paper. It is an open-source project that combines Building Information Modeling (BIM) with large language models (LLMs) to handle queries and automate tasks involving BIM files, IFC files. This tool streamlines reasoning, code generation, and analysis for BIM professionals, making it easier to interact with complex IFC BIM. This project is not perfect. In reference, It depends on various parameters like LLM type etc. Sometimes, it has hallucinations and sometimes it generates incomplete code. About these problem, it can be solved using [LLM Function Calls for AI Agents](https://github.com/mac999/AI_agent_simple_function_call) with LLM fine-tuning. If you wan to contribute this project, please pork and PR. 
This requires more formal RAG handling and LLM usage.
<div style="text-align: center;">
<img src="https://github.com/mac999/BIM_LLM_code_agent/blob/main/doc/img2.JPG" height="350">
<img src="https://github.com/mac999/BIM_LLM_code_agent/blob/main/doc/img1.gif" height="350">
</div>

## Features

- **BIM Query Handling**: Processes natural language queries to extract or analyze BIM data.
- **Code Generation**: Automatically generates Python scripts to fulfill user commands.
- **File Compatibility**: Supports multiple file formats, including IFC, PDF, JSON, TXT, and CSV.
- **Interactive Web Interface**: Provides a user-friendly interface using Streamlit.
- **Visualization and Reporting**: Generates tables, charts, and summaries based on user requests.

## Version history
- 0.1: 2024.9. initial version.
- 0.2: 2025.6. fixed bug.
- 0.3: 2025.7. fixed bug. update code knowledge base for RAG properly.
- 0.4: 2025.9. fixed bug. modified hyperparameter for RAG. updated code example. 

## Getting Started

### Prerequisites

- Python 3.8 or later
- An OpenAI API key
- Optional: API keys for LangChain and Tavily for additional features

### Installation

1. **Clone the Repository**  
   Clone the project to your local system:
   ```bash
   git clone https://github.com/mac999/BIM_LLM_code_agent.git
   cd bim-llm-code-agent
   ```

2. **Install Required Libraries**  
   Manually install the necessary Python libraries:
   ```bash
   pip install matplotlib pandas pyvista plotly langchain-openai langchain-core langchain-community streamlit ifcopenshell faiss-cpu numpy
   ```

3. **Set Up Environment Variables**  
   Create a `.env` file in the project directory and configure it with your API keys:
   ```plaintext
   OPENAI_API_KEY=<your_openai_api_key>
   LANGCHAIN_API_KEY=<your_langchain_api_key>
   LANGCHAIN_PROJECT=AGENT_TUTORIAL
   TAVILY_API_KEY=<your_tavily_api_key>
   HF_TOKEN=<your_hf_token>
   ```

4. **Run the Application**  
   Launch the Streamlit application:
   ```bash
   streamlit run bim_code_agent_app.py
   ```

5. **Access the App**  
   Open your browser and navigate to `http://localhost:8501` to start interacting with the BIM LLM Code Agent.

### Example Usage

- **Query**:  
  "Print the number of rooms whose names start with 'A'."
  **Output**: A table summarizing the count of the rooms.

- **Query**:  
  "'A'이름으로 시작하는 방을 표 형식으로 리스트해줘. 각 방의 이름, 면적 속성도 같이 표 형식으로 출력해."
  **Output**: A table summarizing the room names and areas.

- **Query**:  
  "'A'이름으로 시작하는 방을 표 형식으로 리스트해줘. 각 방의 이름, 면적, 속성, 부피도 같이 표 형식으로 출력해. 차트는 각 방의 이름에 대한 면적, 부피를 3차원 차트로 표시해줘야 해. 차트에 표시되는 각 데이터 포인트는 부피에 따라 색상이나 크기가 달라져야 해."  
  **Output**: A 3D Plotly chart displaying room data and the corresponding Python script.

## Project Structure

- `bim_code_agent.py`: Core functionality for managing BIM queries and generating Python code.
- `bim_code_agent_app.py`: Streamlit-based web application to interact with the BIM agent.
- `.env`: Configuration file for API keys and environment variables.

## Key Components

- **Multi-Agent System**: Employs LangChain to integrate LLMs, memory, vector stores, and more.
- **File Handling**: Automatically processes uploaded files for vectorized searches or BIM analysis.
- **Custom Chain**: A specialized LangChain pipeline for BIM-related queries.

## Contribution

Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear explanation of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

If you wan to contribute this project, please pork and PR. 

- **Taewook Kang**  
  Email: [laputa99999@gmail.com](mailto:laputa99999@gmail.com)

## References

- [LangChain](https://www.langchain.com/)
- [IfcOpenShell](http://ifcopenshell.org/)
- [Streamlit](https://streamlit.io/)
- [BIM Knowledge Expert Agent Research Based on LLM and RAG](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003125522)
- [Is Langchain really pointless?](https://medium.com/@wp237/is-langchain-really-pointless-2302dea10d6d)
