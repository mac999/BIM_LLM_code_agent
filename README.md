# BIM LLM Code Agent

This is BIM LLM Code Agent for the purpose of checking LLM Agent Performance as the viewpoint of complicated model like IFC and publishing paper. It is an open-source project that combines Building Information Modeling (BIM) with large language models (LLMs) to handle queries and automate tasks involving BIM files, IFC files. This tool streamlines reasoning, code generation, and analysis for BIM professionals, making it easier to interact with complex IFC BIM.
<div style="text-align: center;">
<img src="https://github.com/mac999/BIM_LLM_code_agent/blob/main/doc/img1.gif" height="400">
</div>
## Features

- **BIM Query Handling**: Processes natural language queries to extract or analyze BIM data.
- **Code Generation**: Automatically generates Python scripts to fulfill user commands.
- **File Compatibility**: Supports multiple file formats, including IFC, PDF, JSON, TXT, and CSV.
- **Interactive Web Interface**: Provides a user-friendly interface using Streamlit.
- **Visualization and Reporting**: Generates tables, charts, and summaries based on user requests.

## Getting Started

### Prerequisites

- Python 3.8 or later
- An OpenAI API key
- Optional: API keys for LangChain and Tavily for additional features

### Installation

1. **Clone the Repository**  
   Clone the project to your local system:
   ```bash
   git clone https://github.com/yourusername/bim-llm-code-agent.git
   cd bim-llm-code-agent
   ```

2. **Install Required Libraries**  
   Manually install the necessary Python libraries:
   ```bash
   pip install matplotlib pandas pyvista plotly langchain-openai langchain-core langchain-community streamlit ifcopenshell
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
  "List all rooms starting with 'A' and show their names and areas in a table."  
  **Output**: A table summarizing the room names and areas, along with the Python code used to generate it.

- **Query**:  
  "Create a 3D chart of room volumes and areas."  
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

- **Taewook Kang**  
  Email: [laputa99999@gmail.com](mailto:laputa99999@gmail.com)

## References

- [LangChain](https://www.langchain.com/)
- [IfcOpenShell](http://ifcopenshell.org/)
- [Streamlit](https://streamlit.io/)

---

This README provides an accessible, clear explanation of the project and its usage. Let me know if you'd like to make further changes!
