# title: BIM Multi-Agent System
# author: Taewook Kang
# date: 2024-09-25
# email: laputa99999@gmail.com
# description: This code is a part of the BIM Multi-Agent System project. Simple version to explain the concept.
# reference:
# 
import sys, os, json, argparse, re, textwrap, ast, subprocess, sys, shutil
import matplotlib.pyplot as plt, pandas as pd, pyvista as pv, plotly
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import FAISS # Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, OnlinePDFLoader, TextLoader, JSONLoader, CSVLoader
from langchain_community.retrievers import BM25Retriever

os.environ["LANGCHAIN_PROJECT"] = "AGENT TUTORIAL"

llm = memory = embeddings = vectorstore = tools = prompt = agent = agent_executor = chains = None
bim_input_files = []

def output_log(message: str, append: bool = False):
	"""Output log message to console."""
	print(message)
	log_file = open('log.txt', 'a' if append else 'w', encoding='utf-8')
	if log_file != None:
		log_file.write(message + '\n')
		log_file.close()

# define custom chain
@chain
def BIM_chain(inputs: Dict) -> Dict:
	global llm, memory, embeddings, vectorstore, tools, prompt, agent, agent_executor, chains

	ai_msg = human_msg = None
	msg_count = len(inputs.messages)
	index = msg_count - 1
	while index >= 0:
		msg = inputs.messages[index]
		if isinstance(msg, SystemMessage) or isinstance(msg, AIMessage):
			ai_msg = msg
		if isinstance(msg, HumanMessage):
			human_msg = msg
		if ai_msg and human_msg:
			break
		index -= 1

	if ai_msg == None or human_msg == None:
		return inputs

	ai_content = ai_msg.content
	human_content = human_msg.content
	selected_tools = 'IFC Query' # Just Testing for options

	input_files = get_bim_input_files()
	input_fname = 'input.ifc'
	if len(input_files) > 0:
		input_fname = './expert_kb_files/' + input_files[0]

	docs = vectorstore.similarity_search(human_content, 3)
	docs_contents = "\n".join([doc.page_content for doc in docs])
	contents = f'{docs_contents}'

	tools_prompt = f"""
	You are an expert in {selected_tools} field. Refer to the following ### Context and ### Example to generate executable Python code only without comments for ### User command.

	### Context:
	1) Use IfcOpenShell in the BIM file {input_fname} to generate source code the user command in Python without inline code and main entry function.
	2) Don't use try except block in the generated code.
	3) Save the intermediate result of executing the user command using the process_list list variable. The process_list list contains a dictionary called obj. obj must define the name of the BIM object as name, the type as type, and other properties of the BIM object (product) as names and values.
	For example, among the property names, area is Area, and volume is Volume, which are common names. Exclude the BIM property values ​​stored in the process_list that have the same name and value.
	4) The variable name obtained as the result of the user command must always start with the tag named 'result_'. If the user command includes a command to output a table, save the result in a variable called result_df after creating a dataframe using the pandas library.
	5) If the user command includes a command to output a chart, save it in a variable called result_fig using the plotly library.
	6) If you need to get the corresponding objects (products) with attribute values ​​such as the name, use the following example code.
	
	### Example:
	{contents}.
	
	The generated code order is the library import section, function declaration sections such as get_object_as_name functions, and the execution code to execute the command. The variable that stores the calculation result for the user command should be stored in the variable starting with the name result_, and the summary format of the final output should be stored in the result_markup using only HTML table, th, tr, td tags.
	
	IMPORTANT: 
	- Do NOT use \\' or \\" in your code
	- Write clean, executable Python code

	The user command is as follows.	
	### User command:
	"""
	# 'A'이름으로 시작하는 방의 갯수가 몇개인지 출력해.
	# 'A'이름으로 시작하는 방을 표 형식으로 리스트해줘. 각 방의 이름, 면적 속성도 같이 표 형식으로 출력해.
	# 'A'이름으로 시작하는 방을 표 형식으로 리스트해줘. 각 방의 이름, 면적, 속성, 부피도 같이 표 형식으로 출력해. 차트는 각 방의 이름에 대한 면적, 부피를 3차원 차트로 표시해줘야 해. 차트에 표시되는 각 데이터 포인트는 부피에 따라 색상이나 크기가 달라져야 해.
	# 
	full_prompt = f'{tools_prompt}\n"{human_content}"'

	output_log(f"* Full prompt\n{full_prompt}")
	
	return full_prompt

def activate_virtualenv(venv_path):
	bin_path = os.path.join(venv_path, 'bin') if os.name != 'nt' else os.path.join(venv_path, 'Scripts')
	os.environ["PATH"] = f"{bin_path}{os.pathsep}{os.environ['PATH']}"
	os.environ["PYTHONHOME"] = venv_path
	site_packages_path = os.path.join(venv_path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
	if os.path.exists(site_packages_path):
		sys.path.insert(0, site_packages_path)

def check_safe_eval(express):
	tokens = express.split()

	try:
		if tokens.index("import") < 0:
			return 
		unsafe_libs = ["os", "shutil", "subprocess", "ctypes", "pickle", "http", "socket", "eval", "exec"]
		unsafe = False
		for lib in unsafe_libs:
			try:
				if tokens.index(lib) >= 0:
					unsafe = True
					break
			except ValueError:
				pass

		if unsafe:
			raise Exception(f"{express} is not safe.")
	except ValueError:
		pass
	return 

def preprocess_code(text: str) -> str:
	try:
		match = re.search(r'```python\n(.*?)```', text, re.DOTALL) # extract code from text between ```python\n and ```
		code = match.group(1).strip()
		code = code.replace('\t', '    ')
		code = textwrap.dedent(code)
		check_safe_eval(code)
		ast.parse(code)
	except IndentationError as e:
		print(f"IndentationError detected: {e}")
		code = ''
	except SyntaxError as e:
		print(f"SyntaxError detected: {e}")
		code = ''
	except Exception as e:
		print(f"Error: {e}")
		code = ''
	return code

def run_python_code(code: str):
	global llm, memory, embeddings, vectorstore, tools, prompt, agent, agent_executor, chains

	output_list = []
	try:
		code = preprocess_code(code)
		if len(code) == 0:
			return 'Invalid Python code detected. Please check the code and try again.'

		for try_index in range(3):
			try:
				output_log(f"* Generated code\n{code}", append=True)
				exec(code)
					
				if 'result_markup' in locals():
					output = locals()['result_markup']
					break
				else: 
					raise Exception("No result_markup found.")

			except Exception as e:
				print(f"Error: {e}")
				prompt = f"Fix error '{e}' in the below code without comments and generate new executable python code only.\n\n {code}"
				response = llm.invoke(prompt)
				content = response.content
				code = preprocess_code(content)
				pass
		
		if 'result_markup' in locals():
			output = locals()['result_markup']

		if len(output) > 0 and ('<table' in output or '<TABLE' in output):
			local_vars = locals().copy()
			vars_list = []
			for var in local_vars:
				if not var.startswith('__'):
					vars_list.append(var)
			for var in vars_list:
				if 'result_' not in var:
					continue
				obj = local_vars[var]
				try:
					if isinstance(obj, pd.DataFrame) or isinstance(obj, list) or isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str) or isinstance(obj, plotly.graph_objects.Figure):
						output_list.append(obj)
				except Exception as e:
					pass

		elif len(output) > 0 and '|' in output:
			output = output.replace('\\n', '\n')
			output_list.append(output)

	except Exception as e:
		print(f"Error: {e}")
		return output_list
	return output_list

@chain
def run_command_chain(input: str) -> Dict:
	global llm, memory, embeddings, vectorstore, tools, prompt, agent, agent_executor, chains

	hints = ["code", "coding", "python", "PYHTON", "코드", "코딩", "Here is the code", "Below is the implementation", "You can use this code"]
	if any(hint in input for hint in hints) == False:
		return 'Invalid order. Please check the order and try again.'
	output = run_python_code(input)
	return output

from ollama import chat
from ollama import ChatResponse

def init_multi_agent(tools_option, model_name="gpt-4o", init_db=False):
	global llm, memory, embeddings, vectorstore, tools, prompt, agent, agent_executor, chains

	# Initialize LLM
	if model_name.startswith("gpt"):
		llm = ChatOpenAI(model=model_name, temperature=0)
		embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
	else:
		llm = ChatOllama(model=model_name, temperature=0) # ChatOllama(model=model_name, temperature=0)
		embeddings = FastEmbedEmbeddings()

	# Initialize memory
	memory = ConversationBufferMemory(
		memory_key="chat_history",
		return_messages=True
	)
	
	# load documents including .txt .pdf .csv .json in ./code_sample folder
	if init_db:
		if os.path.exists("./vectorstore_db") and os.access("./vectorstore_db", os.W_OK):
			shutil.rmtree("./vectorstore_db")

	if os.path.exists("./vectorstore_db"):
		vectorstore = FAISS.load_local("./vectorstore_db", embeddings, allow_dangerous_deserialization=True)
	else:
		input_files = [f for f in os.listdir("./code_sample") if os.path.isfile(os.path.join("./code_sample", f))]
		split_docs = []
		for fname in input_files:
			loader = None
			if fname.endswith(".pdf"): 
				loader = PyPDFLoader(file_path=f"./code_sample/{fname}") # "./IfcOpenShell_0_8_code_example.pdf") # ./202212_LiDAR.pdf")
			elif fname.endswith(".txt"): 
				loader = TextLoader(file_path=f"./code_sample/{fname}", encoding = 'UTF-8')
			elif fname.endswith(".csv"): 
				loader = CSVLoader(file_path=f"./code_sample/{fname}", encoding = 'UTF-8')
			elif fname.endswith(".json"): 
				loader = JSONLoader(file_path=f"./code_sample/{fname}", encoding = 'UTF-8')
			else: 
				continue

			text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
			docs = loader.load_and_split(text_splitter)
			split_docs.extend(docs)

		# vectorstore = Chroma(collection_name="documentation", embedding_function=embeddings, persist_directory="./chroma_db")
		vectorstore = FAISS.from_documents(split_docs, embeddings)
		vectorstore.save_local("./vectorstore_db") 

	# Tools setup
	tools = []
	if 'Web Search' in tools_option:
		tools.append(web_search)
	if 'Vector Search' in tools_option:
		pass # TBD
	
	if len(tools_option) > 0:
		pass

	prompt = ChatPromptTemplate.from_messages([
		("system", """You are a helpful AI assistant. If you don't know, answer I don't know."""),
		MessagesPlaceholder(variable_name="chat_history"),
		("human", "{input}"),
		# ("assistant", "{agent_scratchpad}")  
	])

	# Define chains
	chains = (
		prompt
		| BIM_chain 
		| llm
		| StrOutputParser()
		| run_command_chain
	)
	
	return llm, chains, vectorstore, memory

def get_bim_input_files():
	return bim_input_files

def update_vector_db(vectorstore, file_path: str):
	global bim_input_files

	file_path = file_path.lower()
	print(f"Updating vector store with file: {file_path}")

	if file_path.endswith(".pdf"):
		loader = PyPDFLoader(file_path)
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
		split_docs = loader.load_and_split(text_splitter)
		vectorstore.add_documents(split_docs)
	elif file_path.endswith(".json"):
		with open(file_path, "r", encoding='utf-8') as f:
			data = json.load(f)
			vectorstore.add_documents(data)
	elif file_path.endswith(".txt"):
		with open(file_path, "r", encoding='utf-8') as f:
			data = f.read()
			vectorstore.add_documents(data)
	elif file_path.endswith(".ifc"): 
		bim_input_files.append(file_path)

	return vectorstore

def web_search(query: str) -> List[Dict[str, Any]]:
	"""Search GIS using web sites, not IFC and ifc query"""
	search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
	output = search.run(query)
	return output

def search_vector_store(query: str) -> List[Document]:
	"""Search ifcopenshell code example for parsing IFC file"""
	global vectorstore
	output = vectorstore.as_retriever().get_relevant_documents(query)
	return output
