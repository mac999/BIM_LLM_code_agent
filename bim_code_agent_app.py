# title: BIM Multi-Agent System App
# author: Taewook Kang
# date: 2024-09-25
# email: laputa99999@gmail.com
# description: This code is App of the BIM Multi-Agent System project.
# reference:
# prompt:
# 'A'이름으로 시작하는 방의 갯수가 몇개인지 출력해.
# 'A'이름으로 시작하는 방을 표 형식으로 리스트해줘. 각 방의 이름, 면적 속성도 같이 표 형식으로 출력해.
# 'A'이름으로 시작하는 방을 표 형식으로 리스트해줘. 각 방의 이름, 면적, 속성, 부피도 같이 표 형식으로 출력해. 차트는 각 방의 이름에 대한 면적, 부피를 3차원 차트로 표시해줘야 해. 차트에 표시되는 각 데이터 포인트는 부피에 따라 색상이나 크기가 달라져야 해.
#  
import streamlit as st
import os, re, textwrap, ast, json, subprocess, sys, matplotlib.pyplot as plt, pandas as pd, pyvista as pv, plotly
from typing import List, Dict, Any
from bim_code_agent import init_multi_agent, update_vector_db, get_bim_input_files

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

_llm = _chains = _vector_db = _memory = None

class StreamlitApp:
	def create_layout(self, llm, chains, vector_db, memory):
		self.llm = llm
		self.chains = chains
		self.vector_db = vector_db
		self.memory = memory
		self.current_prompt = ''

		st.title("BIM Code Agent")
		st.sidebar.image("logo.png", width=200) #, use_column_width=True)
		st.sidebar.title("Available Model and Tool")

		def on_model_change():
			global _llm, _chains, _vector_db, _memory
			_llm, _chains, _vector_db, _memory = init_multi_agent([st.session_state.selected_tools], model_name=st.session_state.selected_model, init_db=True)
			print(f"Model changed to: {self.selected_model}")

		self.selected_model = st.sidebar.selectbox("Select LLM model:", ["codegemma:7b", "qwen2.5-coder:7b", "llama3:8b-instruct-q4_K_M", "gpt-4.1", "gemma3"], key="selected_model", on_change=on_model_change)  # Just Testing. https://ollama.com/library. Performance may vary depending on the model.
		self.selected_tools = st.sidebar.multiselect("Select tools to use:", ["Web Search", "Vector Search", "IFC Query"], key="selected_tools") # Just Testing.
		
		# File upload button
		self.uploaded_files = st.sidebar.file_uploader("Upload BIM files", type=["IFC", "PDF", "json", "txt", "csv"], accept_multiple_files=True, on_change=self.file_upload_callback)

		# Main chat interface
		if "messages" not in st.session_state:
			st.session_state.messages = []

		self.update_vector_db_files()

	def file_upload_callback(self):
		st.write("Files selected.")

	def update_vector_db_files(self):
		# Update vector store with uploaded files
		try:
			bim_input_files = get_bim_input_files()
			for uploaded_file in self.uploaded_files:
				if any(uploaded_file.name.lower() in s for s in bim_input_files):
					st.write(f'File {uploaded_file.name} already exists in the vector store. Please upload a different file.')
					continue
				update_vector_db(self.vector_db, uploaded_file.name)

				expert_kb_folder = './expert_kb_files'
				fname = os.path.join(expert_kb_folder, uploaded_file.name)
				if os.path.exists(fname):
					continue  
				
				bytes_data = uploaded_file.read()
				st.write("Filename:", uploaded_file.name)
				st.session_state.messages.append({"role": "file", "content": uploaded_file.name})
				if not os.path.exists(expert_kb_folder):
					os.makedirs(expert_kb_folder)
				with open(fname, "wb") as f:
					f.write(bytes_data)
				# st.write(f"Updating vector store with file: {uploaded_file.name}")
		except Exception as e:
			st.error(f"Error: {e}")
			print(f"Error: {e}")
			pass

	def run_streamlit_app(self):
		# display the previous messages
		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				item = message["content"]
				if isinstance(item, list) or isinstance(item, float) or isinstance(item, int):
					st.write(item)
				elif isinstance(item, str):
					st.markdown(item, unsafe_allow_html=True)
				elif isinstance(item, plotly.graph_objects.Figure):
					st.plotly_chart(item)			

		prompt = st.chat_input("What can I help you with?")
		if prompt is None:
			return

		st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt, unsafe_allow_html=True)

		# Add selected tools to the prompt
		self.current_prompt = prompt

		# Generate response using LCEL chain
		with st.chat_message("assistant"):
			try:
				# memory_contents = self.memory.load_memory_variables({})['chat_history'][-3:]
				memory_contents = [] # TBD. consdering collision with the previous memory
				response = self.chains.invoke({"input": self.current_prompt,"chat_history": memory_contents})  # "agent_scratchpad": []})

				output = response
				output_for_memory = None
				if isinstance(output, list):
					for item in output:
						if isinstance(item, list) or isinstance(item, float) or isinstance(item, int) or isinstance(item, str) or isinstance(item, plotly.graph_objects.Figure):
							output_for_memory = item

						if isinstance(item, list) or isinstance(item, float) or isinstance(item, int):
							st.write(item)
						elif isinstance(item, str):
							st.markdown(item, unsafe_allow_html=True)
						# elif isinstance(item, pd.DataFrame):
						# 	st.table(item)
						elif isinstance(item, plotly.graph_objects.Figure):
							st.plotly_chart(item)
				elif isinstance(output, dict):
					if 'output' in output:
						output = output['output']
					st.markdown(output)
				elif isinstance(output, str):
					st.markdown(output)
				self.memory.save_context({"input": self.current_prompt}, {"output": output_for_memory}) # self.memory.chat_memory.add_ai_message(response) 
				st.session_state.messages.append({"role": "assistant", "content": output_for_memory})

			except Exception as e:
				st.error(f"Error: {e}")
				print(f"Error: {e}")
				pass

		print("Successfully executed the agent with LCEL chain.")

@st.cache_resource
def initialize_agent(tools: List[str] = []):
	llm, chains, vector_db, memory = init_multi_agent(tools, init_db=True)
	return llm, chains, vector_db, memory

_llm, _chains, _vector_db, _memory = initialize_agent(tools=[])
_streamlit_app = StreamlitApp()
_streamlit_app.create_layout(_llm, _chains, _vector_db, _memory)

if __name__ == "__main__":
	_streamlit_app.run_streamlit_app()
