import transformers
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch

###LANGCHAIN
import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

import textwrap



def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""



tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
model = LlamaForCausalLM.from_pretrained(
    "chainyo/alpaca-lora-7b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
)

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

instruction = "What is the meaning of life?"
input_ctxt = None  # For some tasks, you can provide an input context to help the model generate a better response.

prompt = generate_prompt(instruction, input_ctxt)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )

response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print(response)






#########################QA################################


##LANGCHAIN

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('./new_papers/new_papers/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()
len(documents)

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## Here is the nmew embeddings being used
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)


retriever = vectordb.as_retriever(search_kwargs={"k": 3})



# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=model, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)



def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])





query = "What is Flash attention?"
llm_response = qa_chain(query)
process_llm_response(llm_response)
