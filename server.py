import torch

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextStreamer, 
    pipeline, 
    TextIteratorStreamer
)


# Model and Tokenizer setup
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=f"./model/{model_name}",
    use_fast=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=f"./model/{model_name}",
    # device_map="auto",
    # trust_remote_code=False,
    torch_dtype=torch.bfloat16,
).eval()

# Streamer and pipeline setup
streamer = TextIteratorStreamer(
    tokenizer, 
    skip_prompt=True, 
    skip_special_tokens=True
)

# text_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=128,
#     temperature=0.3,
#     # top_p=0.95,
#     # top_k=40,
#     # repetition_penalty=1.15,
#     num_return_sequences=1,
#     streamer=streamer,
#     # do_sample=True,
#     pad_token_id=tokenizer.eos_token_id,
# )

text_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

DEFAULT_SYSTEM_PROMPT = """"You are helpful AI."""
def prompt_template(sys_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Template for the prompt to be used in the model.
    
    Args:
        sys_prompt (str, optional): System's prompt. Defaults to DEFAULT_SYSTEM_PROMPT.
    
    Returns:
        str: Prompt template.
    """
    context = "{question}"
    template = f"""
    {context}
    """
    return template

# LLM and Prompt setup
llm = HuggingFacePipeline(
    pipeline=text_pipeline,
    # pipeline_kwargs={"max_new_tokens": 100},
)
template = prompt_template()
prompt = PromptTemplate(template=template, input_variables=["question"])

# Chain Configuration
qa_chain = prompt | llm
