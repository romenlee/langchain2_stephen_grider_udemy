from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI()
llm = ChatOpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

code_chain = code_prompt | llm

result = code_chain.invoke(input={
    "language": "python",
    "task": "return a list of numbers"
})

print(result)
