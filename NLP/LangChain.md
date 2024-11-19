```python
from langchain.chains import SimpleChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(model="gpt-4", temperature=0.7)

# Define prompts
summarize_prompt = PromptTemplate(
    input_variables=["document"],
    template="Summarize the following document:\n{document}"
)

qa_prompt = PromptTemplate(
    input_variables=["summary", "question"],
    template="Using the summary below, answer the question:\nSummary: {summary}\nQuestion: {question}"
)

# Build the chain
chain = SimpleChain(
    steps=[
        {"action": "summarize", "prompt": summarize_prompt, "output_key": "summary"},
        {"action": "qa", "prompt": qa_prompt, "output_key": "answer"}
    ]
)

# Input document
document = "LangChain is a framework for building applications with LLMs..."
question = "What are the key features of LangChain?"

# Execute the chain
result = chain.run({"document": document, "question": question})
print(result["answer"])
```

In **LangChain**, when you execute the chain using `chain.run(...)`, the result will contain the **final step's output**. Specifically:

1.  The chain processes each step sequentially.
2.  The output of each step is stored and used as input for subsequent steps (if needed).
3.  The `run` method returns the output of the **last step** in the chain.

