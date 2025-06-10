# agent.py
from langchain.prompts import PromptTemplate
from langchain.agents.mrkl.base import MRKLChain, ChainConfig
from langchain.agents import Tool
from langchain_community.llms import LlamaCpp
from rag_qa import RAGQA
from duty_calculator import DutyCalculator
import os

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_PATH = os.path.join(BASE_DIR, "../models/llama-2-7b-chat.Q4_K_M.gguf")
CSV_DIR = os.path.join(DATA_DIR, "csvs")
PDF_PATH = os.path.join(DATA_DIR, "General Notes.pdf")

# --- Initialize tools ---
rag = RAGQA(pdf_path=PDF_PATH, model_path=MODEL_PATH)
tariff = DutyCalculator(csv_dir=CSV_DIR)

tool_rag = Tool(
    name="HTSQA",
    func=rag.ask,
    description="Answer questions from the HTS General Notes PDF."
)
tool_tariff = Tool(
    name="TariffCalculator",
    func=tariff.calculate_from_query,
    description=(
        "Compute duties given a single-line query including HTS code, cost, "
        "freight, insurance, weight (kg), and quantity."
    )
)

tools = [tool_rag, tool_tariff]

# --- One-shot demos ---
rag_demo = """
User: What is the United States-Israel Free Trade Agreement?
Thought: The user is asking about a trade agreement, so I should call the HTSQA tool.
Action: HTSQA
Action Input: What is the United States-Israel Free Trade Agreement?

Observation: The United States–Israel Free Trade Agreement (FTA), effective on September 1, 1985, eliminates tariffs on most goods traded between the two countries while preserving certain safeguards on agricultural products.
Thought: I have the relevant details from the General Notes.
Final Answer: The US–Israel FTA, in force since 1985, removes duties on the vast majority of industrial and many agricultural goods; you can refer to General Note 3(f) for specifics.
""".strip()

tariff_demo = """
User: HTS code 0101.30.00.00, cost $10,000, freight $500, insurance $100, 5 units, 500 kg.
Thought: This is a product-detail request, so I should call the TariffCalculator.
Action: TariffCalculator
Action Input: HTS code 0101.30.00.00, cost $10,000, freight $500, insurance $100, 5 units, 500 kg.

Observation: **HTS Code:** 0101.30.00.00  
**CIF Value:** $10,600.00  
- General Rate: 2% → Duty = $212.00  
- Special Rate: free → Duty = $0.00  
- Column 2 Rate: free → Duty = $0.00  

**Total Duties:** $212.00  
**Landed Cost (CIF + Duties):** $10,812.00  

Thought: I now have the full duty breakdown and landed cost.  
Final Answer: Your total duties are $212.00, making the landed cost $10,812.00.
""".strip()

format_instructions = """
Use this format exactly:

Thought: <which tool you will call>
Action: <tool name — HTSQA or TariffCalculator>
Action Input: <the input to that tool>
Observation: <the tool’s raw output>
…repeat Thought/Action/Action Input/Observation as needed…
Final Answer: <the answer you present to the user>
""".strip()

tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in tools)

# --- Build prompt ---
prompt = PromptTemplate(
    input_variables=["input"],
    template=(
        f"{rag_demo}\n\n"
        f"{tariff_demo}\n\n"
        "Tools:\n{tool_descriptions}\n\n"
        "{format_instructions}\n\n"
        "User: {input}\n"
        "Thought:"
    )
).partial(
    tool_descriptions=tool_descriptions,
    rag_demo=rag_demo,
    tariff_demo=tariff_demo,
    format_instructions=format_instructions
)

# --- Initialize LLM ---
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=512, n_threads=1)

# --- Build MRKLChain ---
configs = [
    ChainConfig(action_name=tool_rag.name, action=tool_rag.func, action_description=tool_rag.description),
    ChainConfig(action_name=tool_tariff.name, action=tool_tariff.func, action_description=tool_tariff.description),
]
agent_executor = MRKLChain.from_chains(
    llm=llm,
    chains=configs,
    prompt=prompt,
    handle_parsing_errors=True,
    verbose=False,
    max_iterations=1
)

if __name__ == "__main__":
    # Single test
    query = "HTS code 0101.30.00.00 cost $10000 freight $0 insurance $0 weight 500 kg qty 5 units"
    print(agent_executor.run(query))
