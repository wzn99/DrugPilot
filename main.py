import os

# Import modules from llama_index
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
import pandas as pd
from pprint import pprint
# from llama_index.core.agent import ReActAgent

from llama_index.core.agent.react.base import ReActAgent as OriginReActAgent
from core.agent.ablation_mp.base import ReActAgent as ReActAgentAblationMp
from core.agent.react.base import ReActAgent as DrugPilot

# Set working directory
os.chdir('/home/data1/lk/LLM/function_call/baishenglai_backend-main')

# Import custom drug calculation tools
from algorithm.drug_property.main import drug_property_prediction
from algorithm.drug_cell_response_regression.main import drug_cell_response_regression_predict
from algorithm.drug_target_affinity_regression.main import drug_target_affinity_regression_predict
from algorithm.drug_target_affinity_classification.main import drug_target_classification_prediction
from algorithm.drug_drug_response.main import drug_drug_response_predict
from algorithm.drug_generation.main import drug_cell_response_regression_generation
from algorithm.drug_synthesis_design.scripts.main import Retrosynthetic_reaction_pathway_prediction
from algorithm.drug_cell_response_regression_optimization.main import drug_cell_response_regression_optimization

# Create instances for each task tool
drug_property_prediction = FunctionTool.from_defaults(fn = drug_property_prediction)
drug_target_affinity_regression_predict = FunctionTool.from_defaults(fn = drug_target_affinity_regression_predict)
drug_target_classification_prediction = FunctionTool.from_defaults(fn = drug_target_classification_prediction)
drug_cell_response_regression_predict = FunctionTool.from_defaults(fn = drug_cell_response_regression_predict)
drug_drug_response_predict = FunctionTool.from_defaults(fn = drug_drug_response_predict)
drug_cell_response_regression_generation = FunctionTool.from_defaults(fn = drug_cell_response_regression_generation)
Retrosynthetic_reaction_pathway_prediction = FunctionTool.from_defaults(fn = Retrosynthetic_reaction_pathway_prediction)
drug_cell_response_regression_optimization = FunctionTool.from_defaults(fn = drug_cell_response_regression_optimization)

# List of tools
tools = [   
    drug_property_prediction, drug_target_affinity_regression_predict, 
    drug_target_classification_prediction, drug_cell_response_regression_predict,
    drug_drug_response_predict, drug_cell_response_regression_generation, 
    Retrosynthetic_reaction_pathway_prediction, drug_cell_response_regression_optimization
]

# Create Ollama model instance, specifying model and timeout
llm = Ollama(model="drug_tools_v4", request_timeout=120.0)

# Create ReActAgent instances, passing the function tools list and model to the agent
agent = DrugPilot.from_tools(tools, llm=llm, verbose=True, max_iterations=20)
agent_ablation_mp = ReActAgentAblationMp.from_tools(tools, llm=llm, verbose=True, max_iterations=20)

# Read system prompt template file
file_path = '/home/data1/lk/LLM/function_call/agent_with_memory/core/agent/react/templates/read_memory_pool.md'
with open(file_path, 'r') as file:
    react_system_header_str = str(file.read())

# Create PromptTemplate instance using the template
react_system_prompt = PromptTemplate(react_system_header_str)

# Update system Prompt template
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

# Reset agent state
agent.reset()

# print(agent.get_prompts())

# Example questions for various tasks
mpp_question1 = "Predict the inhibitory activity on BACE1 of the drug 'FC(F)Oc1ccc(cc1)[C@@]1(N=C(N)N(C)C1=O)c1cc(ccc1)\\C=C\\CCCO'."
mpp_question2 = "predict the log solubility property for the drug 'C=C1C2=CCCC=C(NC(C)=CC34CC3COC=C(CCC13CC3)C4(C)CC)N2'."
dta_question = "Predict the affinity of drug CC(Casdfihasif2 and the target protein. The target protein expression is MEILCEDNISLSSIPNSLMQLGDGPRLYHNDFNSRDANTSEASNWTIDAENRTNLSCEGYLPPTCLSILHLQEKNWSALLTTVVIILTIAGNILVIMAVSLEKKLQNATNYFLMSLAIADMLLGFLVMPVSMLTILYGYRWPLPSKLCAIWIYLDVLFSTASIMHLCAISLDRYVAIQNPIHHSRFNSRTKAFLKIIAVWTISVGISMPIPVFGLQDDSKVFKEGSCLLADDNFVLIGSFVAFFIPLTIMVITYFLTIKSLQKEATLCVSDLSTRAKLASFSFLPQSSLSSEKLFQRSIHREPGSYAGRRTMQSISNEQKACKVLGIVFFLFVVMWCPFFITNIMAVICKESCNENVIGALLNVFVWIGYLSSAVNPLVYTLFNKTYRSAFSRYIQCQYKENRKPLQLILVNTIPALAYKSSQLQVGQKKNSQEDAEQTVDDCSMVTLGKQQSEENCTDNIETVNEKVSCV."
dti_question = "Predict whether the drug C1COCCN1C2=NC(=NC3=C2OC4=C3C=CC=N4)C5=CC(=CC=C5)O will interact with the protein target with sequence MLKFQEAAKCVSGSTAISTYPKTLIARRYVLQQKLGSGSFGTVYLVSDKKAKRGEELKVLKEISVGELNPNETVQANLEAQLLSKLDHPAIVKFHASFVEQDNFCIITEYCEGRDLDDKIQEYKQAGKIFPENQIIEWFIQLLLGVDYMHERRILHRDLKSKNVFLKNNLLKIGDFGVSRLLMGSCDLATTLTGTPHYMSPEALKHQGYDTKSDIWSLACILYEMCCMNHAFAGSNFLSIVLKIVEGDTPSLPERYPKELNAIMESMLNKNPSLRPSAIEILKIPYLDEQLQNLMCRYSEMTLEDKNLDCQKEAAHIINAMQKRIHLQTLRALSEVQKMTPRERMRLRKLQAADEKARKLKKIVEEKYEENSKRMQELRSRNFQQLSVDVLHEKTHLKGMEEKEEQPEGRLSCSPQDEDEERWQGREEESDEPTLENLPESQPIPSMDLHELESIVEDATSDLGYHEIPEDPLVAEEYYADAFDSYCEESDEEEEEIALERPEKEIRNEGSQPAYRTNQQDSDIEALARCLENVLGCTSLDTKTITTMAEDMSPGPPIFNSVMARTKMKRMRESAMQKLGTEVFEEVYNYLKRARHQNASEAEIRECLEKVVPQASDCFEVDQLLYFEEQLLITMGKEPTLQNHL."
drp_question = "Predict the regression response value between the drug C1=CC=C(C=C1)[C@H](COC2=CC3=C(C=C2)NC(=O)N3)NC(=O)C4=CC=CN(C4=O)CC5=CC(=C(C=C5)F)F and the cell with cell_name 906855, that is, the affinity between them."
ddi_question = "Please predict the drug-drug interaction between the molecules '[H]\\C(=C(\\[H])C1=NCCCN1C)C1=CC=CS1' and '[H][C@@]1(C[C@@](N)(CC2=C1C(O)=C1C(=O)C3=CC=CC=C3C(=O)C1=C2O)C(C)=O)O[C@H]1C[C@H](O)[C@H](O)CO1', the prediction task is twosides."
dg_question = "Generate some molecules, select the cell line 684059, specify the z-score as -4.054651."
retro_question = "Please predict the synthesis path of C(OC(C(F)(F)F)C(F)(F)F)F, that is, what is its precursor reactant."
do_question = "Optimize the molecule 'CCCCC(C=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)OCC1=CC=CC=C1', generate some optimized molecules, select the cell line 683665, specify the z-score as -0.287463, and mask the atomic sequence as [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]."

# Add content to memory pool
agent.memory_pool.add_to_memory('user_smile', 'C=C1C2=CCCC=C(NC(C)=CC34CC3COC=C(CCC13CC3)C4(C)CC)N2')
agent.memory_pool.add_to_memory('user_property', 'Class')
csv_file_path = "/home/data1/lk/LLM/function_call/agent_with_memory/smiles.csv"
df = pd.read_csv(csv_file_path)
first_column = df.iloc[:, 0].tolist()
agent.memory_pool.add_to_memory('user_smiles', first_column)

# chat
# question = 'what is the logSolubility property for these drugs, and what is the affinity value of them and MEILCEDNISLSSIPNSLMQLGDGPRLYHNDFNSRDANTSEASNWTIDAENRTNLSCEGYLPPTCLSILHLQEKNWSALLTTVVIILTIAGNILVIMAVSLEKKLQNATNYFLMSLAIADMLLGFLVMPVSMLTILYGYRWPLPSKLCAIWIYLDVLFSTASIMHLCAISLDRYVAIQNPIHHSRFNSRTKAFLKIIAVWTISVGISMPIPVFGLQDDSKVFKEGSCLLADDNFVLIGSFVAFFIPLTIMVITYFLTIKSLQKEATLCVSDLSTRAKLASFSFLPQSSLSSEKLFQRSIHREPGSYAGRRTMQSISNEQKACKVLGIVFFLFVVMWCPFFITNIMAVICKESCNENVIGALLNVFVWIGYLSSAVNPLVYTLFNKTYRSAFSRYIQCQYKENRKPLQLILVNTIPALAYKSSQLQVGQKKNSQEDAEQTVDDCSMVTLGKQQSEENCTDNIETVNEKVSCV.'
question = 'predict property'
# question += str(agent.memory_pool)

complex_question = "Generate some molecules, select the cell line 684059, specify the z-score as -4.054651. Then predict the inhibitory activity on BACE1 of the generated molecules."
response = agent_ablation_mp.chat(complex_question)
print("response: ", response)
# pprint(agent.memory_pool)
# agent.reset()
# response2 = agent.chat("prdict the inhibitory activity on BACE1 of these smiles")