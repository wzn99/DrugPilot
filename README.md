# DrugPilot
A LLM agent framework for drug discovery

Optimizing the ReActAgent based on the Llama-Index framework, leveraging the reasoning capabilities of LLM in combination with eight tool functions to accomplish drug discovery tasks. 

Users input textual descriptions, and DrugPilot returns the corresponding drug discovery task results.

# Tool Functions 
In DrugPilot/baishenglai_backend-main, we have implemented algorithms for eight types of drug discovery tasks, covering the complete process of drug discovery.
- drug_cell_response_regression_prediction
- drug_cell_response_regression_optimization
- drug_drug_response_prediction
- drug_generation
- drug_property_prediction
- drug_synthesis_design
- drug_target_affinity_classification_prediction
- drug_target_affinity_regression_prediction

# Agent FrameWork
Based on llama-index's ReActAgent, we have implemented a series of improvements:  

• **Optimized `output_parser`** for better parsing of LLM outputs.
• **Enhanced `feedback mechanism`** tailored for drug discovery tasks, improving error detection in LLM reasoning and providing corrective feedback.
• **Introduced a f`ocus mechanism`** to prevent LLMs from forgetting the original task in long conversations.
• **Proposed the `memory pool` component**:  
  • Solves the challenge of large-scale data transmission.  
  • Automatically extracts drug-related parameters from conversations and maintains them in a structured format for subsequent task usage.  
  • Provides an interface for users to efficiently view and modify parameters, enabling better control over the LLM's reasoning process. This allows users to more effectively guide and supervise the LLM's decision-making.


# Requirements for LLM agent
| Package Name                                      | Version         |  
|--------------------------------------------------|---------------|  
| llama-agents                                    | 0.0.14        |  
| llama-cloud                                     | 0.1.4         |  
| llama-index                                     | 0.11.20       |  
| llama-index-agent-openai                        | 0.3.4         |  
| llama-index-cli                                 | 0.3.1         |  
| llama-index-core                                | 0.11.20       |  
| llama-index-embeddings-openai                   | 0.2.5         |  
| llama-index-indices-managed-llama-cloud        | 0.4.0         |  
| llama-index-legacy                              | 0.9.48.post3  |  
| llama-index-llms-ollama                         | 0.3.2         |  
| llama-index-llms-openai                         | 0.2.16        |  
| llama-index-multi-modal-llms-openai             | 0.2.3         |  
| llama-index-program-openai                      | 0.2.0         |  
| llama-index-question-gen-openai                 | 0.2.0         |  
| llama-index-readers-file                        | 0.2.2         |  
| llama-index-readers-llama-parse                 | 0.3.0         |  
| llama-parse                                     | 0.5.12        |  

# Dependencies for Drug Tools
```
#以下是框架、中间件依赖
conda install django==4.1
conda install mysqlclient==2.0.3
pip install celery==5.3.6 # 不得低于该版本
pip install eventlet==0.36.0
pip install django-cors-headers==4.3.1
pip install djangorestframework-simplejwt==5.3.1
pip install django-redis==5.4.0

# 以下是算法模块依赖
# pytorch需要根据服务器实际cuda版本（nvcc --version选择）如果conda安装卡住，可以使用pip，二者选其一
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install torch-geometric==2.3.1  # 必须使用该版本 否则模型权重无法加载
pip install ftfy==6.1.3
pip install regex==2023.12.25
pip install rdkit==2023.9.5
pip install networkx==3.1
conda install -c dglteam dgl-cuda11.7==0.9.1post1
pip install dgllife==0.3.2
pip install yacs==0.1.8
pip install sympy==1.12
pip install clip==0.2.0
pip install chardet==5.2.0

# 需要安装torch_scatter==2.1.0
# 在https://pytorch-geometric.com/whl/index.html中根据torch和cuda版本找到对应文件
# 下载 torch_scatter-2.1.0+pt113cu117-cp38-cp38-linux_x86_64.whl 上传服务器
# 并在whl文件所在目录执行以下命令
pip install torch_scatter-2.1.0+pt113cu117-cp38-cp38-linux_x86_64.whl

pip install clip-anytorch==2.6.0
pip install easydict==1.13
pip install einops==0.8.0
pip install pubchempy==1.0.4
pip install pyemd==1.0.0
pip install dill==0.3.8
pip install fcd_torch==1.0.7
pip install pandas==1.5.3 # 不得高于等于2.0.0
pip install matplotlib
pip install h5py
pip install text2vec==1.2.9
```