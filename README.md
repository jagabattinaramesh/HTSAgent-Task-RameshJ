# HTS AI Agent

**Two capabilities in one CLI**:  
1. **RAG-based QA** over the HTS General Notes PDF  
2. **Tariff duty calculator** from HTS CSV data

## Repo structure


## Setup

```bash
git clone <your-repo-url>
cd hts_agent
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
````
## <mark>Usage</mark>
### Download and Install the LLaMA Model

1. Make sure you’re in the project root:  
   ```bash
   cd /home/rameshj/Data_Science/Agents/hts_agent_rameshj
   ````
2. Create the models directory (if it doesn’t exist):
  ```bash
   mkdir -p models
   ````
3. Download the model checkpoint from Hugging Face:
  "Visit: https://huggingface.co/<your-username>/llama-2-7b-chat". Download the file llama-2-7b-chat.Q4_K_M.gguf
4.
   ```bash
    mv ~/Downloads/llama-2-7b-chat.Q4_K_M.gguf models/ 
    ````


