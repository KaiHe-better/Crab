sudo yum install nano screen -y
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers==0.0.25
pip install .[bitsandbytes]
pip install gpustat tiktoken langchain jsonlines openpyxl
pip install -U transformers==4.43.2