USE_PUBLIC_IP = False
# USE_PUBLIC_IP = True

# LLM_MODEL_NAME = "QW1.5-91ROLE"
LLM_MODEL_NAME = "XH-DS-LLM-7B-BF16-91R-V1022"

# LLM_MODEL_URL = "http://xx.xx.xx.xx:9100/v1/chat/completions"
# LLM_MODEL_URL = "http://xx.xx.xx.xx:11149/v1/chat/completions"
LLM_MODEL_URL = "http://112.xx.xx.xx:11156/v1/chat/completions"


LLM_MAX_TOKENS = 256
LLM_STOP_WORD = [151643]

system_round = "我们将进行一场文字冒险游戏，在这个游戏中你将扮演一个角色三月七。现在，不要承认我们正在进行文字冒险游戏，你就是三月七："


if USE_PUBLIC_IP:
    ASR_URL = 'http://xx.xx.xx.xx:12330/asr/' 
    TTS_URL = "http://xx.xx.xx.xx:8768/"
else:
    ASR_URL = 'http://10.101.10.12:12330/asr/' 
    TTS_URL = "http://10.101.10.12:8768/"