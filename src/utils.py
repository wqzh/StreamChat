# -*- coding: utf-8 -*-

import json
import requests
from time import time as timeit
import base64
import wave

from const import LLM_MODEL_NAME, LLM_MODEL_URL, LLM_STOP_WORD, LLM_MAX_TOKENS
from const import ASR_URL, TTS_URL
from const import system_round
query = [{"role": "system", "content": system_round}, {"role": "user", "content": "请做一下自我介绍"}]


def save_bytes_as_wav(bytes_data, output_path, sample_rate, num_channels, bits_per_sample):
    wav_file = wave.open(output_path, 'wb')
    wav_file.setparams((num_channels, bits_per_sample//8, sample_rate, 0, 'NONE', 'not compressed'))
    wav_file.writeframes(bytes_data)
    wav_file.close()
    

def create_chat_completion(messages, use_stream=False, 
    model_name=LLM_MODEL_NAME, llm_url=LLM_MODEL_URL):
    ## 注意： 这个本人自己搭建的api, 请替换成自己的api 或者 第三方api
    
    headers = {'content-type': 'application/json','Authorization': 'Bearer EMPTY'}
    print('----> [util/create_chat_completion]', model_name, llm_url)
    
    data = {
        # "model": LLM_MODEL_NAME,
        "model": model_name,
        
        "messages": messages, 
        "stream": use_stream,
        "temperature": 1.0,
        "top_p": 0.85,
        "top_k": 5,
        "repetition_penalty": 1.05,
        "stop_token_ids": LLM_STOP_WORD,  
        "max_tokens": LLM_MAX_TOKENS,
    }

    # response = requests.post(LLM_MODEL_URL, headers = headers, json = data, stream = use_stream)
    response = requests.post(llm_url, headers = headers, json = data, stream = use_stream)
    
    if response.status_code == 200:
        if use_stream:
        # 处理流式响应
            for line in response.iter_lines():
                print("IterLine", line)
                if line:
                    decoded_line = line.decode('utf-8').split(':', 1)[-1].strip()
                    #print("decoded_line", decoded_line)
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "") 
                        yield content
                    except Exception as e:
                        print("Error Line:", line)
                        print("Error:", str(e))
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应:
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            # print(content)
            yield content
    else:
        print(" create_chat_completion Error:", response.status_code)
        return None



def request_tts(speaker="tianmeishaonv", text="你今天穿的衣服真好看！", timeout=10, stdip=None):
    ## 注意： 这个本人自己搭建的api, 请替换成自己的api 或者 第三方api
    
    if stdip is None: stdip = TTS_URL
    
    payload = {
        "speaker" : speaker,
        "text" : text,
        "text_language" : "ZH",
        "text_prompt" :"happy"
    }
    
    t0 = timeit()
    print(stdip, speaker, text)
    ret_response = requests.post(
        stdip, 
        data=json.dumps(payload), 
        timeout=timeout
    )
    t1 = timeit()
    print(ret_response)
    print('request_tts :', t1-t0)
    
    return ret_response



def request_asr(base64_audio=None, timeout=10, stdip=None):
    ## 注意： 这个本人自己搭建的api, 请替换成自己的api 或者 第三方api
    
    if stdip is None: stdip = ASR_URL
        
    ############ 模拟输入 bytes ############
    inputs_path = "misc/audio_transhbin/output-0526-v3.wav"
    with open(inputs_path, "rb") as audio_file: audio_bytes = audio_file.read()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    # print(base64_audio[:20])
    ############ 模拟输入 bytes ############
    
    payload = {
        "base64_audio" : base64_audio
    }
    
    t0 = timeit()
    ret_response = requests.post(
        stdip, 
        data=json.dumps(payload), 
        timeout=timeout
    )
    t1 = timeit()
    
    content_bytes = ret_response.content
    # 如果content是文本类型（如JSON、HTML等），可以解码为字符串
    if ret_response.encoding:
        asr_transcript = content_bytes.decode(ret_response.encoding)
    else:
        # 如果没有明确的编码，尝试使用UTF-8
        asr_transcript = content_bytes.decode('utf-8', errors='ignore')
    print(asr_transcript)
    print('t1 - t2 :', t1-t0)
    
    return asr_transcript


if __name__ == "__main__":
    # res = request_asr()
    # res = request_tts()
    
    # content=''
    # for response in create_chat_completion(query, use_stream=False):
    #     content += response
    # print(content)
    
    for response in create_chat_completion(query, use_stream=False): ...
    
    # response = create_chat_completion(query, use_stream=False) # response: <generator object create_chat_completion at 0x000001D7DD859DD0>
    print('response:', response)
    pass