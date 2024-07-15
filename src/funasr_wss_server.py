# ref: https://github.com/modelscope/FunASR/blob/main/runtime/python/http/server.py

import asyncio
import json
import websockets

import ssl
import base64, os
import time
import logging
import tracemalloc
import numpy as np
import traceback
from time import time as ttime


from funasr import AutoModel

from utils import create_chat_completion, system_round, \
    request_tts, save_bytes_as_wav
from argsparser import get_args
from clientsession import ClientSession


"""
CUDA_VISIBLE_DEVICES=2 /home/xhai/software/anaconda3/envs/zwq-asr-gpu/bin/python \
    src/funasr_wss_server.py \
    --port 10122 \
    --asr_model /data1/wqzh/HF-Models/FunASR-Chat/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
    --asr_model_online /data1/wqzh/HF-Models/FunASR-Chat/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online \
    --punc_model /data1/wqzh/HF-Models/FunASR-Chat/punc_ct-transformer_zh-cn-common-vocab272727-pytorch \
    --vad_model /data1/wqzh/HF-Models/FunASR-Chat/speech_fsmn_vad_zh-cn-16k-common-pytorch 

"""

# LLM_MODEL_NAME = "xxx-QW-7B-BF16-91R-xxx"
# LLM_MODEL_URL = "http://122.xxx.xxx:xxx/v1/chat/completions"
# TTS_URL = "http://10.101.10.12:8768/"
# TTS_URL = "http://10.101.10.12:18769/"


args = get_args()
print(args)


print("model loading")

def load_auto_model(model, model_revision):
    load_model = AutoModel(
        model=model,
        model_revision=model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )
    return load_model

model_asr = load_auto_model(model=args.asr_model, model_revision=args.asr_model_revision)
# model_asr_streaming = load_auto_model(model=args.asr_model_online, model_revision=args.asr_model_online_revision)
model_asr_streaming = None
model_vad = load_auto_model(model=args.vad_model, model_revision=args.vad_model_revision)
model_punc = load_auto_model(model=args.punc_model, model_revision=args.punc_model_revision)


# print("model loaded! only support one client at the same time now!!!!")
print("model loaded! Multi-clients are supported at the same time now!!!!")



async def client_handler(websocket, path):
    session = ClientSession(websocket)
    print('---> client_handler：', id(websocket), flush=True)
    
    websocket = session.websocket
    assert id(websocket)==id(session.websocket), "id(websocket) != id(session.websocket)"
    
    try:
        async for message in session.websocket:
            
            if isinstance(message, str):
                print('=======> Received meg: [str]')
                message_json = json.loads(message)
                
                if "chunk_size" in message_json:
                    ## `chunk_size`: 表明client microphone 第一次发送的配置信息
                    print('-----> Received meg: [found `chunk_size` in meg]', flush=True)
                    session.websocket = load_config(session, message_json)
                elif "audio_playing" in message_json:
                    ## `audio_playing`: 表明 client 正在 开始/结束 播放音频
                    print('-----> Received meg: [found `audio_playing` in meg]: ', message_json["audio_playing"], flush=True)
                    session.client_is_playing_audio = int(message_json["audio_playing"])
                
                
            ## 【有新的bytes音频数据】 或者 【有历史数据】
            if not isinstance(message, str) or len(session.frames_asr) > 0 or len(session.frames_asr_online) > 0:
                if not isinstance(message, str):  # bytes 类型， 表明是音频数据
                    
                    ## 如何客户端正在播放音频，直接忽略掉 这些音频
                    if session.client_is_playing_audio:
                        continue
                    
                    session.frames.append(message)
                    duration_ms = len(message) // 32    # 60 ms (len*1000/(16000*2))
                    session.websocket.vad_pre_idx += duration_ms
                    session.current_time += duration_ms   # 当前时间
                    
                    
                    ############  delete asr online operation
                    ## 这里可以添加： 使用实时asr， 并将文本返回给用户
                    
                    
                    if session.speech_start: session.frames_asr.append(message)  # frames_asr【从头开始记录】
                    
                    ### vad online
                    try:
                        ttt1 = ttime()
                        speech_start_i, speech_end_i = await async_vad(session, message)
                        # print('==== VAD', ttime()-ttt1, flush=True)  # 3 ~10 ms
                        
                        if speech_start_i != -1 or speech_end_i != -1:
                            print('-----> speech_start_i, speech_end_i:', speech_start_i, speech_end_i)
                        session.speech_start_i, session.speech_end_i = speech_start_i, speech_end_i
                    except Exception as e:
                        print(f"error in vad : {e}")
                    
                    if session.speech_start_i != -1:     ## 检测到 【开始】
                        session.speech_start = True
                        beg_bias = (session.websocket.vad_pre_idx - session.speech_start_i) // duration_ms
                        session.frames_pre = session.frames[-beg_bias:]
                        session.frames_asr = []
                        session.frames_asr.extend(session.frames_pre)
                
                ### delete asr punc offline   # ref to v6
                
                
                if session.speech_end_i != -1:
                    session.speech_start = False
                    ## 记录当前最后一次 【停顿时间】
                    session.last_inactive_time = session.current_time
                    session.last_inactive_end_i = session.speech_end_i
                    print('====> speech_end_i', session.speech_end_i, 'current_time', session.current_time)
                
                
                
                ### 判断idle time
                if session.last_inactive_time != -1 and \
                    session.current_time - session.last_inactive_time > session.IDLE_TIME:
                    audio_in = b"".join(session.frames_asr)
                    
                    print('===========> idle timeout transcribe!')
                    if len(audio_in) > 0:
                        try:  ## 2pass-offline asr
                            # punc_result = await async_asr(websocket, audio_in, speech_end_i)
                            tic_asr =  ttime()
                            punc_result = await async_asr(session, audio_in, session.last_inactive_end_i)
                            print('----> async asr: ', round(ttime()-tic_asr, 2), flush=True)
                            
                        except Exception as e:
                            print(f"error in asr offline: {e}")
                        
                        if len(punc_result) > 0:  ## 带标点文本长度
                            print('###############---->:', punc_result, session.last_inactive_time, 
                                session.current_time, session.current_time-session.last_inactive_time)
                        else:
                            print('###############---->: [empty] punc_result', session.last_inactive_time, 
                                session.current_time, session.current_time-session.last_inactive_time)
                            print('---> continue')
                            
                            session.last_inactive_time = -1
                            continue
                    
                    
                    tic_llm_tts = ttime()
                    await llm_and_tts(session, punc_result)
                    print('=====> [ llm_and_tts] : ', round(ttime()-tic_llm_tts, 2), flush=True)
                    
                    #################  暂时不考虑  说话期间被打断, 可以调整 await llm_and_tts 的顺序
                    ###   重置 【静默检测时间】
                    session.last_inactive_time = -1
                    # websocket.vad_pre_idx = 0    ## 否则会从0 开始计数
                    session.frames = []
                    session.frames_asr = []
                    session.websocket.status_dict_vad["cache"] = {}
                    session.websocket.status_dict_asr_online["cache"] = {}
                    session.websocket.status_dict_punc["cache"] = {}
                    
                    print("===> llm_and_tts done")
                    
                    
        
    except session.websocket.ConnectionClosed:
        print("ConnectionClosed... Session:", id(session), flush=True)
        ws_reset(session)
        # del session
    except session.websocket.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("??????? Exception:", e)
    

async def ws_reset(session):
    session.websocket.status_dict_asr_online["cache"] = {}
    session.websocket.status_dict_asr_online["is_final"] = True
    session.websocket.status_dict_vad["cache"] = {}
    session.websocket.status_dict_vad["is_final"] = True
    session.websocket.status_dict_punc["cache"] = {}

    await session.websocket.close()


async def async_vad(session, audio_in):
    segments_result = model_vad.generate(input=audio_in, **session.websocket.status_dict_vad)[0]["value"]

    speech_start, speech_end = -1, -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end



async def async_asr(session, audio_in, speech_end_i):
    if len(audio_in) > 0:
        rec_result = model_asr.generate(input=audio_in, **session.websocket.status_dict_asr)[0]
        if model_punc is not None and len(rec_result["text"]) > 0:
            rec_result = model_punc.generate(
                input=rec_result["text"], **session.websocket.status_dict_punc
            )[0]
        if len(rec_result["text"]) > 0:
            mode = "2pass-offline" if "2pass" in session.websocket.mode else session.websocket.mode
            message = json.dumps(
                {
                    'type': 'text',
                    # "src": 'async_asr',
                    "src": 'async_asr_offline_idle',
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": session.websocket.wav_name,
                    "is_final": session.websocket.is_speaking,
                    # "speech_start_i": speech_start_i, 
                    "speech_end_i": speech_end_i
                }
            )
            await session.websocket.send(message)
            return rec_result["text"]
        
        return ''

async def async_asr_online(session, audio_in, speech_end_i):
    if len(audio_in) > 0:
        rec_result = model_asr_streaming.generate(
            input=audio_in, **session.websocket.status_dict_asr_online
        )[0]
        if session.websocket.mode == "2pass" and session.websocket.status_dict_asr_online.get("is_final", False):
            return
        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in session.websocket.mode else session.websocket.mode
            message = json.dumps(
                {
                    'type': 'text',
                    "src": 'async_asr_online',
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": session.websocket.wav_name,
                    "is_final": session.websocket.is_speaking,
                    # "speech_start_i": speech_start_i, 
                    "speech_end_i": speech_end_i
                }
            )
            await session.websocket.send(message)

def load_config(session, messagejson):
    print('---> `load_config` websocket id is: ', id(session.websocket), flush=True)
    websocket = session.websocket
    assert id(websocket) == id(session.websocket), "session.websocket is not the same as websocket"
    
    
    if "is_speaking" in messagejson:
        websocket.is_speaking = messagejson["is_speaking"]
        websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
    if "chunk_interval" in messagejson:
        websocket.chunk_interval = messagejson["chunk_interval"]
    if "wav_name" in messagejson:
        websocket.wav_name = messagejson.get("wav_name")
    if "chunk_size" in messagejson:
        chunk_size = messagejson["chunk_size"]
        if isinstance(chunk_size, str):
            chunk_size = chunk_size.split(",")
        websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
    if "encoder_chunk_look_back" in messagejson:
        websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson[
            "encoder_chunk_look_back"
        ]
    if "decoder_chunk_look_back" in messagejson:
        websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson[
            "decoder_chunk_look_back"
        ]
    if "hotword" in messagejson:
        websocket.status_dict_asr["hotword"] = messagejson["hotword"]
    if "mode" in messagejson:
        websocket.mode = messagejson["mode"]
    
    websocket.status_dict_vad["chunk_size"] = int(
        websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
    )
    
    ## JavaScript code
    # var chunk_size = new Array( 5, 10, 5 );
    # var request = {
    #     "chunk_size": chunk_size,
    #     "wav_name":  "h5",
    #     "is_speaking":  true,
    #     "chunk_interval":10,
    #     "itn":getUseITN(),
    #     "mode":getAsrMode(),
    # };
    
    # websocket.status_dict_asr_online["chunk_size"] = [5, 10, 5]
    # websocket.chunk_interval = 10
    # websocket.status_dict_vad["chunk_size"] = int(
    #     websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
    # )
    # websocket.mode = '2pass'
    # websocket.is_speaking = True
    # websocket.wav_name = 'h5'
    
    return websocket



async def llm_and_tts(session, prompt):
    print('----> llm_and_tts [prompt]', prompt, flush=True)
    
    ##############  LLM operation  ##############
    tic = time.time()
    messages=[
        {"role": "system", "content": system_round},
        {"role": "user", "content": prompt}
    ]
    for response in create_chat_completion(
        messages,
        use_stream=False,
        # use_stream=True,
        
        ###  Tip: 替换模型名称和地址
        # model_name="xxxxx",
        # llm_url="http://xxx.xxx.xxx.181:xxx/v1/chat/completions"
    ): ...
    toc = time.time()
    
    await send_text_to_client(
        session.websocket,
        json.dumps({
            'type': 'text',
            'src': 'llm_response',
            'text': response
        })
    )
    tictoc = time.time()
    print('---> send [text] done!', f"【{response}】")
    print('LLM: 生成：', round(toc-tic, 2), 's', ' | 传输：', round(tictoc-toc, 2), 's')
    
    ## response 是空的话，直接返回
    if response == '':  return 
    
    ##############  TTS operation  ##############
    tic_tts = time.time()
    tts_ret = request_tts(text=response)
    toc_tts = time.time()
    
    
    await send_audio_to_client_bytes(session.websocket, tts_ret.content, audio_format='mp3')
    tictoc_tts = time.time()
    
    print('---> send [audio] done!')
    print('TTS: 生成：', round(toc_tts-tic_tts, 2), 's', ' | 传输：', round(tictoc_tts-toc_tts, 2), 's')




async def send_text_to_client(text_client_websocket, message):
    try:
        print('---> server `send_text_to_client`: text_client_websocket is None', 
            text_client_websocket is None)

        if text_client_websocket:
            await text_client_websocket.send(message)
        else:
            print("!!!!!! > text_client_websocket is None")
    except Exception as e:
        print(f"发送text到客户端失败：send_text_to_client: {str(e)}")


async def send_audio_to_client_bytes(
        audio_client_websocket, 
        audio_data, 
        audio_format='wav'
    ):

    audio_data_base64 = base64.b64encode(audio_data).decode('utf-8')
    message = json.dumps({
        'type': 'audio', 
        'format': audio_format,
        'audioData': audio_data_base64
    })

    # 通过WebSocket发送消息
    try:
        if audio_client_websocket:
            await audio_client_websocket.send(message)
        else:
            print("!!!!!! > audio_client_websocket is None")
    except Exception as e:
        print(f"发送audio到客户端失败：send_audio_to_client_bytes: {str(e)}")


if __name__ == "__main__":
    start_server = websockets.serve(client_handler, args.host, args.port)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
