# ref: https://github.com/modelscope/FunASR/blob/main/runtime/python/http/client.py

""" client requirements.txt

pip install websockets pygame
pip install pyaudio


python src/funasr_wss_client.py  --host 122.xxx.xxx.xxx --port 10122  --mode 2pass --chunk_size "5,10,5" --output_dir ./ --ssl 0

"""

import os
import time
import websockets, ssl
import asyncio
import io
import pygame
import base64

# import threading
import argparse
import json
from multiprocessing import Process

import traceback

import logging
logging.basicConfig(level=logging.ERROR)

import pyaudio


CLS_FORMAT = 'clear' # Linux， 
# CLS_FORMAT = 'cls' # Windows


start_cache, end_cache = 0, -1


parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="localhost", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10094, required=False, help="grpc server port")
parser.add_argument("--text_port", type=int, default=10095, required=False, help="grpc server text port")
parser.add_argument("--audio_port", type=int, default=10096, required=False, help="grpc server audio port")
parser.add_argument("--chunk_size", type=str, default="5, 10, 5", help="chunk")
parser.add_argument("--encoder_chunk_look_back", type=int, default=4, help="chunk")
parser.add_argument("--decoder_chunk_look_back", type=int, default=0, help="chunk")
parser.add_argument("--chunk_interval", type=int, default=10, help="chunk")
parser.add_argument(
    "--hotword",
    type=str,
    default="",
    help="hotword file path, one hotword perline (e.g.:阿里巴巴 20)",
)
parser.add_argument("--audio_in", type=str, default=None, help="audio_in")
parser.add_argument("--audio_fs", type=int, default=16000, help="audio_fs")
parser.add_argument(
    "--send_without_sleep",
    action="store_true",
    default=True,
    help="if audio_in is set, send_without_sleep",
)
parser.add_argument("--thread_num", type=int, default=1, help="thread_num")
parser.add_argument("--words_max_print", type=int, default=10000, help="chunk")
parser.add_argument("--output_dir", type=str, default=None, help="output_dir")
# parser.add_argument("--ssl", type=int, default=1, help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--ssl", type=int, default=1, help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--use_itn", type=int, default=1, help="1 for using itn, 0 for not itn")
parser.add_argument("--mode", type=str, default="2pass", help="offline, online, 2pass")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)
# voices = asyncio.Queue()
from queue import Queue

IS_PLAYING_AUDIO = False

voices = Queue()
offline_msg_done = False
cur_time = 0

text_print = ""
text_print_2pass_online = ""
text_print_2pass_offline = ""

llm_cache_text = ""
LLM_INPUT_CACHE = []

if args.output_dir is not None:
    ibest_writer = open(
        # os.path.join(args.output_dir, "text.{}".format(id)), "a", encoding="utf-8"
        os.path.join(args.output_dir, "text.{}.log".format(20240618)), "a", encoding="utf-8"
    )
else:
    ibest_writer = None


if args.output_dir is not None:
    # if os.path.exists(args.output_dir):
    #     os.remove(args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


async def record_microphone():
    is_finished = False

    global voices
    global cur_time
    global text_print_2pass_offline, end_cache
    
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword)
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    print("Please checkout format of hotwords")
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    print("Please checkout format of hotwords")
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = args.hotword

    use_itn = True
    if args.use_itn == 0:
        use_itn = False

    message = json.dumps(
        {
            "mode": args.mode,
            "chunk_size": args.chunk_size,
            "chunk_interval": args.chunk_interval,
            "encoder_chunk_look_back": args.encoder_chunk_look_back,
            "decoder_chunk_look_back": args.decoder_chunk_look_back,
            "wav_name": "microphone",
            "is_speaking": True,
            "hotwords": hotword_msg,
            "itn": use_itn,
        }
    )
    # voices.put(message)
    await websocket.send(message)
    while True:
        data = stream.read(CHUNK)
        message = data
        # voices.put(message)
        await websocket.send(message)

        # await asyncio.sleep(0.005)
        await asyncio.sleep(0.01)


async def record_from_scp(chunk_begin, chunk_size):
    pass


async def message(id):
    global websocket, voices, offline_msg_done
    global start_cache, end_cache 
    global cur_time
    
    global text_print, text_print_2pass_online, text_print_2pass_offline
    global llm_cache_text, LLM_INPUT_CACHE
    global ibest_writer
    
    try:
        while True:
            meg = await websocket.recv()
            meg = json.loads(meg)
            
            ###  LLM 生成的文本返回
            if 'src' in meg and meg['src'] == 'llm_response':
                print('--> llm response', meg["text"])
                continue
            
            
            ###  播放音频
            data_type = meg.get("type", '')
            if data_type == 'audio':
                try:
                    audio_data = io.BytesIO(base64.b64decode(meg["audioData"]))
                    
                    # IS_PLAYING_AUDIO = True    ## 正在播放音频, 通知服务器，可忽略这期间的语音输入
                    # 因为存在回声混合， 暂时不支持用户中途打断。 所以，需要等待音频播放结束
                    await websocket.send(json.dumps({"audio_playing": "1"}))
                    
                    await play_audio(audio_data)
                    
                    # IS_PLAYING_AUDIO = False   ## 播放音频结束
                    await websocket.send(json.dumps({"audio_playing": "0"}))
                    
                except Exception as e:
                    print(f'Exception {e}')
                    traceback.print_exc()
                continue
            
            
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]
            
            offline_msg_done = meg.get("is_final", False)

            if "mode" not in meg: continue
            if meg["mode"] == "online":
                raise KeyError('Mode : online')
            elif meg["mode"] == "offline":
                raise KeyError('Mode : offline')
            else:
                if meg["mode"] == "2pass-online":
                    text_print_2pass_online += "{}".format(text)
                    text_print = '[2pass-online]: '+text_print_2pass_online
                    
                else:    
                    text_print = text
                    ibest_writer.write(" [2pass-offline] - {}\t{} \n".format(wav_name, text_print))
                    print('--------> ###########  [llm_cache_text] 2pass-offline', text_print)
                
                # os.system("clear")
                os.system(CLS_FORMAT)
                
                print("\rpid" + str(id) + ": " + text_print)
                # offline_msg_done=True

    except Exception as e:
        print("Exception:", e)
        # traceback.print_exc()
        # await websocket.close()


async def ws_client(id, chunk_begin, chunk_size):
    if args.audio_in is None:
        chunk_begin = 0
        chunk_size = 1
    global websocket, voices, offline_msg_done

    for i in range(chunk_begin, chunk_begin + chunk_size):
        offline_msg_done = False
        voices = Queue()
        if args.ssl == 1:
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            uri = "wss://{}:{}".format(args.host, args.port)
        else:
            uri = "ws://{}:{}".format(args.host, args.port)
            ssl_context = None
        print("====> [OK] connect to", uri)
        
        async with websockets.connect(
            uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ) as websocket:
            task = asyncio.create_task(record_microphone())
            task3 = asyncio.create_task(message(str(id) + "_" + str(i)))  # processid+fileid
            await asyncio.gather(task, task3)
    
    print('---> async def ws_text_client:  exit(0)')


async def play_audio(audio_data):
    
    pygame.mixer.init()
    pygame.mixer.music.load(audio_data)
    pygame.mixer.music.play()

    # 等待音频播放完成
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.5)


def one_thread(id, chunk_begin, chunk_size):
    asyncio.get_event_loop().run_until_complete(ws_client(id, chunk_begin, chunk_size))
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    # for microphone
    if args.audio_in is None:
        p = Process(target=one_thread, args=(0, 0, 0))
        p.start()
        p.join()
        print("end")
    else:
        # calculate the number of wavs for each preocess
        print("no support wav-file input")

