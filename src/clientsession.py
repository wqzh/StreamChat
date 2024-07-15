

import asyncio
import websockets


class ClientSession:
    """
    这个类用于存储每个客户端的变量，每个客户端连接到服务器时，都会创建一个实例。因此可以实现不同客户端临时数据之间的隔离。
    """
    def __init__(self, websocket):
        self.websocket = websocket
        # 创建一个字典来存储每个客户端的变量
        # self.client_variables = {}
        
        # 音频片段列表
        self.frames = []
        self.frames_asr = []
        self.frames_asr_online = []
        self.frames_pre = []
        
        # vad 状态, 说话开始、结束 时间戳
        self.speech_start = False
        self.speech_start_i, self.speech_end_i = -1, -1
        
        # 最后一次 vad 检测活跃时间
        self.last_inactive_time = -1
        self.current_time = -1
        self.last_inactive_end_i = -1
        self.IDLE_TIME= 800  # 静默检测时间 800ms
        # self.IDLE_TIME= 540  # 静默检测时间 
        
        self.client_is_playing_audio = False
        
        self.websocket.status_dict_asr = {}
        self.websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
        self.websocket.status_dict_vad = {"cache": {}, "is_final": False}
        self.websocket.status_dict_punc = {"cache": {}}
        self.websocket.chunk_interval = 10
        self.websocket.vad_pre_idx = 0
        
        self.websocket.wav_name = "microphone"
        self.websocket.mode = "2pass"
        # print("new user connected", flush=True)
        print("=======> [OK]: handler Client connected", flush=True)
    

    async def send_message(self, message):
        await self.websocket.send(message)

    async def receive_message(self):
        return await self.websocket.recv()

    async def close(self):
        await self.websocket.close()



async def handle_client_demo(websocket, path):
    session = ClientSession(websocket)
    try:
        while True:
            message = await session.receive_message()
            # 在这里处理消息并使用session.client_variables
            # 示例：将接收到的消息存储在变量中
            session.client_variables['received_message'] = message
            # 将处理后的消息发送回客户端
            await session.send_message("Message received")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await session.close()


if __name__ == "__main__":
    
    start_server = websockets.serve(handle_client_demo, "localhost", 8765)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()