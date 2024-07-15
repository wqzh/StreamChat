# ref: https://github.com/modelscope/FunASR/blob/main/runtime/python/http/server.py



"""
arguments parser for server
"""

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
    )
    parser.add_argument("--port", type=int, default=10094, required=False, help="grpc server port")
    parser.add_argument("--text_port", type=int, default=10095, required=False, help="grpc server text port")
    parser.add_argument("--audio_port", type=int, default=10096, required=False, help="grpc server audio port")
    parser.add_argument(
        "--asr_model",
        type=str,
        default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        help="model from modelscope",
    )
    parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="")
    parser.add_argument(
        "--asr_model_online",
        type=str,
        default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        help="model from modelscope",
    )
    parser.add_argument("--asr_model_online_revision", type=str, default="v2.0.4", help="")
    parser.add_argument(
        "--vad_model",
        type=str,
        default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        help="model from modelscope",
    )
    parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")
    parser.add_argument(
        "--punc_model",
        type=str,
        default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        help="model from modelscope",
    )
    parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")
    parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
    parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
    parser.add_argument(
        "--certfile",
        type=str,
        # default="../../ssl_key/server.crt",
        default="",
        # default="../ssl_key/server.crt",
        required=False,
        help="certfile for ssl",
    )

    parser.add_argument(
        "--keyfile",
        type=str,
        # default="../../ssl_key/server.key",
        default="",
        # default="../ssl_key/server.key",
        required=False,
        help="keyfile for ssl",
    )
    args = parser.parse_args()
    
    return args

