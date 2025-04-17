
import os
import wave
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding
from riva.client import Auth
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("API_KEY")

# 將語音轉文字
def speech_to_text(audio_file_path):
    try:
        print(f"嘗試處理音頻檔案: {audio_file_path}")
        
        auth = riva.client.Auth(
            ssl_cert=None, 
            use_ssl=True,
            uri="grpc.nvcf.nvidia.com:443", 
            metadata_args=[
                ["function-id", "ee8dc628-76de-4acc-8595-1836e7e857bd"],
                ["authorization", "Bearer " + API_KEY],
            ]
        )

        asr_service = riva.client.ASRService(auth)
        
        # 讀取WAV文件獲取信息
        with wave.open(audio_file_path, 'rb') as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            print(f"WAV文件信息: 聲道={channels}, 位深度={width}, 採樣率={sample_rate}")
        
        # 創建配置，明確指定編碼
        config = riva.client.RecognitionConfig(
            language_code='en-US',
            max_alternatives=1,
            profanity_filter=False,
            enable_automatic_punctuation=True,
            verbatim_transcripts=True,
            enable_word_time_offsets=False
        )
        
        # 從文件讀取原始數據
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
        
        print(f"讀取的音頻數據大小: {len(audio_data)} 字節")
        print("發送請求到 Riva ASR 服務...")
        
        response = asr_service.offline_recognize(audio_data, config)
        
        if response and response.results:
            print(f"收到識別結果: {len(response.results)} 個結果")
            transcript = response.results[0].alternatives[0].transcript
            print(f"識別文本: '{transcript}'")
            return transcript
        else:
            print("收到空識別結果")
            return ""
    except Exception as e:
        print(f"語音識別出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"speech to text error: {str(e)}"

# 將文字轉語音
def text_to_speech(text):
    try:
        auth = Auth(ssl_cert=None, use_ssl=True,
                  uri="grpc.nvcf.nvidia.com:443", 
                  metadata_args=[
                      ["function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"],
                      ["authorization", "Bearer " + API_KEY],
                  ])
        
        service = riva.client.SpeechSynthesisService(auth)
        resp = service.synthesize(
            text, 
            voice_name="Magpie-Multilingual.EN-US.Female.Female-1",
            language_code="en-US",
            sample_rate_hz=44100,
            encoding=AudioEncoding.LINEAR_PCM
        )

        nchannels = 1
        sampwidth = 2
        sample_rate_hz = 44100

        audio_path = 'audio.wav'

        out_f = wave.open(audio_path, 'wb')
        out_f.setnchannels(nchannels)
        out_f.setsampwidth(sampwidth)
        out_f.setframerate(sample_rate_hz)
        out_f.writeframesraw(resp.audio)
        
        return audio_path
    except Exception as e:
        print(f"text to speech error: {str(e)}")
        return None
