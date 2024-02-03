import gradio as gr
from pytube import YouTube
from moviepy.editor import *
from pydub import AudioSegment
import os
from glob import glob
from tqdm import tqdm
from spleeter.separator import Separator

separator = Separator('spleeter:2stems')

def split_mp3_to_wav(file_path, save_path,artist_name):
    segment_length=15000

    # MP3 파일 로드
    audio = AudioSegment.from_mp3(file_path)

    # 오디오 길이 계산 (밀리초 단위)
    length_audio = len(audio)
    pbar = tqdm(total=length_audio, desc=f'{artist_name} Processing')

    # 15초 간격으로 나누기
    start = 0
    end = segment_length
    part = 1

    while start < length_audio:
        # 오디오 잘라내기
        chunk = audio[start:end]
        chunk.export(os.path.join(save_path,f'{artist_name}{part}.wav'), format='wav')

        # 다음 청크로 이동
        start += segment_length
        end += segment_length
        part += 1
        pbar.update(segment_length)

def preprocessing_and_training(artist_name, youtube_link):
    save_dir = "data/"+artist_name.replace(" ","_")

    # Download the video from the link
    if not os.path.exists(f'./{save_dir}'):
        os.makedirs(f'./{save_dir}')
    origin_mp3_path = f'./{save_dir}/origin.mp3'

    yt = YouTube(youtube_link)
    video_path = yt.streams.filter(only_audio=True).first().download()
    AudioFile = AudioFileClip(video_path)
    AudioFile.write_audiofile(origin_mp3_path)
    AudioFile.close()
    os.remove(video_path)

    # Split the mp3 file into 15 seconds
    if not os.path.exists(f'./{save_dir}/split'):
        os.makedirs(f'./{save_dir}/split')
    split_mp3_to_wav(origin_mp3_path,f'./{save_dir}/split',artist_name)

    # Split the audio file into vocal and accompaniment
    if not os.path.exists(f'./{save_dir}/tmp'):
        os.makedirs(f'./{save_dir}/tmp')

    for audio_path in tqdm(glob(os.path.join(f'./{save_dir}/split',"*.wav"))):
        separator.separate_to_file(audio_path, f'./{save_dir}/tmp')

    # Convert the audio files to 44100Hz, mono, 16-bit
    sample_rate = 44100
    if not os.path.exists(f'./{save_dir}/vocal'):
        os.makedirs(f'./{save_dir}/vocal')

    cnt = 0
    for file_path in tqdm(glob(os.path.join(f'./{save_dir}/tmp',"*/vocals.wav"))):
        audio = AudioSegment.from_wav(file_path)
        converted = audio.set_frame_rate(sample_rate)
        converted = converted.set_channels(1)
        converted = converted.set_sample_width(2)

        converted.export(os.path.join(f'./{save_dir}/vocal',f'{artist_name}{cnt}.wav'), format='wav')
        cnt = cnt + 1

    os.system(f"rm -rf ./{save_dir}/split")
    os.system(f"rm -rf ./{save_dir}/tmp")

    with open(f'config_nsf.yaml', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i].replace("${artist_name}",artist_name)
        
    with open(f'./{save_dir}/config.yaml', 'w') as f:
        f.writelines(lines)

    os.system(f"export PYTHONPATH=./diff-svc")
    os.system(f"CUDA_VISIBLE_DEVICES=0 python ./diff-svc/preprocessing/binarize.py --config ./{save_dir}/config.yaml")

    os.system(f"CUDA_VISIBLE_DEVICES=0 python ./diff-svc/infer.py --config ./{save_dir}/config.yaml  --exp_name {artist_name} --reset")
    return "Done!"

def infer(artist_name, files):
    model_paths = glob(f'./data/{artist_name}/checkpoints/*.ckpt')
    if len(model_paths) == 0:
        return "No model found for this artist"
    model_path = model_paths.sort(reverse=True)[0]
    config_path = f'./data/{artist_name}/config.yaml'
    os.system("rm ./diff-svc/raw/*")
    for file in files:
        shutil.cp(file.value, f'./diff-svc/raw/audio/{file.value.split("/")[-1]}')

    os.system(f"export PYTHONPATH=./diff-svc")
    os.system(f"CUDA_VISIBLE_DEVICES=0 python ./diff-svc/infer.py --config {config_path} --model_path {model_path} --artist_name {artist_name}")
    os.system(f"zip -r ./data/{artist_name}/results.zip ./diff-svc/results")
    
    return "Done!"

train = gr.Interface(fn=preprocessing_and_training, inputs=["text","text"], outputs="text")
infer = gr.Interface(fn=infer, inputs=[gr.Dropdown(os.listdir('./data')),"files"], outputs=["text"])

demo = gr.TabbedInterface([train, infer], ["Welcome page", "Visualization page"])
if __name__ == "__main__":
    demo.launch(share=True)
