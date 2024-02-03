import io
import logging

import librosa
import soundfile
from flask import Flask, request, send_file
from flask_cors import CORS

from infer_tools.infer_tool import Svc
from utils.hparams import hparams

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)

    f_pitch_change = float(request_form.get("fPitchChange", 0))
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    input_wav_path = io.BytesIO(wave_file.read())
    _f0_tst, _f0_pred, _audio = model.infer(input_wav_path, key=f_pitch_change, acc=accelerate, use_pe=False,
                                            use_crepe=False)
    tar_audio = librosa.resample(_audio, hparams["audio_sample_rate"], daw_sample)
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, daw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    project_name = "firefox"
    model_path = f'./checkpoints/{project_name}/model_ckpt_steps_188000.ckpt'
    config_path = f'./checkpoints/{project_name}/config.yaml'

    accelerate = 50
    hubert_gpu = True

    model = Svc(project_name, config_path, hubert_gpu, model_path)

    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
