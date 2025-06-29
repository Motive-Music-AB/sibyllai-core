import os
import shutil
import json
import torch
import torchaudio
import numpy as np
import logging
import warnings
import subprocess
import math
import random
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import mir_eval
import pretty_midi as pm
import gradio as gr
from gradio import Markdown
from music21 import converter
import torchaudio.transforms as T

# Custom utility imports
from utils import logger
from utils.btc_model import BTC_model
from utils.transformer_modules import *
from utils.transformer_modules import _gen_timing_signal, _gen_bias_mask
from utils.hparams import HParams
from utils.mir_eval_modules import (
    audio_file_to_features, idx2chord, idx2voca_chord,
    get_audio_paths, get_lab_paths
)
from utils.mert import FeatureExtractorMERT
from model.linear_mt_attn_ck import FeedforwardModelMTAttnCK

# Suppress unnecessary warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# from gradio import Markdown

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

pitch_num_dic = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

minor_major_dic = {
    'D-':'C#', 'E-':'D#', 'G-':'F#', 'A-':'G#', 'B-':'A#'
}
minor_major_dic2 = {
    'Db':'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#'
}

shift_major_dic = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

shift_minor_dic = {
    'A': 0, 'A#': 1, 'B': 2, 'C': 3, 'C#': 4, 'D': 5,  
    'D#': 6, 'E': 7, 'F': 8, 'F#': 9, 'G': 10, 'G#': 11, 
}

flat_to_sharp_mapping = {
    "Cb": "B", 
    "Db": "C#", 
    "Eb": "D#", 
    "Fb": "E", 
    "Gb": "F#", 
    "Ab": "G#", 
    "Bb": "A#"
}

segment_duration = 30
resample_rate = 24000
is_split = True

def normalize_chord(file_path, key, key_type='major'):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if key == "None":
        new_key = "C major"
        shift = 0
    else:
        #print ("asdas",key)
        if len(key) == 1:
            key = key[0].upper()
        else:
            key = key[0].upper() + key[1:]

        if key in minor_major_dic2:
            key = minor_major_dic2[key]
        
        shift = 0
        
        if key_type == "major":
            new_key = "C major"
            
            shift = shift_major_dic[key]
        else:
            new_key = "A minor"
            shift = shift_minor_dic[key]
    
    converted_lines = []
    for line in lines:
        if line.strip():  # Skip empty lines
            parts = line.split()
            start_time = parts[0]
            end_time = parts[1]
            chord = parts[2]  # The chord is in the 3rd column
            if chord == "N":
                newchordnorm = "N"
            elif chord == "X":
                newchordnorm = "X"
            elif ":" in chord:
                pitch = chord.split(":")[0]
                attr = chord.split(":")[1]
                pnum = pitch_num_dic [pitch]
                new_idx = (pnum - shift)%12
                newchord = PITCH_CLASS[new_idx]
                newchordnorm = newchord + ":" + attr
            else:
                pitch = chord
                pnum = pitch_num_dic [pitch]
                new_idx = (pnum - shift)%12
                newchord = PITCH_CLASS[new_idx]
                newchordnorm = newchord
            
            converted_lines.append(f"{start_time} {end_time} {newchordnorm}\n")
    
    return converted_lines

def sanitize_key_signature(key):
    return key.replace('-', 'b')

def resample_waveform(waveform, original_sample_rate, target_sample_rate):
    if original_sample_rate != target_sample_rate:
        resampler = T.Resample(original_sample_rate, target_sample_rate)
        return resampler(waveform), target_sample_rate
    return waveform, original_sample_rate

def split_audio(waveform, sample_rate):
    segment_samples = segment_duration * sample_rate
    total_samples = waveform.size(0)

    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        if end <= total_samples:
            segment = waveform[start:end]
            segments.append(segment)
    
    # In case audio length is shorter than segment length.
    if len(segments) == 0: 
        segment = waveform
        segments.append(segment)

    return segments



class Music2emo:
    def __init__(
        self,
        name="amaai-lab/music2emo",
        device="cuda:0",
        cache_dir=None,
        local_files_only=False,
    ):
        
        # use_cuda = torch.cuda.is_available()
        # self.device = torch.device("cuda" if use_cuda else "cpu")
        model_weights = "saved_models/J_all.ckpt"
        self.device = device

        self.feature_extractor = FeatureExtractorMERT(model_name='m-a-p/MERT-v1-95M', device=self.device, sr=resample_rate)
        self.model_weights = model_weights

        self.music2emo_model = FeedforwardModelMTAttnCK(
            input_size= 768 * 2,
            output_size_classification=56,
            output_size_regression=2
        )

        checkpoint = torch.load(self.model_weights, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        
        # Adjust the keys in the state_dict
        state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
        
        # Filter state_dict to match model's keys
        model_keys = set(self.music2emo_model.state_dict().keys())
        filtered_state_dict = {key: value for key, value in state_dict.items() if key in model_keys}
        
        # Load the filtered state_dict and set the model to evaluation mode
        self.music2emo_model.load_state_dict(filtered_state_dict)
        
        self.music2emo_model.to(self.device)
        self.music2emo_model.eval()

    def predict(self, audio, threshold = 0.5):

        feature_dir = Path("./inference/temp_out")
        output_dir = Path("./inference/output")
        
        if feature_dir.exists():
            shutil.rmtree(str(feature_dir))
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        
        feature_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        warnings.filterwarnings('ignore')
        logger.logging_verbosity(1)
        
        mert_dir = feature_dir / "mert"
        mert_dir.mkdir(parents=True)
        
        waveform, sample_rate = torchaudio.load(audio)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0)
        waveform = waveform.squeeze()
        waveform, sample_rate = resample_waveform(waveform, sample_rate, resample_rate)
        
        if is_split:        
            segments = split_audio(waveform, sample_rate)
            for i, segment in enumerate(segments):
                segment_save_path = os.path.join(mert_dir, f"segment_{i}.npy")
                self.feature_extractor.extract_features_from_segment(segment, sample_rate, segment_save_path)
        else:
            segment_save_path = os.path.join(mert_dir, f"segment_0.npy")
            self.feature_extractor.extract_features_from_segment(waveform, sample_rate, segment_save_path)

        embeddings = []
        layers_to_extract = [5,6]
        segment_embeddings = []
        for filename in sorted(os.listdir(mert_dir)):  # Sort files to ensure sequential order
            file_path = os.path.join(mert_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.npy'):
                segment = np.load(file_path)
                concatenated_features = np.concatenate(
                    [segment[:, layer_idx, :] for layer_idx in layers_to_extract], axis=1
                )
                concatenated_features = np.squeeze(concatenated_features)  # Shape: 768 * 2 = 1536
                segment_embeddings.append(concatenated_features)

        segment_embeddings = np.array(segment_embeddings)
        if len(segment_embeddings) > 0:
            final_embedding_mert = np.mean(segment_embeddings, axis=0)
        else:
            final_embedding_mert = np.zeros((1536,))

        final_embedding_mert = torch.from_numpy(final_embedding_mert)
        final_embedding_mert.to(self.device)

        # --- Chord feature extract ---
        config = HParams.load("./inference/data/run_config.yaml")
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        model_file = './inference/data/btc_model_large_voca.pt'
        idx_to_chord = idx2voca_chord()
        model = BTC_model(config=config.model).to(self.device)

        if os.path.isfile(model_file):
            checkpoint = torch.load(model_file)
            mean = checkpoint['mean']
            std = checkpoint['std']
            model.load_state_dict(checkpoint['model'])

        audio_path = audio
        audio_id = audio_path.split("/")[-1][:-4]
        try:
            feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        except:
            logger.info("audio file failed to load : %s" % audio_path)
            assert(False)
            
        logger.info("audio file loaded and feature computation success : %s" % audio_path)
        
        feature = feature.T
        feature = (feature - mean) / std
        time_unit = feature_per_second
        n_timestep = config.model['timestep']

        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

        start_time = 0.0
        lines = []
        with torch.no_grad():
            model.eval()
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            for t in range(num_instance):
                self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                prediction, _ = model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        continue
                    if prediction[i].item() != prev_chord:
                        lines.append(
                            '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction[i].item()
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != time_unit * (n_timestep * t + i):
                            lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        break

        save_path = os.path.join(feature_dir, os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '.lab')
        with open(save_path, 'w') as f:
            for line in lines:
                f.write(line)

        # logger.info("label file saved : %s" % save_path)

        # lab file to midi file
        starts, ends, pitchs = list(), list(), list()

        intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
        for p in range(12):
            for i, (interval, chord) in enumerate(zip(intervals, chords)):
                root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
                tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
                if i == 0:
                    start_time = interval[0]
                    label = tmp_label
                    continue
                if tmp_label != label:
                    if label == 1.0:
                        starts.append(start_time), ends.append(interval[0]), pitchs.append(p + 48)
                    start_time = interval[0]
                    label = tmp_label
                if i == (len(intervals) - 1): 
                    if label == 1.0:
                        starts.append(start_time), ends.append(interval[1]), pitchs.append(p + 48)

        midi = pm.PrettyMIDI()
        instrument = pm.Instrument(program=0)

        for start, end, pitch in zip(starts, ends, pitchs):
            pm_note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
            instrument.notes.append(pm_note)

        midi.instruments.append(instrument)
        midi.write(save_path.replace('.lab', '.midi'))

        tonic_signatures = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        mode_signatures = ["major", "minor"]  # Major and minor modes

        tonic_to_idx = {tonic: idx for idx, tonic in enumerate(tonic_signatures)}
        mode_to_idx = {mode: idx for idx, mode in enumerate(mode_signatures)}
        idx_to_tonic = {idx: tonic for tonic, idx in tonic_to_idx.items()}
        idx_to_mode = {idx: mode for mode, idx in mode_to_idx.items()}

        with open('inference/data/chord.json', 'r') as f:
            chord_to_idx = json.load(f)
        with open('inference/data/chord_inv.json', 'r') as f:
            idx_to_chord = json.load(f)
            idx_to_chord = {int(k): v for k, v in idx_to_chord.items()}  # Ensure keys are ints        
        with open('inference/data/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('inference/data/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        try:
            midi_file = converter.parse(save_path.replace('.lab', '.midi'))
            key_signature = str(midi_file.analyze('key'))
        except Exception as e:
            key_signature = "None"

        key_parts = key_signature.split()
        key_signature = sanitize_key_signature(key_parts[0])  # Sanitize key signature
        key_type = key_parts[1] if len(key_parts) > 1 else 'major'

        # --- Key feature (Tonic and Mode separation) --- 
        if key_signature == "None":
            mode = "major"
        else:
            mode = key_signature.split()[-1]
        
        encoded_mode = mode_to_idx.get(mode, 0)
        mode_tensor = torch.tensor([encoded_mode], dtype=torch.long).to(self.device)

        converted_lines = normalize_chord(save_path, key_signature, key_type)

        lab_norm_path = save_path[:-4] + "_norm.lab"
        
        # Write the converted lines to the new file
        with open(lab_norm_path, 'w') as f:
            f.writelines(converted_lines)

        chords = []
        
        if not os.path.exists(lab_norm_path):
            chords.append((float(0), float(0), "N"))
        else:
            with open(lab_norm_path, 'r') as file:
                for line in file:
                    start, end, chord = line.strip().split()
                    chords.append((float(start), float(end), chord))

        encoded = []
        encoded_root= []
        encoded_attr=[]
        durations = []

        for start, end, chord in chords:
            chord_arr = chord.split(":")
            if len(chord_arr) == 1:
                chordRootID = chordRootDic[chord_arr[0]]
                if chord_arr[0] == "N" or chord_arr[0] == "X":
                    chordAttrID = 0
                else:
                    chordAttrID = 1
            elif len(chord_arr) == 2:
                chordRootID = chordRootDic[chord_arr[0]]
                chordAttrID = chordAttrDic[chord_arr[1]]
            encoded_root.append(chordRootID)
            encoded_attr.append(chordAttrID)

            if chord in chord_to_idx:
                encoded.append(chord_to_idx[chord])
            else:
                print(f"Warning: Chord {chord} not found in chord.json. Skipping.")
            
            durations.append(end - start)  # Compute duration
        
        encoded_chords = np.array(encoded)
        encoded_chords_root = np.array(encoded_root)
        encoded_chords_attr = np.array(encoded_attr)
        
        # Maximum sequence length for chords
        max_sequence_length = 100  # Define this globally or as a parameter

        # Truncate or pad chord sequences
        if len(encoded_chords) > max_sequence_length:
            # Truncate to max length
            encoded_chords = encoded_chords[:max_sequence_length]
            encoded_chords_root = encoded_chords_root[:max_sequence_length]
            encoded_chords_attr = encoded_chords_attr[:max_sequence_length]
        
        else:
            # Pad with zeros (padding value for chords)
            padding = [0] * (max_sequence_length - len(encoded_chords))
            encoded_chords = np.concatenate([encoded_chords, padding])
            encoded_chords_root = np.concatenate([encoded_chords_root, padding])
            encoded_chords_attr = np.concatenate([encoded_chords_attr, padding])
            
        # Convert to tensor
        chords_tensor = torch.tensor(encoded_chords, dtype=torch.long).to(self.device)
        chords_root_tensor = torch.tensor(encoded_chords_root, dtype=torch.long).to(self.device)
        chords_attr_tensor = torch.tensor(encoded_chords_attr, dtype=torch.long).to(self.device)

        model_input_dic = {
            "x_mert": final_embedding_mert.unsqueeze(0),
            "x_chord": chords_tensor.unsqueeze(0),
            "x_chord_root": chords_root_tensor.unsqueeze(0),
            "x_chord_attr": chords_attr_tensor.unsqueeze(0),
            "x_key": mode_tensor.unsqueeze(0)
        }

        model_input_dic = {k: v.to(self.device) for k, v in model_input_dic.items()}
        classification_output, regression_output = self.music2emo_model(model_input_dic)
        probs = torch.sigmoid(classification_output)

        tag_list = np.load ( "./inference/data/tag_list.npy")
        tag_list = tag_list[127:]
        mood_list = [t.replace("mood/theme---", "") for t in tag_list]
        threshold = threshold
        predicted_moods = [mood_list[i] for i, p in enumerate(probs.squeeze().tolist()) if p > threshold]
        valence, arousal = regression_output.squeeze().tolist()

        model_output_dic = {
            "valence": valence,
            "arousal": arousal,
            "predicted_moods": predicted_moods
        }

        return model_output_dic

# Initialize Mustango
if torch.cuda.is_available():
    music2emo = Music2emo()
else:
    music2emo = Music2emo(device="cpu")


def format_prediction(model_output_dic):
    """Format the model output in a more readable and attractive format"""
    valence = model_output_dic["valence"]
    arousal = model_output_dic["arousal"]
    moods = model_output_dic["predicted_moods"]
    
    # Create a formatted string with emojis and proper formatting
    output_text = """
🎵 **Music Emotion Recognition Results** 🎵
--------------------------------------------------
🎭 **Predicted Mood Tags:** {}
💖 **Valence:** {:.2f} (Scale: 1-9)
⚡ **Arousal:** {:.2f} (Scale: 1-9)
--------------------------------------------------
    """.format(
        ', '.join(moods) if moods else 'None',
        valence,
        arousal
    )
    
    return output_text

title = "Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models"
description_text = """
<p>
Upload an audio file to analyze its emotional characteristics using Music2Emo.
The model will predict:
• Mood tags describing the emotional content
• Valence score (1-9 scale, representing emotional positivity)
• Arousal score (1-9 scale, representing emotional intensity)
</p>
"""

css = """
#output-text {
    font-family: monospace;
    white-space: pre-wrap;
    font-size: 16px;
    background-color: #333333;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.gradio-container {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
}
.gr-button {
    color: white;
    background: #1565c0;
    border-radius: 100vh;
}
"""




# Initialize Music2Emo
if torch.cuda.is_available():
    music2emo = Music2emo()
else:
    music2emo = Music2emo(device="cpu")

with gr.Blocks(css=css) as demo:
    gr.HTML(f"<h1><center>{title}</center></h1>")
    gr.Markdown(description_text)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_audio = gr.Audio(
                label="Upload Audio File",
                type="filepath"  # Removed 'source' parameter
            )
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.01,
                label="Mood Detection Threshold",
                info="Adjust threshold for mood detection (0.0 to 1.0)"
            )
            predict_btn = gr.Button("🎭 Analyze Emotions", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(
                label="Analysis Results",
                elem_id="output-text"
            )
            
            # Add example usage
            gr.Examples(
                examples=["inference/input/test.mp3"],
                inputs=input_audio,
                outputs=output_text,
                fn=lambda x: format_prediction(music2emo.predict(x, 0.5)),
                cache_examples=True
            )

    predict_btn.click(
        fn=lambda audio, thresh: format_prediction(music2emo.predict(audio, thresh)),
        inputs=[input_audio, threshold],
        outputs=output_text
    )

    gr.Markdown("""
    ### 📝 Notes:
    - Supported audio formats: MP3, WAV
    - For best results, use high-quality audio files
    - Processing may take a few moments depending on file size
    """)

# Launch the demo
demo.queue().launch()

# with gr.Blocks(css=css) as demo:
#     gr.HTML(f"<h1><center>{title}</center></h1>")
#     gr.Markdown(description_text)
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             input_audio = gr.Audio(
#                 label="Upload Audio File",
#                 type="filepath",
#                 source="upload"
#             )
#             threshold = gr.Slider(
#                 minimum=0.0,
#                 maximum=1.0,
#                 value=0.5,
#                 step=0.01,
#                 label="Mood Detection Threshold",
#                 info="Adjust threshold for mood detection (0.0 to 1.0)"
#             )
#             predict_btn = gr.Button("🎭 Analyze Emotions", variant="primary")
        
#         with gr.Column(scale=1):
#             output_text = gr.Markdown(
#                 label="Analysis Results",
#                 elem_id="output-text"
#             )
            
#             # Add example usage
#             gr.Examples(
#                 examples=["inference/input/test.mp3"],
#                 inputs=input_audio,
#                 outputs=output_text,
#                 fn=lambda x: format_prediction(music2emo.predict(x, 0.5)),
#                 cache_examples=True
#             )

#     predict_btn.click(
#         fn=lambda audio, thresh: format_prediction(music2emo.predict(audio, thresh)),
#         inputs=[input_audio, threshold],
#         outputs=output_text
#     )

#     gr.Markdown("""
#     ### 📝 Notes:
#     - Supported audio formats: MP3, WAV
#     - For best results, use high-quality audio files
#     - Processing may take a few moments depending on file size
#     """)

# # Launch the demo
# demo.queue().launch()


# def gradio_predict(input_audio, threshold):
#     model_output_dic = music2emo.predict(input_audio, threshold)
#     return model_output_dic


# def format_prediction(model_output_dic):
#     """Format the model output for display"""
#     valence = model_output_dic["valence"]
#     arousal = model_output_dic["arousal"]
#     moods = model_output_dic["predicted_moods"]
    
#     # Format the output as a dictionary for the JSON component
#     formatted_output = {
#         "Dimensional Scores": {
#             "Valence": f"{valence:.3f}",
#             "Arousal": f"{arousal:.3f}"
#         },
#         "Predicted Moods": moods
#     }
    
#     return formatted_output

# title = "Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models"
# description_text = """
# <p>
# Predict emotion using Music2Emo by providing an input audio.
# <br/><br/> This is the demo for Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models
# <a href="https://arxiv.org/abs/2502.03979">Read our paper.</a>
# </p>
# """

# css = '''
# #duplicate-button {
#     margin: auto;
#     color: white;
#     background: #1565c0;
#     border-radius: 100vh;
# }
# '''

# # Initialize Music2Emo
# if torch.cuda.is_available():
#     music2emo = Music2emo()
# else:
#     music2emo = Music2emo(device="cpu")



# with gr.Blocks(css=css) as demo:
#     title = gr.HTML(f"<h1><center>{title}</center></h1>")
#     gr.Markdown(
#         """
#         This is the demo for Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models.
#         [Read our paper](https://arxiv.org/abs/2502.03979).
#         """
#     )
    
#     with gr.Row():
#         with gr.Column():
#             with gr.Column(visible=True) as rowA:
#                 with gr.Row():
#                     input_audio = gr.Audio(
#                         label="Input Audio",
#                         type="filepath",
#                         source="upload"
#                     )
#                 with gr.Row():
#                     threshold = gr.Slider(
#                         minimum=0.0,
#                         maximum=1.0,
#                         value=0.5,
#                         step=0.01,
#                         label="Mood Detection Threshold",
#                         info="Adjust threshold for mood detection (0.0 to 1.0)"
#                     )
#                 with gr.Row():
#                     btn = gr.Button("Predict", variant="primary")

#         with gr.Column():
#             with gr.Row():
#                 output_emo = gr.JSON(
#                     label="Prediction Results",
#                     info="Displays valence, arousal scores and predicted moods"
#                 )

#     btn.click(
#         fn=lambda audio, thresh: format_prediction(music2emo.predict(audio, thresh)),
#         inputs=[input_audio, threshold],
#         outputs=[output_emo],
#     )

# # Launch the demo
# demo.queue().launch()

# title="Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models"
# description_text = """
# <p>
# Predict emotion using Music2Emo by providing an input audio.
# <br/><br/> This is the demo for Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models
# <a href="https://arxiv.org/abs/2502.03979">Read our paper.</a>
# <p/>
# """


# css = '''
# #duplicate-button {
# margin: auto;
# color: white;
# background: #1565c0;
# border-radius: 100vh;
# }
# '''
# # with gr.Blocks() as demo:
# with gr.Blocks(css=css) as demo:
#     title=gr.HTML(f"<h1><center>{title}</center></h1>")
#     gr.Markdown(
#             """
#             This is the demo for Music2Emo: Towards Unified Music Emotion Recognition across Dimensional and Categorical Models.
#             [Read our paper](https://arxiv.org/abs/2502.03979).
#             """
#     )
#     with gr.Row():
#         with gr.Column():
#             # with gr.Row(visible=True) as mainA:
#             # with gr.Column(visible=True) as colA:
#             with gr.Column(visible=True) as rowA:
#                 with gr.Row():
#                     input_audio = ???
#                 with gr.Row():
#                     with gr.Row():
#                         threshold = ???
#                 with gr.Row():
#                     btn = gr.Button("Predict")

#         with gr.Column():
#             with gr.Row():
#                 output_emo = gr.Label ???

#     btn.click(
#         fn=gradio_predict, 
#         inputs=[input_audio,threshold],
#         outputs=[output_emo],
#     )

# demo.queue().launch()