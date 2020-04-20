from deepspeech import Model
import numpy as np
import speech_recognition as sr
import sys

sample_rate = 16000
beam_width = 500
lm_alpha = 0.75
lm_beta = 1.85
n_features = 26
n_context = 9
models_folder = 'deepspeech-0.6.1-models/'
model_name = models_folder+"output_graph.pbmm"
alphabet = models_folder+"alphabet.txt"
language_model = models_folder+"lm.binary"
trie = models_folder+"trie"

ds = Model(model_name, beam_width)
ds.enableDecoderWithLM(language_model, trie, lm_alpha, lm_beta)

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

r = sr.Recognizer()
with sr.Microphone(sample_rate=sample_rate,device_index=int(sys.argv[1])) as source:
    print("Say Something")
    audio = r.listen(source)
    fs = audio.sample_rate
    audio = np.frombuffer(audio.frame_data, np.int16)
    print(ds.stt(audio))
