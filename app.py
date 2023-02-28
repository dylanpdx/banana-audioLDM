from audioldm import text_to_audio, build_model, save_wave
import soundfile as sf
import os
import torch
import base64
from random import randint

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    print("build model")
    model = build_model()
    print("build model done")


def text2audio(text, duration, guidance_scale, random_seed, n_candidates):
    global model
    # print(text, length, guidance_scale)
    waveform = text_to_audio(
        model,
        text,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )
    return waveform
        
# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    
    mode=model_inputs.get('mode', "text2audio")
    duration=model_inputs.get('duration', 5)
    guidance_scale=model_inputs.get('guidance_scale', 2.5)
    random_seed=model_inputs.get('random_seed', randint(0,100000))
    n_candidates=model_inputs.get('n_candidates', 3)
    
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = text2audio(prompt,duration,guidance_scale,random_seed,n_candidates)
    save_wave(result,"/tmp","gen.wav")
    in_file = open("/tmp/gen_0.wav", "rb")
    data = in_file.read()
    in_file.close()
    os.remove("/tmp/gen_0.wav")
    b64_text = base64.b64encode(data).decode("ascii")
    # Return the results as a dictionary
    return b64_text
