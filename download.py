from audioldm import text_to_audio, build_model

def download_model():
    model_id = "haoheliu/AudioLDM-S-Full"
    audioldm = build_model()

if __name__ == "__main__":
    download_model()