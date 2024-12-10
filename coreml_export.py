import demucs.api
filename = "test2.mp3"

separator = demucs.api.Separator()
origin, separated = separator.separate_audio_file(filename)
for stem, audio_data in separated.items():
    demucs.api.save_audio(audio_data, f"separated/coreml/{stem}_{filename}", samplerate=separator.samplerate)