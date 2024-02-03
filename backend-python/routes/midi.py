import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth

router = APIRouter()


class TextToMidiBody(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "p:24:a p:2a:a p:31:a p:39:a p:3b:a p:45:a b:26:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:24:0 p:2a:0 p:31:0 p:39:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:26:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:2e:a p:3b:a p:45:a b:26:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:2e:0 p:3b:0 p:45:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:2e:a p:3b:a p:45:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:2e:0 p:3b:0 p:45:0 b:26:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:26:a p:2a:a p:3b:a p:45:a t14 p:26:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a b:26:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:2a:0 p:3b:0 p:45:0 b:26:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:2d:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 b:2d:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:24:a p:2e:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:24:0 p:2e:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:26:a p:2a:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:26:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:26:a p:2e:a p:31:a p:39:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:26:0 p:2e:0 p:31:0 p:39:0 p:3b:0 p:45:0 b:21:0 t2 p:26:a p:2e:a p:31:a p:39:a p:3b:a p:45:a b:21:a t14 p:26:0 p:2e:0 p:31:0 p:39:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:24:a p:2a:a p:31:a p:39:a p:3b:a p:45:a b:1f:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:24:0 p:2a:0 p:31:0 p:39:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:1f:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:2e:a p:3b:a p:45:a b:1f:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:2e:0 p:3b:0 p:45:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:2e:a p:3b:a p:45:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:2e:0 p:3b:0 p:45:0 b:1f:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:26:a p:2a:a p:3b:a p:45:a t14 p:26:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a b:1f:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:2a:0 p:3b:0 p:45:0 b:1f:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:1f:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 b:1f:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:24:a p:2e:a p:3b:a p:45:a b:26:a g:39:a g:39:a g:3e:a g:3e:a g:42:a g:42:a pi:39:a pi:3e:a pi:42:a t14 p:24:0 p:2e:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0",
            }
        }
    }


@router.post("/text-to-midi", tags=["MIDI"])
def text_to_midi(body: TextToMidiBody):
    vocab_config_type = global_var.get(global_var.Midi_Vocab_Config_Type)
    if vocab_config_type == global_var.MidiVocabConfig.Piano:
        vocab_config = "backend-python/utils/vocab_config_piano.json"
    else:
        vocab_config = "backend-python/utils/midi_vocab_config.json"
    cfg = VocabConfig.from_json(vocab_config)
    mid = convert_str_to_midi(cfg, body.text.strip())
    mid_data = io.BytesIO()
    mid.save(None, mid_data)
    mid_data.seek(0)

    return StreamingResponse(mid_data, media_type="audio/midi")


@router.post("/midi-to-text", tags=["MIDI"])
async def midi_to_text(file_data: UploadFile):
    vocab_config_type = global_var.get(global_var.Midi_Vocab_Config_Type)
    if vocab_config_type == global_var.MidiVocabConfig.Piano:
        vocab_config = "backend-python/utils/vocab_config_piano.json"
    else:
        vocab_config = "backend-python/utils/midi_vocab_config.json"
    cfg = VocabConfig.from_json(vocab_config)
    filter_config = "backend-python/utils/midi_filter_config.json"
    filter_cfg = FilterConfig.from_json(filter_config)
    mid = mido.MidiFile(file=file_data.file)
    output_list = convert_midi_to_str(cfg, filter_cfg, mid)
    if len(output_list) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "bad midi file")

    return {"text": output_list[0]}


class TxtToMidiBody(BaseModel):
    txt_path: str
    midi_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "txt_path": "midi/sample.txt",
                "midi_path": "midi/sample.mid",
            }
        }
    }


@router.post("/txt-to-midi", tags=["MIDI"])
def txt_to_midi(body: TxtToMidiBody):
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    if not body.midi_path.startswith("midi/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "bad output path")

    vocab_config_type = global_var.get(global_var.Midi_Vocab_Config_Type)
    if vocab_config_type == global_var.MidiVocabConfig.Piano:
        vocab_config = "backend-python/utils/vocab_config_piano.json"
    else:
        vocab_config = "backend-python/utils/midi_vocab_config.json"
    cfg = VocabConfig.from_json(vocab_config)
    with open(body.txt_path, "r") as f:
        text = f.read()
    text = text.strip()
    mid = convert_str_to_midi(cfg, text)
    mid.save(body.midi_path)

    return "success"


class MidiToWavBody(BaseModel):
    midi_path: str
    wav_path: str
    sound_font_path: str = "assets/default_sound_font.sf2"

    model_config = {
        "json_schema_extra": {
            "example": {
                "midi_path": "midi/sample.mid",
                "wav_path": "midi/sample.wav",
                "sound_font_path": "assets/default_sound_font.sf2",
            }
        }
    }


@router.post("/midi-to-wav", tags=["MIDI"])
def midi_to_wav(body: MidiToWavBody):
    """
    Install fluidsynth first, see more: https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions
    """

    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    if not body.wav_path.startswith("midi/"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "bad output path")

    fs = FluidSynth(body.sound_font_path)
    fs.midi_to_audio(body.midi_path, body.wav_path)

    return "success"


class TextToWavBody(BaseModel):
    text: str
    wav_name: str
    sound_font_path: str = "assets/default_sound_font.sf2"

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "p:24:a p:2a:a p:31:a p:39:a p:3b:a p:45:a b:26:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:24:0 p:2a:0 p:31:0 p:39:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:26:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:2e:a p:3b:a p:45:a b:26:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:2e:0 p:3b:0 p:45:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:2e:a p:3b:a p:45:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:2e:0 p:3b:0 p:45:0 b:26:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:26:a p:2a:a p:3b:a p:45:a t14 p:26:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a b:26:a g:3e:a g:3e:a g:42:a g:42:a g:45:a g:45:a pi:3e:a pi:42:a pi:45:a t14 p:2a:0 p:3b:0 p:45:0 b:26:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:2d:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 b:2d:0 g:3e:0 g:3e:0 g:42:0 g:42:0 g:45:0 g:45:0 pi:3e:0 pi:42:0 pi:45:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:24:a p:2e:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:24:0 p:2e:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:26:a p:2a:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:26:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:26:a p:2e:a p:31:a p:39:a p:3b:a p:45:a b:21:a g:39:a g:39:a g:3d:a g:3d:a g:40:a g:40:a pi:39:a pi:3d:a pi:40:a t14 p:26:0 p:2e:0 p:31:0 p:39:0 p:3b:0 p:45:0 b:21:0 t2 p:26:a p:2e:a p:31:a p:39:a p:3b:a p:45:a b:21:a t14 p:26:0 p:2e:0 p:31:0 p:39:0 p:3b:0 p:45:0 b:21:0 g:39:0 g:39:0 g:3d:0 g:3d:0 g:40:0 g:40:0 pi:39:0 pi:3d:0 pi:40:0 t2 p:24:a p:2a:a p:31:a p:39:a p:3b:a p:45:a b:1f:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:24:0 p:2a:0 p:31:0 p:39:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0 p:45:0 b:1f:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:2e:a p:3b:a p:45:a b:1f:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:2e:0 p:3b:0 p:45:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:2e:a p:3b:a p:45:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:2e:0 p:3b:0 p:45:0 b:1f:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:26:a p:2a:a p:3b:a p:45:a t14 p:26:0 p:2a:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a b:1f:a g:3b:a g:3b:a g:3e:a g:3e:a g:43:a g:43:a pi:3b:a pi:3e:a pi:43:a t14 p:2a:0 p:3b:0 p:45:0 b:1f:0 t2 p:24:a p:2a:a p:3b:a p:45:a b:1f:a t14 p:24:0 p:2a:0 p:3b:0 p:45:0 b:1f:0 g:3b:0 g:3b:0 g:3e:0 g:3e:0 g:43:0 g:43:0 pi:3b:0 pi:3e:0 pi:43:0 t2 p:24:a p:2e:a p:3b:a p:45:a b:26:a g:39:a g:39:a g:3e:a g:3e:a g:42:a g:42:a pi:39:a pi:3e:a pi:42:a t14 p:24:0 p:2e:0 p:3b:0 p:45:0 t2 p:2a:a p:3b:a p:45:a t14 p:2a:0 p:3b:0",
                "wav_name": "sample",
                "sound_font_path": "assets/default_sound_font.sf2",
            }
        }
    }


@router.post("/text-to-wav", tags=["MIDI"])
def text_to_wav(body: TextToWavBody):
    """
    Install fluidsynth first, see more: https://github.com/FluidSynth/fluidsynth/wiki/Download#distributions
    """

    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    text = body.text.strip()
    if not text.startswith("<start>"):
        text = "<start> " + text
    if not text.endswith("<end>"):
        text = text + " <end>"
    txt_path = f"midi/{body.wav_name}.txt"
    midi_path = f"midi/{body.wav_name}.mid"
    wav_path = f"midi/{body.wav_name}.wav"
    with open(txt_path, "w") as f:
        f.write(text)
    txt_to_midi(TxtToMidiBody(txt_path=txt_path, midi_path=midi_path))
    midi_to_wav(
        MidiToWavBody(
            midi_path=midi_path, wav_path=wav_path, sound_font_path=body.sound_font_path
        )
    )

    return "success"
