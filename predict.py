"""
Predictor for Whisper Algerian Dialect on Replicate
Model: MohammedNasri/whisper-algerian-dialect (Whisper-tiny fine-tuned)

Uses HuggingFace pipeline with chunk_length_s=30 + stride to handle:
- Audio decoding (WebM/Opus, WAV, MP3, FLAC) via ffmpeg backend
- Long audio (> 30s) via automatic chunking with overlap
- Language forcing (ar) and bounded token generation
"""

import torch
from cog import BasePredictor, Input, Path as CogPath
from transformers import pipeline


class Predictor(BasePredictor):
    def setup(self):
        """Load the ASR pipeline once at startup."""
        device = 0 if torch.cuda.is_available() else -1

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model="MohammedNasri/whisper-algerian-dialect",
            torch_dtype=torch.float32,
            device=device,
            chunk_length_s=30,
            stride_length_s=5,
        )

    def predict(
        self,
        audio: CogPath = Input(
            description="Audio file (WAV, MP3, WebM, FLAC, etc.)"
        ),
    ) -> str:
        """
        Transcribe Algerian Arabic dialect audio (with French code-switching).

        The pipeline handles audio decoding, 30s chunking with 5s stride
        overlap, and decodes the merged transcription.
        """
        result = self.pipe(
            str(audio),
            return_timestamps=False,
            generate_kwargs={
                "language": "ar",
                "task": "transcribe",
                "max_new_tokens": 440,
            },
        )

        return result["text"].strip()
