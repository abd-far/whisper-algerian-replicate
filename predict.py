"""
Predictor for Whisper Algerian Dialect model on Replicate
Model: MohammedNasri/whisper-algerian-dialect
WER: ~23% on Algerian Arabic dialect
Supports: Algerian Darija + French code-switching
"""

import torch
import librosa
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class Predictor(BasePredictor):
    def setup(self):
        """Load model and processor once at startup"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the fine-tuned Whisper Algerian model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "MohammedNasri/whisper-algerian-dialect",
            torch_dtype=torch.float32,
            device_map=device,
        )

        # Load the processor (handles audio preprocessing)
        self.processor = WhisperProcessor.from_pretrained(
            "MohammedNasri/whisper-algerian-dialect"
        )

        self.model.eval()

    def predict(
        self,
        audio: CogPath = Input(
            description="Audio file (WAV, MP3, WebM, FLAC, etc.)"
        ),
    ) -> str:
        """
        Transcribe Algerian Arabic dialect audio to text.

        Supports:
        - Pure Algerian Darija
        - Algerian Darija + French code-switching
        - Medical consultation audio

        Args:
            audio: Audio file path

        Returns:
            Transcribed text in Algerian Arabic/French mix
        """
        # Load audio at 16kHz (required by Whisper)
        audio_array, sampling_rate = librosa.load(
            str(audio),
            sr=16000,
        )

        # Process audio for the model
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )

        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs["input_features"])

        # Decode to text
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return transcription.strip()
