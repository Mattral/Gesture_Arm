"""
gesture_arm.speech.multimodal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Threaded speech recognition (ASR) and text-to-speech (TTS).

Both run as daemon threads so they never block the main control loop.
The ASR thread pushes recognized commands to a thread-safe queue.
The TTS thread drains a speech queue without repeating consecutive duplicates.

Paper Section III-D:
  "The system uses a TTS engine to confirm executed actions,
   enhancing user interaction and operational awareness."
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TTS — Text-to-speech output
# ══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    """
    Non-blocking text-to-speech engine.

    Queues utterances and speaks them in a daemon thread.
    Consecutive duplicate utterances are suppressed so a repeated
    gesture command doesn't trigger endless speech.

    Usage::

        tts = TTSEngine(rate=160, volume=0.8)
        tts.start()
        tts.say("System ready")
    """

    def __init__(self, rate: int = 160, volume: float = 0.8) -> None:
        self._rate   = rate
        self._volume = volume
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tts-thread")

    def start(self) -> None:
        self._thread.start()
        logger.info("TTS engine started (rate=%d wpm, vol=%.1f)", self._rate, self._volume)

    def say(self, text: str) -> None:
        """Queue an utterance (non-blocking). Drops consecutive duplicates."""
        try:
            last = self._q.queue[-1] if self._q.queue else None
        except Exception:
            last = None
        if last != text:
            self._q.put(text)

    def stop(self) -> None:
        self._q.put(None)   # sentinel

    def _run(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
        except Exception as exc:
            logger.error("pyttsx3 init failed: %s — TTS disabled.", exc)
            return

        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                logger.warning("TTS error: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# ASR — Automatic speech recognition
# ══════════════════════════════════════════════════════════════════════════════

class ASRListener:
    """
    Continuous speech recognition in a background daemon thread.

    Recognized commands are dispatched via the on_command callback.
    Unknown words are logged but not forwarded.

    Usage::

        def handle(cmd: str): print("Command:", cmd)
        asr = ASRListener(commands={"forward", "stop", "left", "right"}, on_command=handle)
        asr.start()
    """

    def __init__(
        self,
        commands: set[str],
        on_command: Callable[[str], None],
        tts: Optional[TTSEngine] = None,
    ) -> None:
        self._commands   = {c.lower() for c in commands}
        self._on_command = on_command
        self._tts        = tts
        self._thread     = threading.Thread(
            target=self._run, daemon=True, name="asr-thread"
        )

    def start(self) -> None:
        self._thread.start()
        logger.info("ASR listener started — watching for %d commands.", len(self._commands))

    def _run(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            logger.error(
                "SpeechRecognition not installed. pip install SpeechRecognition\n"
                "ASR disabled."
            )
            return

        recognizer = sr.Recognizer()
        mic         = sr.Microphone()

        while True:
            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)

                text = recognizer.recognize_google(audio).lower().strip()
                logger.info("ASR heard: '%s'", text)

                # Match multi-word phrases and single keywords
                matched = next((cmd for cmd in self._commands if cmd in text), None)
                if matched:
                    self._on_command(matched)
                    if self._tts:
                        self._tts.say(f"Command: {matched}")
                else:
                    logger.debug("ASR: no command match for '%s'", text)

            except sr.WaitTimeoutError:
                pass   # silence — not an error
            except sr.UnknownValueError:
                pass   # unintelligible audio
            except sr.RequestError as exc:
                logger.warning("ASR service error: %s", exc)
            except Exception as exc:
                logger.error("ASR unexpected error: %s", exc)
