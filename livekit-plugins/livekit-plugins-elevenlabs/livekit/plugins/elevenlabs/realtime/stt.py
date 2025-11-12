# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from ..log import logger
from .models import AudioFormat, CommitStrategy, RealtimeModels


@dataclass
class RealtimeSTTOptions:
    model: RealtimeModels | str
    sample_rate: int
    audio_format: AudioFormat
    commit_strategy: CommitStrategy = CommitStrategy.MANUAL
    language_code: str | None = None
    vad_silence_threshold_secs: float | None = None
    vad_threshold: float | None = None
    min_speech_duration_ms: int | None = None
    min_silence_duration_ms: int | None = None
    base_url: str = "wss://api.elevenlabs.io"


class RealtimeSTT(stt.STT):
    def __init__(
        self,
        *,
        model: RealtimeModels | str = RealtimeModels.SCRIBE_V2_REALTIME,
        sample_rate: int = 16000,
        audio_format: AudioFormat = AudioFormat.PCM_16000,
        commit_strategy: CommitStrategy = CommitStrategy.MANUAL,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of ElevenLabs Realtime STT.

        Args:
            model: The ElevenLabs model to use for realtime speech recognition. Defaults to "scribe_v2_realtime".
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            audio_format: The audio format for streaming. Defaults to PCM_16000.
            commit_strategy: Strategy for committing transcriptions (VAD or MANUAL). Defaults to MANUAL.
            language_code: ISO-639-1 or ISO-639-3 language code. Optional.
            vad_silence_threshold_secs: Silence threshold in seconds for VAD (0.3-3.0). Optional.
            vad_threshold: Threshold for voice activity detection (0.1-0.9). Optional.
            min_speech_duration_ms: Minimum speech duration in milliseconds (50-2000). Optional.
            min_silence_duration_ms: Minimum silence duration in milliseconds (50-2000). Optional.
            api_key: Your ElevenLabs API key. If not provided, will look for ELEVEN_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.
            base_url: The base URL for ElevenLabs realtime API. Defaults to "wss://api.elevenlabs.io".

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting
            the ELEVEN_API_KEY environmental variable.
        """  # noqa: E501

        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError("ElevenLabs API key is required")
        self._api_key = elevenlabs_api_key

        self._opts = RealtimeSTTOptions(
            model=model,
            sample_rate=sample_rate,
            audio_format=audio_format,
            commit_strategy=commit_strategy,
            language_code=language_code if is_given(language_code) else None,
            vad_silence_threshold_secs=vad_silence_threshold_secs
            if is_given(vad_silence_threshold_secs)
            else None,
            vad_threshold=vad_threshold if is_given(vad_threshold) else None,
            min_speech_duration_ms=min_speech_duration_ms
            if is_given(min_speech_duration_ms)
            else None,
            min_silence_duration_ms=min_silence_duration_ms
            if is_given(min_silence_duration_ms)
            else None,
            base_url=base_url if is_given(base_url) else "wss://api.elevenlabs.io",
        )
        self._session = http_session
        self._streams = weakref.WeakSet[RealtimeSpeechStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Realtime API does not support non-streaming recognize. Use with a StreamAdapter"
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RealtimeSpeechStream:
        stream = RealtimeSpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[RealtimeModels | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat] = NOT_GIVEN,
        commit_strategy: NotGivenOr[CommitStrategy] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(audio_format):
            self._opts.audio_format = audio_format
        if is_given(commit_strategy):
            self._opts.commit_strategy = commit_strategy
        if is_given(language_code):
            self._opts.language_code = language_code
        if is_given(vad_silence_threshold_secs):
            self._opts.vad_silence_threshold_secs = vad_silence_threshold_secs
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms
        if is_given(base_url):
            self._opts.base_url = base_url

        for stream in self._streams:
            stream.update_options(
                model=model,
                sample_rate=sample_rate,
                audio_format=audio_format,
                commit_strategy=commit_strategy,
                language_code=language_code,
                vad_silence_threshold_secs=vad_silence_threshold_secs,
                vad_threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                base_url=base_url,
            )


class RealtimeSpeechStream(stt.SpeechStream):
    _CLOSE_MSG: str = json.dumps(
        {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True}
    )

    def __init__(
        self,
        *,
        stt: RealtimeSTT,
        opts: RealtimeSTTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        model: NotGivenOr[RealtimeModels | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat] = NOT_GIVEN,
        commit_strategy: NotGivenOr[CommitStrategy] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(audio_format):
            self._opts.audio_format = audio_format
        if is_given(commit_strategy):
            self._opts.commit_strategy = commit_strategy
        if is_given(language_code):
            self._opts.language_code = language_code
        if is_given(vad_silence_threshold_secs):
            self._opts.vad_silence_threshold_secs = vad_silence_threshold_secs
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms
        if is_given(base_url):
            self._opts.base_url = base_url

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # forward audio to elevenlabs in chunks of 50ms
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            has_ended = False
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    has_ended = True

                for frame in frames:
                    # Convert to base64 and send
                    chunk_base64 = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                    message = {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": chunk_base64,
                        "commit": False,
                        "sample_rate": self._opts.sample_rate,
                    }
                    await ws.send_str(json.dumps(message))

                    if has_ended:
                        # Send commit message when ending
                        await ws.send_str(RealtimeSpeechStream._CLOSE_MSG)
                        has_ended = False

            # tell elevenlabs we are done sending audio/inputs
            closing_ws = True
            await ws.send_str(RealtimeSpeechStream._CLOSE_MSG)

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # close is expected, see SpeechStream.aclose
                    # or when the agent session ends, the http session is closed
                    if closing_ws or self._session.closed:
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(message="elevenlabs connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected elevenlabs message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process elevenlabs message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # retrieve the exception
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        # Build WebSocket URL with query parameters
        params = [
            f"model_id={self._opts.model}",
            f"encoding={self._opts.audio_format.value}",
            f"sample_rate={self._opts.sample_rate}",
            f"commit_strategy={self._opts.commit_strategy.value}",
        ]

        # Add optional VAD parameters
        if self._opts.vad_silence_threshold_secs is not None:
            params.append(f"vad_silence_threshold_secs={self._opts.vad_silence_threshold_secs}")
        if self._opts.vad_threshold is not None:
            params.append(f"vad_threshold={self._opts.vad_threshold}")
        if self._opts.min_speech_duration_ms is not None:
            params.append(f"min_speech_duration_ms={self._opts.min_speech_duration_ms}")
        if self._opts.min_silence_duration_ms is not None:
            params.append(f"min_silence_duration_ms={self._opts.min_silence_duration_ms}")
        if self._opts.language_code is not None:
            params.append(f"language_code={self._opts.language_code}")

        query_string = "&".join(params)
        ws_url = f"{self._opts.base_url}/v1/speech-to-text/realtime?{query_string}"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                    headers={"xi-api-key": self._api_key},
                    heartbeat=30.0,
                ),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to elevenlabs") from e
        return ws

    def _send_transcript_event(self, event_type: stt.SpeechEventType, data: dict) -> None:
        alts = _parse_transcription(self._opts.language_code or "en", data)
        if alts:
            event = stt.SpeechEvent(
                type=event_type,
                alternatives=alts,
            )
            self._event_ch.send_nowait(event)

    def _process_stream_event(self, data: dict) -> None:
        message_type = data.get("message_type")

        if message_type == "session_started":
            # Session started successfully
            pass

        elif message_type == "partial_transcript":
            if not self._speaking:
                self._speaking = True
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)

            self._send_transcript_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, data)

        elif message_type == "committed_transcript":
            if not self._speaking:
                return

            self._send_transcript_event(stt.SpeechEventType.FINAL_TRANSCRIPT, data)
            self._speaking = False
            end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
            self._event_ch.send_nowait(end_event)

        elif message_type == "error":
            logger.warning("elevenlabs sent an error", extra={"data": data})
            desc = data.get("error") or "unknown error from elevenlabs"
            code = -1
            raise APIStatusError(message=desc, status_code=code)


def _parse_transcription(language: str, data: dict[str, Any]) -> list[stt.SpeechData]:
    transcript = data.get("transcript")
    if not transcript:
        return []

    # For ElevenLabs realtime, we don't get detailed timing info in all cases
    # Use what's available
    start_time = data.get("audio_start", 0)
    end_time = data.get("audio_end", 0)
    confidence = data.get("confidence", 1.0)

    sd = stt.SpeechData(
        language=language,
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
        text=transcript,
    )
    return [sd]
