from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import audio as audio_utils
from livekit.agents.utils import http_context, is_given

from ..stt import AUTHORIZATION_HEADER
from ..log import logger


API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"


@dataclass
class _RealtimeOptions:
    api_key: str
    base_url: str
    language_code: str | None = None
    # realtime defaults
    realtime_model_id: str = "scribe_v2_realtime"
    sample_rate: int = 16000
    commit_strategy: str = "vad"  # "vad" | "manual"
    vad_silence_threshold_secs: Optional[float] = None
    vad_threshold: Optional[float] = None
    min_speech_duration_ms: Optional[int] = None
    min_silence_duration_ms: Optional[int] = None


class STTRealtime(stt.STT):
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        model_id: str = "scribe_v2_realtime",
        sample_rate: int = 16000,
        commit_strategy: str = "vad",
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError("ElevenLabs API key is required, set ELEVEN_API_KEY or pass api_key")

        self._opts = _RealtimeOptions(
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            realtime_model_id=model_id,
            sample_rate=sample_rate,
            commit_strategy=commit_strategy,
        )
        if is_given(language_code):
            self._opts.language_code = language_code
        if is_given(vad_silence_threshold_secs):
            self._opts.vad_silence_threshold_secs = vad_silence_threshold_secs  # type: ignore
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold  # type: ignore
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms  # type: ignore
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms  # type: ignore

        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStreamRealtime]()

    @property
    def model(self) -> str:
        return self._opts.realtime_model_id

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = http_context.http_session()
        return self._session

    async def _recognize_impl(  # type: ignore[override]
        self,
        buffer: audio_utils.AudioBuffer,  # not used
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Use stream() with STTRealtime")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SpeechStreamRealtime":
        if is_given(language):
            self._opts.language_code = language  # type: ignore

        stream = SpeechStreamRealtime(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model_id: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        commit_strategy: NotGivenOr[str] = NOT_GIVEN,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model_id):
            self._opts.realtime_model_id = model_id  # type: ignore
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate  # type: ignore
        if is_given(commit_strategy):
            self._opts.commit_strategy = commit_strategy  # type: ignore
        if is_given(vad_silence_threshold_secs):
            self._opts.vad_silence_threshold_secs = vad_silence_threshold_secs  # type: ignore
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold  # type: ignore
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms  # type: ignore
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms  # type: ignore
        if is_given(language_code):
            self._opts.language_code = language_code  # type: ignore
        if is_given(base_url):
            self._opts.base_url = base_url  # type: ignore

        for s in list(self._streams):
            s.update_options(
                model_id=self._opts.realtime_model_id,
                sample_rate=self._opts.sample_rate,
                commit_strategy=self._opts.commit_strategy,
                vad_silence_threshold_secs=self._opts.vad_silence_threshold_secs,
                vad_threshold=self._opts.vad_threshold,
                min_speech_duration_ms=self._opts.min_speech_duration_ms,
                min_silence_duration_ms=self._opts.min_silence_duration_ms,
                language_code=self._opts.language_code,
                base_url=self._opts.base_url,
            )


class _DurationCollector:
    def __init__(self, callback, duration: float = 5.0) -> None:
        self._callback = callback
        self._duration = duration
        self._accum = 0.0
        self._last = asyncio.get_event_loop().time()

    def push(self, value: float) -> None:
        now = asyncio.get_event_loop().time()
        self._accum += value
        if (now - self._last) >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._accum > 0.0:
            self._callback(self._accum)
            self._accum = 0.0
        self._last = asyncio.get_event_loop().time()


class SpeechStreamRealtime(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STTRealtime,
        conn_options: APIConnectOptions,
        opts: _RealtimeOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._speaking = False
        self._request_id = ""
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._usage_collector = _DurationCollector(self._on_audio_duration_report, duration=5.0)

    def update_options(
        self,
        *,
        model_id: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        commit_strategy: NotGivenOr[str] = NOT_GIVEN,
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model_id):
            self._opts.realtime_model_id = model_id  # type: ignore
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate  # type: ignore
        if is_given(commit_strategy):
            self._opts.commit_strategy = commit_strategy  # type: ignore
        if is_given(vad_silence_threshold_secs):
            self._opts.vad_silence_threshold_secs = vad_silence_threshold_secs  # type: ignore
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold  # type: ignore
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms  # type: ignore
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms  # type: ignore
        if is_given(language_code):
            self._opts.language_code = language_code  # type: ignore
        if is_given(base_url):
            self._opts.base_url = base_url  # type: ignore

    async def _run(self) -> None:
        closed_event = asyncio.Event()

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            samples_50ms = self._opts.sample_rate // 20
            bstream = audio_utils.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )
            pending_commit = False
            async for data in self._input_ch:
                if closed_event.is_set() or ws.closed:
                    return
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(bstream.flush())
                    pending_commit = True

                for frame in frames:
                    if closed_event.is_set() or ws.closed:
                        return
                    self._usage_collector.push(frame.duration)
                    chunk_b64 = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                    try:
                        await ws.send_str(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": chunk_b64,
                                    "commit": False,
                                    "sample_rate": self._opts.sample_rate,
                                }
                            )
                        )
                    except Exception:
                        logger.warning("failed to send audio chunk to elevenlabs (ws likely closing)", exc_info=True)
                        return

                if pending_commit:
                    pending_commit = False
                    if closed_event.is_set() or ws.closed:
                        return
                    try:
                        await ws.send_str(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": "",
                                    "commit": True,
                                    "sample_rate": self._opts.sample_rate,
                                }
                            )
                        )
                    except Exception:
                        logger.warning("failed to send commit to elevenlabs (ws likely closing)", exc_info=True)
                        return
                    self._usage_collector.flush()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    closed_event.set()
                    return

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected elevenlabs message type %s", msg.type)
                    continue
                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process elevenlabs message")

        ws = await self._connect_ws()
        self._ws = ws
        try:
            await asyncio.gather(send_task(ws), recv_task(ws))
        finally:
            await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = _to_elevenlabs_ws_url(
            base_url=self._opts.base_url,
            opts={
                "model_id": self._opts.realtime_model_id,
                "encoding": f"pcm_{self._opts.sample_rate}",
                "sample_rate": self._opts.sample_rate,
                "commit_strategy": self._opts.commit_strategy,
                **(
                    {"vad_silence_threshold_secs": self._opts.vad_silence_threshold_secs}
                    if self._opts.vad_silence_threshold_secs is not None
                    else {}
                ),
                **({"vad_threshold": self._opts.vad_threshold} if self._opts.vad_threshold is not None else {}),
                **(
                    {"min_speech_duration_ms": self._opts.min_speech_duration_ms}
                    if self._opts.min_speech_duration_ms is not None
                    else {}
                ),
                **(
                    {"min_silence_duration_ms": self._opts.min_silence_duration_ms}
                    if self._opts.min_silence_duration_ms is not None
                    else {}
                ),
                **({"language_code": self._opts.language_code} if self._opts.language_code else {}),
            },
        )
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(url, headers={AUTHORIZATION_HEADER: self._opts.api_key}, heartbeat=30.0),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to elevenlabs realtime") from e
        return ws

    def _on_audio_duration_report(self, duration: float) -> None:
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=self._request_id,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)

    def _emit_interim(self, data: dict[str, Any]) -> None:
        alts = _parse_transcription(self._opts.language_code or "en", data)
        if not alts:
            return
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=alts,
            )
        )

    def _emit_final(self, data: dict[str, Any]) -> None:
        alts = _parse_transcription(self._opts.language_code or "en", data)
        if not alts:
            return
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=alts,
            )
        )
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        mt = data.get("message_type")
        if req_id := data.get("request_id"):
            self._request_id = req_id
        if mt == "partial_transcript":
            self._emit_interim(data)
        elif mt in ("committed_transcript", "committed_transcript_with_timestamps"):
            self._emit_final(data)
        elif mt == "error":
            desc = data.get("error") or "unknown error from elevenlabs"
            raise APIStatusError(message=desc, status_code=-1, request_id=self._request_id, body=None)

    async def aclose(self) -> None:
        await super().aclose()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass


def _to_elevenlabs_ws_url(*, base_url: str, opts: dict[str, Any]) -> str:
    base = base_url.rstrip("/")
    if base.startswith("http://"):
        base = base.replace("http://", "ws://", 1)
    elif base.startswith("https://"):
        base = base.replace("https://", "wss://", 1)
    path = f"{base}/speech-to-text/realtime"

    q = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in opts.items() if v is not None}
    from urllib.parse import urlencode

    return f"{path}?{urlencode(q)}"


def _parse_transcription(language: str, data: dict[str, Any]) -> list[stt.SpeechData]:
    transcript = data.get("transcript") or ""
    words = data.get("words") or []
    start_time = 0.0
    end_time = 0.0
    if isinstance(words, list) and words:
        try:
            start_time = min(w.get("start", 0.0) for w in words)
            end_time = max(w.get("end", 0.0) for w in words)
        except Exception:
            start_time = 0.0
            end_time = 0.0
    sd = stt.SpeechData(language=language, text=transcript, start_time=start_time, end_time=end_time)
    return [sd] if transcript or words else []

