import json
import math
import threading
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import resampy
import torch
import yaml

from .devices import DeviceBackend, DeviceSelection, normalize_backend, resolve_device
from .opencl import OpenCLVitsMode
from .pyworld_compat import load_pyworld


@lru_cache(maxsize=1)
def _load_text_to_speech():
    """モデル生成時にだけESPnetの推論スタックを読み込む。"""
    from espnet2.bin.tts_inference import Text2Speech

    return Text2Speech


@lru_cache(maxsize=1)
def _load_g2p_prosody():
    """文字変換だけの呼び出しで推論依存を読み込まない。"""
    from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody

    return pyopenjtalk_g2p_prosody


@lru_cache(maxsize=1)
def _load_token_id_converter():
    """モデル生成時にトークンID変換器を読み込む。"""
    from espnet2.text.token_id_converter import TokenIDConverter

    return TokenIDConverter


class CoeiroCoreError(RuntimeError):
    """COEIROINK Coreが送出する例外の基底クラス。"""


class SpeakerInfoError(CoeiroCoreError):
    """MYCOEIROINK話者パッケージが不正な場合の例外。"""


class StyleNotFoundError(CoeiroCoreError):
    """指定されたMYCOEIROINKスタイルが未導入の場合の例外。"""


class AmbiguousStyleError(CoeiroCoreError):
    """従来形式のスタイルIDが複数話者に一致する場合の例外。"""


class ModelLoadError(CoeiroCoreError):
    """導入済みMYCOEIROINKモデルを読み込めない場合の例外。"""


class SynthesisError(CoeiroCoreError):
    """読み込み済みMYCOEIROINKモデルの推論に失敗した場合の例外。"""


class InvalidSynthesisParameterError(CoeiroCoreError, ValueError):
    """音声合成パラメーターが有効範囲外の場合の例外。"""


@dataclass(frozen=True)
class ModelPath:
    model_path: Path
    config_path: Path


@dataclass(frozen=True)
class PredictionResult:
    """VITSの生波形と音響フレーム単位のトークン継続長。"""

    wav: np.ndarray
    duration_frames: list[int]


ModelKey = tuple[str, int]


def _resolve_inference_device(
    *,
    device: str | DeviceBackend | DeviceSelection | None,
    use_gpu: bool | None,
    device_index: int,
    opencl_platform_index: int,
) -> DeviceSelection:
    """新しいdevice指定と旧use_gpu指定を一か所で整合させる。"""

    if isinstance(device, DeviceSelection):
        if use_gpu is not None or device_index != 0 or opencl_platform_index != 0:
            raise ValueError(
                "a resolved device cannot be combined with use_gpu or device indices"
            )
        return device

    if use_gpu is not None:
        legacy_backend = DeviceBackend.CUDA if use_gpu else DeviceBackend.CPU
        if device is not None and normalize_backend(device) is not legacy_backend:
            raise ValueError(
                f"device={normalize_backend(device).value!r} conflicts with "
                f"use_gpu={use_gpu!r}"
            )
        device = legacy_backend

    return resolve_device(
        device or DeviceBackend.CPU,
        device_index=device_index,
        platform_index=opencl_platform_index,
    )


class MetaManager:
    """MYCOEIROINKのメタデータとモデル配置を起動時に検証し、話者UUIDとスタイルIDの索引を構築する。"""

    def __init__(self, speaker_info_dir: str | Path = Path("speaker_info")):
        self.speaker_info_dir = Path(speaker_info_dir).expanduser().resolve()
        self.speaker_infos: list[dict] = []
        self.model_map: dict[ModelKey, ModelPath] = {}
        self.style_id_speaker_uuids: dict[int, list[str]] = {}
        # 話者間で一意な従来形式のスタイルIDを使う呼び出し元との互換性を保つ。
        self.id_model_map: dict[int, ModelPath] = {}

        if not self.speaker_info_dir.is_dir():
            raise SpeakerInfoError(
                f"Speaker information directory does not exist: {self.speaker_info_dir}"
            )

        speaker_paths = sorted(
            path
            for path in self.speaker_info_dir.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and path.name != "__MACOSX"
        )
        uuid_set = set()
        for speaker_path in speaker_paths:
            metas_path = speaker_path / "metas.json"
            if not metas_path.is_file():
                raise SpeakerInfoError(
                    f"metas.json is missing from MYCOEIROINK directory: {speaker_path}"
                )
            try:
                meta = json.loads(metas_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as err:
                raise SpeakerInfoError(
                    f"Failed to read MYCOEIROINK metadata: {metas_path}"
                ) from err

            if not isinstance(meta, dict):
                raise SpeakerInfoError(
                    f"MYCOEIROINK metadata must be a JSON object: {metas_path}"
                )

            uuid = meta.get("speakerUuid")
            speaker_name = meta.get("speakerName")
            meta_styles = meta.get("styles")
            if not isinstance(uuid, str) or not uuid.strip():
                raise SpeakerInfoError(
                    f"speakerUuid is missing or invalid: {metas_path}"
                )
            if not isinstance(speaker_name, str) or not speaker_name.strip():
                raise SpeakerInfoError(
                    f"speakerName is missing or invalid: {metas_path}"
                )
            if not isinstance(meta_styles, list) or not meta_styles:
                raise SpeakerInfoError(f"styles must be a non-empty list: {metas_path}")
            if uuid in uuid_set:
                raise SpeakerInfoError(f"Duplicate speakerUuid: {uuid}")
            uuid_set.add(uuid)

            styles = []
            speaker_style_ids = set()
            for style in meta_styles:
                if not isinstance(style, dict):
                    raise SpeakerInfoError(
                        f"Each style must be a JSON object: {metas_path}"
                    )
                style_id = style.get("styleId")
                style_name = style.get("styleName")
                if isinstance(style_id, bool) or not isinstance(style_id, int):
                    raise SpeakerInfoError(
                        f"styleId is missing or invalid: {metas_path}"
                    )
                if not isinstance(style_name, str) or not style_name.strip():
                    raise SpeakerInfoError(
                        f"styleName is missing or invalid: {metas_path}"
                    )
                if style_id in speaker_style_ids:
                    raise SpeakerInfoError(
                        f"Duplicate styleId {style_id} for speakerUuid {uuid}"
                    )
                speaker_style_ids.add(style_id)

                model_folder_path = speaker_path / "model" / str(style_id)
                config_path = model_folder_path / "config.yaml"
                model_paths = sorted(model_folder_path.glob("*.pth"))
                if not config_path.is_file():
                    raise SpeakerInfoError(
                        f"config.yaml is missing for style {style_id}: {config_path}"
                    )
                if len(model_paths) != 1:
                    raise SpeakerInfoError(
                        f"Style {style_id} must contain exactly one .pth model; "
                        f"found {len(model_paths)} in {model_folder_path}"
                    )

                try:
                    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, yaml.YAMLError) as err:
                    raise SpeakerInfoError(
                        f"Failed to read model config: {config_path}"
                    ) from err
                if not isinstance(config, dict):
                    raise SpeakerInfoError(
                        f"Model config must be a YAML mapping: {config_path}"
                    )

                styles.append({"name": style_name, "id": style_id})
                model_path = ModelPath(
                    model_path=model_paths[0].resolve(),
                    config_path=config_path.resolve(),
                )
                model_key = (uuid, style_id)
                self.model_map[model_key] = model_path
                self.style_id_speaker_uuids.setdefault(style_id, []).append(uuid)

            version = meta.get("version", "0.0.1")
            if not isinstance(version, str):
                version = str(version)
            speaker_info = {
                "name": speaker_name,
                "speaker_uuid": uuid,
                "styles": styles,
                "version": version,
            }
            self.speaker_infos.append(speaker_info)
        self.speaker_infos = sorted(
            self.speaker_infos,
            key=lambda speaker: (
                min(style["id"] for style in speaker["styles"]),
                speaker["speaker_uuid"],
            ),
        )
        for style_id, speaker_uuids in self.style_id_speaker_uuids.items():
            speaker_uuids.sort()
            if len(speaker_uuids) == 1:
                self.id_model_map[style_id] = self.model_map[
                    (speaker_uuids[0], style_id)
                ]

    def get_metas_dict(self) -> list[dict]:
        return self.speaker_infos

    def resolve_model_path(
        self,
        style_id: int,
        speaker_uuid: str | None = None,
    ) -> tuple[str, ModelPath]:
        """UUID指定時は完全一致、未指定時は全話者で一意なスタイルIDだけを解決する。"""

        if speaker_uuid is not None:
            if not isinstance(speaker_uuid, str) or not speaker_uuid.strip():
                raise StyleNotFoundError(
                    f"MYCOEIROINK speakerUuid is invalid: {speaker_uuid!r}"
                )
            model_key = (speaker_uuid, style_id)
            try:
                return speaker_uuid, self.model_map[model_key]
            except KeyError as err:
                raise StyleNotFoundError(
                    f"MYCOEIROINK style is not installed for speakerUuid "
                    f"{speaker_uuid}: {style_id}"
                ) from err

        speaker_uuids = self.style_id_speaker_uuids.get(style_id, [])
        if not speaker_uuids:
            raise StyleNotFoundError(f"MYCOEIROINK style is not installed: {style_id}")
        if len(speaker_uuids) > 1:
            installed_speakers = ", ".join(speaker_uuids)
            raise AmbiguousStyleError(
                f"MYCOEIROINK style {style_id} is ambiguous across speakerUuid "
                f"values ({installed_speakers}); specify speaker_uuid"
            )

        resolved_speaker_uuid = speaker_uuids[0]
        return resolved_speaker_uuid, self.model_map[(resolved_speaker_uuid, style_id)]

    def get_model_path(
        self,
        style_id: int,
        speaker_uuid: str | None = None,
    ) -> ModelPath:
        """話者UUIDとスタイル、または一意な従来形式IDからモデルを解決する。"""
        return self.resolve_model_path(
            style_id=style_id,
            speaker_uuid=speaker_uuid,
        )[1]


class EspnetModel:
    """ESPnetのMYCOEIROINKモデルと、そのモデル固有のトークン変換器を保持する。"""

    def __init__(
        self,
        config_path: Path,
        model_path: Path,
        speed_scale=1.0,
        use_gpu: bool | None = None,
        device: str | DeviceBackend | DeviceSelection | None = None,
        device_index: int = 0,
        opencl_platform_index: int = 0,
    ):
        self.device_selection = _resolve_inference_device(
            device=device,
            use_gpu=use_gpu,
            device_index=device_index,
            opencl_platform_index=opencl_platform_index,
        )
        Text2Speech = _load_text_to_speech()
        TokenIDConverter = _load_token_id_converter()
        # 現行ESPnetはVITSのalphaを推論時に受け取るため、速度変更でモデルを再読込する必要はない。
        self._speed_control_alpha = speed_scale
        self.tts_model = Text2Speech(
            config_path,
            model_path,
            # ESPnetはtypeguardでdeviceをstrに限定するため、検証済みTorchデバイスの標準名を渡す。
            device=str(self.device_selection.runtime_device),
            seed=0,
            # リクエスト固有の値はdecode_confで渡す。
            speed_control_alpha=1.0,
            # VITS専用
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )
        # alpha非対応モデルへ推論引数を誤って渡さないよう、モデル読込時に対応状況を確定する。
        self._supports_speed_control = "alpha" in self.tts_model.decode_conf
        if not self._supports_speed_control and speed_scale != 1.0:
            raise ValueError("the loaded ESPnet model does not support speed control")

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.token_id_converter = TokenIDConverter(
            token_list=config["token_list"],
            unk_symbol="<unk>",
        )

    def set_speed_control_alpha(self, speed_control_alpha: float) -> None:
        """重みを再読込せず、次回推論のVITS速度を変更する。"""
        if not math.isfinite(speed_control_alpha) or speed_control_alpha <= 0:
            raise ValueError("speed_control_alpha must be a positive finite number")
        if not self._supports_speed_control and speed_control_alpha != 1.0:
            raise ValueError("the loaded ESPnet model does not support speed control")
        self._speed_control_alpha = speed_control_alpha

    @staticmethod
    def text2tokens(text: str) -> list[str]:
        pyopenjtalk_g2p_prosody = _load_g2p_prosody()
        return pyopenjtalk_g2p_prosody(text)

    def tokens2ids(self, tokens: Iterable[str]) -> np.ndarray:
        return np.array(self.token_id_converter.tokens2ids(tokens), dtype=np.int64)

    def _prepare_text(
        self,
        text: str | list[str] | torch.Tensor | np.ndarray,
    ) -> str | torch.Tensor | np.ndarray:
        """公開APIのトークン列をCore境界で一度だけIDへ変換する。"""
        if isinstance(text, list):
            return self.tokens2ids(text)
        return text

    def _run_inference(
        self,
        text: str | torch.Tensor | np.ndarray,
        seed: int,
    ) -> dict:
        """サーバーで指定されたseedを使ってESPnet推論を実行する。"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        dispatch_mode = (
            OpenCLVitsMode()
            if self.device_selection.backend is DeviceBackend.OPENCL
            else nullcontext()
        )
        with dispatch_mode:
            if not self._supports_speed_control:
                return self.tts_model(text)
            return self.tts_model(
                text,
                decode_conf={"alpha": self._speed_control_alpha},
            )

    @staticmethod
    def _waveform(output: dict) -> np.ndarray:
        """任意の継続長データを実体化せず波形だけを取り出す。"""
        return output["wav"].view(-1).cpu().numpy()

    def make_voice(
        self, text: str | list[str] | torch.Tensor | np.ndarray, seed: int = 0
    ) -> np.ndarray:
        # 通常合成とv1/predictでは継続長を使わないため、テンソルからリストへの変換を省く。
        output = self._run_inference(self._prepare_text(text), seed=seed)
        return self._waveform(output)

    def make_voice_with_duration(
        self,
        text: str | list[str] | torch.Tensor | np.ndarray,
        seed: int = 0,
    ) -> PredictionResult:
        output = self._run_inference(self._prepare_text(text), seed=seed)
        wav = self._waveform(output)
        duration = output.get("duration")
        if duration is None:
            raise SynthesisError("ESPnet did not return token durations")
        duration_frames = [
            int(value) for value in duration.view(-1).detach().cpu().tolist()
        ]
        if any(value < 0 for value in duration_frames):
            raise SynthesisError("ESPnet returned a negative token duration")
        return PredictionResult(wav=wav, duration_frames=duration_frames)


class AudioManager:
    """モデル選択・推論・後処理を管理し、直近に使ったMYCOEIROINKモデルを再利用する。"""

    def __init__(
        self,
        fs=44100,
        use_gpu: bool | None = None,
        device: str | DeviceBackend | DeviceSelection | None = None,
        device_index: int = 0,
        opencl_platform_index: int = 0,
        speaker_info_dir: str | Path = Path("speaker_info"),
        cpu_num_threads: int | None = None,
    ):
        if isinstance(fs, bool) or not isinstance(fs, int) or fs <= 0:
            raise ValueError("fs must be a positive integer")
        if cpu_num_threads is not None:
            if (
                isinstance(cpu_num_threads, bool)
                or not isinstance(cpu_num_threads, int)
                or cpu_num_threads < 0
            ):
                raise ValueError("cpu_num_threads must be zero or a positive integer")
            if cpu_num_threads > 0:
                torch.set_num_threads(cpu_num_threads)
                # VITSのアライメント処理はNumba、モデルはTorchを使うため、両方のスレッド数を同じCPU予算に制限する。
                # inter-op数は並列処理開始後に変更できないので、先行するAudioManagerとの設定競合は例外として通知する。
                if torch.get_num_interop_threads() != cpu_num_threads:
                    torch.set_num_interop_threads(cpu_num_threads)
                import numba

                if numba.get_num_threads() != cpu_num_threads:
                    numba.set_num_threads(cpu_num_threads)

        self.fs = fs
        self.device_selection = _resolve_inference_device(
            device=device,
            use_gpu=use_gpu,
            device_index=device_index,
            opencl_platform_index=opencl_platform_index,
        )
        self.device = self.device_selection.backend.value
        self.use_gpu = self.device_selection.backend is not DeviceBackend.CPU
        self.meta_manager = MetaManager(speaker_info_dir=speaker_info_dir)
        self.previous_model_key: ModelKey | None = None
        self.previous_style_id: int | None = None
        self.previous_speaker_uuid: str | None = None
        self.previous_speed_scale: float | None = None
        self.current_speaker_model: EspnetModel | None = None
        self._synthesis_lock = threading.RLock()

    @staticmethod
    def _validate_speed_scale(speed_scale: float) -> None:
        if (
            isinstance(speed_scale, bool)
            or not isinstance(speed_scale, (int, float))
            or not math.isfinite(speed_scale)
            or speed_scale <= 0
        ):
            raise InvalidSynthesisParameterError(
                "speed_scale must be a positive finite number"
            )

    @staticmethod
    def _validate_finite_scale(
        name: str,
        value: float,
        minimum: float | None = None,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (minimum is not None and value < minimum)
        ):
            suffix = (
                "finite number"
                if minimum is None
                else f"finite number greater than or equal to {minimum}"
            )
            raise InvalidSynthesisParameterError(f"{name} must be a {suffix}")

    def _load_model(
        self,
        style_id: int,
        speed_scale: float,
        speaker_uuid: str | None = None,
        force_reload: bool = False,
    ) -> EspnetModel:
        """同じモデルは速度設定だけを更新し、話者またはスタイルが変わった場合に重みを読み直す。"""

        self._validate_speed_scale(speed_scale)
        resolved_speaker_uuid, model_paths = self.meta_manager.resolve_model_path(
            style_id=style_id,
            speaker_uuid=speaker_uuid,
        )
        model_key = (resolved_speaker_uuid, style_id)
        if (
            not force_reload
            and self.current_speaker_model is not None
            and self.previous_model_key == model_key
        ):
            set_speed = getattr(
                self.current_speaker_model, "set_speed_control_alpha", None
            )
            if callable(set_speed):
                set_speed(1 / speed_scale)
            self.previous_speed_scale = speed_scale
            return self.current_speaker_model

        try:
            model = EspnetModel(
                model_path=model_paths.model_path,
                config_path=model_paths.config_path,
                speed_scale=1 / speed_scale,
                device=self.device_selection,
            )
        except Exception as err:
            raise ModelLoadError(
                f"Failed to load MYCOEIROINK style {style_id} for speakerUuid "
                f"{resolved_speaker_uuid} from {model_paths.model_path}"
            ) from err

        self.current_speaker_model = model
        self.previous_model_key = model_key
        self.previous_style_id = style_id
        self.previous_speaker_uuid = resolved_speaker_uuid
        self.previous_speed_scale = speed_scale
        return model

    def initialize_speaker(
        self,
        style_id: int,
        speed_scale: float = 1.0,
        skip_reinit: bool = True,
        speaker_uuid: str | None = None,
    ) -> None:
        """公開APIから指定されたモデルを読み込み、必要なら既存モデルを強制的に再初期化する。"""

        with self._synthesis_lock:
            resolved_speaker_uuid, _ = self.meta_manager.resolve_model_path(
                style_id=style_id,
                speaker_uuid=speaker_uuid,
            )
            model_key = (resolved_speaker_uuid, style_id)
            if (
                skip_reinit
                and self.current_speaker_model is not None
                and self.previous_model_key == model_key
            ):
                set_speed = getattr(
                    self.current_speaker_model, "set_speed_control_alpha", None
                )
                if callable(set_speed):
                    set_speed(1 / speed_scale)
                self.previous_speed_scale = speed_scale
                return
            self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
                force_reload=not skip_reinit,
            )

    def is_speaker_initialized(
        self,
        style_id: int,
        speaker_uuid: str | None = None,
    ) -> bool:
        with self._synthesis_lock:
            try:
                resolved_speaker_uuid, _ = self.meta_manager.resolve_model_path(
                    style_id=style_id,
                    speaker_uuid=speaker_uuid,
                )
            except StyleNotFoundError:
                return False
            return (
                self.current_speaker_model is not None
                and self.previous_model_key == (resolved_speaker_uuid, style_id)
            )

    def synthesis(
        self,
        text: str | list[str],
        style_id: int,
        speed_scale: float = 1.0,
        volume_scale: float = 1.0,
        pitch_scale: float = 0,
        intonation_scale: float = 1.0,
        pre_phoneme_length: float = 0,
        post_phoneme_length: float = 0,
        output_sampling_rate: int = 44100,
        speaker_uuid: str | None = None,
    ):
        """モデル推論後に音量・F0・無音長・サンプリングレートを指定順で適用する。"""

        self._validate_speed_scale(speed_scale)
        self._validate_finite_scale("volume_scale", volume_scale, minimum=0)
        self._validate_finite_scale("pitch_scale", pitch_scale)
        self._validate_finite_scale("intonation_scale", intonation_scale, minimum=0)
        self._validate_finite_scale("pre_phoneme_length", pre_phoneme_length, minimum=0)
        self._validate_finite_scale(
            "post_phoneme_length", post_phoneme_length, minimum=0
        )
        if (
            isinstance(output_sampling_rate, bool)
            or not isinstance(output_sampling_rate, int)
            or output_sampling_rate <= 0
        ):
            raise InvalidSynthesisParameterError(
                "output_sampling_rate must be a positive integer"
            )

        # ESPnetはプロセス全体の乱数状態を変更するため、モデル切替と推論を同じ排他区間に置く。
        with self._synthesis_lock:
            model = self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
            )
            active_speaker_uuid = self.previous_speaker_uuid
            try:
                model_input = text if isinstance(text, str) else model.tokens2ids(text)
                wav = model.make_voice(model_input)
            except Exception as err:
                raise SynthesisError(
                    f"Failed to synthesize MYCOEIROINK style {style_id} for "
                    f"speakerUuid {active_speaker_uuid}"
                ) from err

        try:
            wav = self.trim(wav)
            if volume_scale != 1:
                wav = self.volume(wav, volume_scale)
            if pitch_scale != 0 or intonation_scale != 1:
                wav = self.pitch_intonation(wav, self.fs, pitch_scale, intonation_scale)
            if pre_phoneme_length != 0 or post_phoneme_length != 0:
                wav = self.sil(wav, self.fs, pre_phoneme_length, post_phoneme_length)
            if output_sampling_rate != self.fs:
                wav = self.resampling(wav, self.fs, output_sampling_rate)
        except Exception as err:
            raise SynthesisError(
                f"Failed to post-process MYCOEIROINK style {style_id} for "
                f"speakerUuid {active_speaker_uuid}"
            ) from err

        return wav

    def predict_with_duration(
        self,
        text: str | list[str],
        style_id: int,
        speed_scale: float = 1.0,
        speaker_uuid: str | None = None,
    ) -> PredictionResult:
        """未トリムのモデル波形とトークンごとのフレーム長を返す。

        COEIROINK v2では生の推論と波形後処理を分離する。Core内に保持することで、ESPnetの乱数状態と単一モデルキャッシュを従来合成と同じロックで保護できる。
        """

        self._validate_speed_scale(speed_scale)
        with self._synthesis_lock:
            model = self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
            )
            active_speaker_uuid = self.previous_speaker_uuid
            try:
                model_input = text if isinstance(text, str) else model.tokens2ids(text)
                return model.make_voice_with_duration(model_input)
            except SynthesisError:
                raise
            except Exception as err:
                raise SynthesisError(
                    f"Failed to predict MYCOEIROINK style {style_id} for "
                    f"speakerUuid {active_speaker_uuid}"
                ) from err

    def predict(
        self,
        text: str | list[str],
        style_id: int,
        speed_scale: float = 1.0,
        speaker_uuid: str | None = None,
    ) -> np.ndarray:
        """v2 predict API向けに未トリムのモデル波形を返す。"""

        self._validate_speed_scale(speed_scale)
        with self._synthesis_lock:
            model = self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
            )
            active_speaker_uuid = self.previous_speaker_uuid
            try:
                model_input = text if isinstance(text, str) else model.tokens2ids(text)
                return model.make_voice(model_input)
            except Exception as err:
                raise SynthesisError(
                    f"Failed to predict MYCOEIROINK style {style_id} for "
                    f"speakerUuid {active_speaker_uuid}"
                ) from err

    @staticmethod
    def trim(wav):
        return librosa.effects.trim(wav, top_db=30)[0]

    @staticmethod
    def volume(wav, volume_scale):
        return wav * volume_scale

    @staticmethod
    def pitch_intonation(wav, fs, pitch_scale, intonation_scale):
        f0, sp, ap = AudioManager.get_world(wav.astype(np.float64), fs)
        # 音高
        if pitch_scale != 0:
            f0 *= 2**pitch_scale
        # 抑揚
        if intonation_scale != 1:
            # WORLDは無声音をF0=0で表すため、有声音の平均や抑揚倍率の計算には含めない。
            voiced = f0 > 0
            if np.any(voiced):
                voiced_f0 = f0[voiced]
                mean = float(voiced_f0.mean())
                deviation = voiced_f0 - mean
                f0[voiced] = mean + deviation * intonation_scale
        return load_pyworld().synthesize(f0, sp, ap, fs).astype(np.float32)

    @staticmethod
    def sil(wav, fs, pre_phoneme_length, post_phoneme_length):
        pre_pause = np.zeros(int(fs * pre_phoneme_length), dtype=wav.dtype)
        post_pause = np.zeros(int(fs * post_phoneme_length), dtype=wav.dtype)
        return np.concatenate([pre_pause, wav, post_pause], 0)

    @staticmethod
    def resampling(wav, fs, output_sampling_rate):
        # 1次元波形では並列カーネルも逐次版と同じ結果になるため、44.1 kHz以外への変換だけ並列化する。
        return resampy.resample(
            wav,
            fs,
            output_sampling_rate,
            filter="kaiser_fast",
            parallel=True,
        )

    # https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/blob/3a7c99a32c717deb8e66bde64b5e60b1a4afce79/demo/demo.py
    @staticmethod
    def get_world(x, fs):
        world = load_pyworld()
        _f0_h, t_h = world.harvest(x, fs)
        f0_h = world.stonemask(x, _f0_h, t_h, fs)
        sp_h = world.cheaptrick(x, f0_h, t_h, fs)
        ap_h = world.d4c(x, f0_h, t_h, fs)
        return f0_h, sp_h, ap_h
