import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import librosa
import numpy as np
import resampy
import torch
import yaml
from packaging.version import Version

from .pyworld_compat import load_pyworld

pw = load_pyworld()


def _install_espnet_numpy_compat() -> None:
    """凍結版ESPnetの任意経路が使うNumPy別名を復元します。
    COEIROINKのVITS推論はこれらを使いませんが、凍結版ESPnetの前処理とGriffin-Lim実装が参照します。
    NumPy 1.26で別名が削除されたため、ESPnet自体を改変・同梱せずCoreのimport境界で組み込み型を設定します。
    """

    aliases = {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
    }
    for name, replacement in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, replacement)


_install_espnet_numpy_compat()


def _install_espnet_version_compat() -> None:
    """凍結版ESPnetが使う非推奨のLooseVersion実装を置き換えます。"""

    from setuptools._distutils import version as distutils_version

    distutils_version.LooseVersion = Version


_install_espnet_version_compat()


def _install_torch_weight_norm_compat() -> None:
    """凍結版ESPnetの呼び出しをTorchがサポートするparametrization APIへ接続します。"""

    from torch.nn.utils import parametrizations, parametrize

    modern_weight_norm = getattr(parametrizations, "weight_norm", None)
    if modern_weight_norm is None:
        return

    current_weight_norm = torch.nn.utils.weight_norm
    if getattr(current_weight_norm, "__coeiroink_modern_compat__", False):
        return

    legacy_remove_weight_norm = torch.nn.utils.remove_weight_norm

    def weight_norm(module, name="weight", dim=0):
        return modern_weight_norm(module, name=name, dim=dim)

    def remove_weight_norm(module, name="weight"):
        module_parametrizations = getattr(module, "parametrizations", None)
        if module_parametrizations is not None and name in module_parametrizations:
            parametrization_list = module_parametrizations[name]
            if any(
                type(item).__name__ == "_WeightNorm"
                for item in parametrization_list
            ):
                return parametrize.remove_parametrizations(
                    module, name, leave_parametrized=True
                )
        return legacy_remove_weight_norm(module, name=name)

    weight_norm.__coeiroink_modern_compat__ = True
    torch.nn.utils.weight_norm = weight_norm
    torch.nn.utils.remove_weight_norm = remove_weight_norm


_install_torch_weight_norm_compat()

def _load_espnet_dependencies():
    """互換シムを適用してから凍結版ESPnetをimportします。"""

    from espnet2.bin.tts_inference import Text2Speech
    from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
    from espnet2.text.token_id_converter import TokenIDConverter

    return Text2Speech, pyopenjtalk_g2p_prosody, TokenIDConverter


# 互換シムを先に適用してから読み込む必要があります。
# 凍結版ESPnetがimport中に変更済みのNumPyとdistutilsのシンボルを参照するためです。
Text2Speech, pyopenjtalk_g2p_prosody, TokenIDConverter = _load_espnet_dependencies()


class CoeiroCoreError(RuntimeError):
    """COEIROINK Coreが発生させる例外の基底クラスです。"""


class SpeakerInfoError(CoeiroCoreError):
    """MYCOEIROINK話者パッケージが不正な場合に発生します。"""


class StyleNotFoundError(CoeiroCoreError):
    """指定されたMYCOEIROINKスタイルがインストールされていない場合に発生します。"""


class AmbiguousStyleError(CoeiroCoreError):
    """旧形式のスタイルIDが複数の話者に一致した場合に発生します。"""


class ModelLoadError(CoeiroCoreError):
    """インストール済みのMYCOEIROINKモデルを読み込めない場合に発生します。"""


class SynthesisError(CoeiroCoreError):
    """読み込み済みのMYCOEIROINKモデルで推論に失敗した場合に発生します。"""


class InvalidSynthesisParameterError(CoeiroCoreError, ValueError):
    """音声合成パラメータが有効範囲外の場合に発生します。"""


@dataclass(frozen=True)
class ModelPath:
    model_path: Path
    config_path: Path


@dataclass(frozen=True)
class PredictionResult:
    """VITSの未加工推論結果と音響フレーム単位のトークン長です。"""

    wav: np.ndarray
    duration_frames: List[int]


ModelKey = Tuple[str, int]


class MetaManager:
    def __init__(
            self,
            speaker_info_dir: Union[str, Path] = Path("speaker_info")
    ):
        self.speaker_info_dir = Path(speaker_info_dir).expanduser().resolve()
        self.speaker_infos: List[Dict] = []
        self.model_map: Dict[ModelKey, ModelPath] = {}
        self.style_id_speaker_uuids: Dict[int, List[str]] = {}
        # 一意なスタイルIDだけを使う旧形式の呼び出し元向けに対応表を残します。
        self.id_model_map: Dict[int, ModelPath] = {}

        if not self.speaker_info_dir.is_dir():
            raise SpeakerInfoError(
                f"Speaker information directory does not exist: "
                f"{self.speaker_info_dir}"
            )

        # 隠しディレクトリとmacOSのメタデータ用ディレクトリは話者として扱いません。
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
                    f"metas.json is missing from MYCOEIROINK directory: "
                    f"{speaker_path}"
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
                raise SpeakerInfoError(
                    f"styles must be a non-empty list: {metas_path}"
                )
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
                # 1スタイルを1つの重みファイルに限定し、曖昧なモデル読込を防ぎます。
                model_paths = sorted(model_folder_path.glob("*.pth"))
                if not config_path.is_file():
                    raise SpeakerInfoError(
                        f"config.yaml is missing for style {style_id}: "
                        f"{config_path}"
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
                # YAMLの外形だけを検証し、モデル固有の設定解釈はESPnetへ委譲します。
                if not isinstance(config, dict):
                    raise SpeakerInfoError(
                        f"Model config must be a YAML mapping: {config_path}"
                    )

                styles.append({"name": style_name, "id": style_id})
                model_path = ModelPath(
                    model_path=model_paths[0].resolve(),
                    config_path=config_path.resolve(),
                )
                # 同じstyleIdを複数話者が持てるため、speakerUuidもモデルキーに含めます。
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

    def get_metas_dict(self) -> List[dict]:
        """話者メタデータの内部リストを返します。呼び出し元の変更は次回応答にも影響します。"""
        return self.speaker_infos

    def resolve_model_path(
            self,
            style_id: int,
            speaker_uuid: Optional[str] = None,
    ) -> Tuple[str, ModelPath]:
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
            raise StyleNotFoundError(
                f"MYCOEIROINK style is not installed: {style_id}"
            )
        if len(speaker_uuids) > 1:
            installed_speakers = ", ".join(speaker_uuids)
            raise AmbiguousStyleError(
                f"MYCOEIROINK style {style_id} is ambiguous across speakerUuid "
                f"values ({installed_speakers}); specify speaker_uuid"
            )

        resolved_speaker_uuid = speaker_uuids[0]
        return resolved_speaker_uuid, self.model_map[
            (resolved_speaker_uuid, style_id)
        ]

    def get_model_path(
            self,
            style_id: int,
            speaker_uuid: Optional[str] = None,
    ) -> ModelPath:
        """UUIDとスタイル、または一意な旧形式スタイルIDからモデルを解決します。"""
        return self.resolve_model_path(
            style_id=style_id,
            speaker_uuid=speaker_uuid,
        )[1]


class EspnetModel:
    def __init__(
            self,
            config_path: Path,
            model_path: Path,
            speed_scale=1.0,
            use_gpu=False
    ):
        device = 'cuda' if use_gpu else 'cpu'
        # モデル構造はリクエストごとの話速から独立させます。
        # 凍結版ESPnetは推論時にVITSのalphaを受け取れるため、モデルを再読み込みせず複数の話速に対応できます。
        self._speed_control_alpha = speed_scale
        self.tts_model = Text2Speech(
            config_path,
            model_path,
            device=device,
            seed=0,
            # リクエストごとの値はdecode_confから渡します。
            speed_control_alpha=1.0,
            # VITS専用の設定です。
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )
        # ESPnetがalphaを公開するのはVITS系とFastSpeech系のモデルだけです。
        # 将来別形式のモデルを追加した際に未対応の推論引数を渡さないよう、対応可否を明示的に判定します。
        self._supports_speed_control = "alpha" in self.tts_model.decode_conf
        if not self._supports_speed_control and speed_scale != 1.0:
            raise ValueError(
                "the loaded ESPnet model does not support speed control"
            )

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.token_id_converter = TokenIDConverter(
            token_list=config["token_list"],
            unk_symbol="<unk>",
        )

    def set_speed_control_alpha(self, speed_control_alpha: float) -> None:
        """重みを再読み込みせず次回推論のVITS話速を変更します。"""
        if not math.isfinite(speed_control_alpha) or speed_control_alpha <= 0:
            raise ValueError("speed_control_alpha must be a positive finite number")
        if not self._supports_speed_control and speed_control_alpha != 1.0:
            raise ValueError(
                "the loaded ESPnet model does not support speed control"
            )
        self._speed_control_alpha = speed_control_alpha

    @staticmethod
    def text2tokens(text: str) -> List[str]:
        return pyopenjtalk_g2p_prosody(text)

    def tokens2ids(
            self,
            tokens: Iterable[str]
    ) -> np.ndarray:
        return np.array(self.token_id_converter.tokens2ids(tokens), dtype=np.int64)

    def _prepare_text(
            self,
            text: Union[str, List[str], torch.Tensor, np.ndarray],
    ) -> Union[str, torch.Tensor, np.ndarray]:
        """公開APIのトークンリスト形式をCore境界で一度だけ変換します。
        文字列はESPnet側で解析し、トークン列だけTokenIDConverterでID列へ変換します。
        """
        if isinstance(text, list):
            return self.tokens2ids(text)
        return text

    def _run_inference(
            self,
            text: Union[str, torch.Tensor, np.ndarray],
            seed: int,
    ) -> dict:
        """サーバーの決定的なシードで凍結版ESPnetの推論を実行します。
        NumPyとTorchのプロセス全体の乱数状態を更新するため、呼び出し元のロック下で実行します。
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self._supports_speed_control:
            return self.tts_model(text)
        return self.tts_model(
            text,
            decode_conf={"alpha": self._speed_control_alpha},
        )

    @staticmethod
    def _waveform(output: dict) -> np.ndarray:
        """任意のdurationデータをPythonへ展開せず波形だけを取り出します。"""
        return output["wav"].view(-1).cpu().numpy()

    def make_voice(
            self,
            text: Union[str, List[str], torch.Tensor, np.ndarray],
            seed: int = 0
    ) -> np.ndarray:
        # 通常の合成とv1/predict経路ではモーラ長を使いません。
        # これらの経路でESPnetのdurationテンソルをPythonリストへ変換せず、推論結果と波形をそのまま保ちます。
        output = self._run_inference(self._prepare_text(text), seed=seed)
        return self._waveform(output)

    def make_voice_with_duration(
            self,
            text: Union[str, List[str], torch.Tensor, np.ndarray],
            seed: int = 0,
    ) -> PredictionResult:
        output = self._run_inference(self._prepare_text(text), seed=seed)
        wav = self._waveform(output)
        duration = output.get("duration")
        if duration is None:
            raise SynthesisError("ESPnet did not return token durations")
        duration_frames = [
            int(value)
            for value in duration.view(-1).detach().cpu().tolist()
        ]
        if any(value < 0 for value in duration_frames):
            raise SynthesisError("ESPnet returned a negative token duration")
        return PredictionResult(wav=wav, duration_frames=duration_frames)


class AudioManager:
    def __init__(
            self,
            fs=44100,
            use_gpu=False,
            speaker_info_dir: Union[str, Path] = Path("speaker_info"),
            cpu_num_threads: Optional[int] = None,
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
                # VITSのアライメントフォールバックはNumba、モデル推論はTorchを使います。
                # CPU使用数が明示された場合は両方のスレッドプールを揃えて過剰な入れ子並列を避けます。
                # PyTorchのプロセス間並列数は並列処理開始前にしか変更できないため、既存設定との衝突はエラーとして通知します。
                if torch.get_num_interop_threads() != cpu_num_threads:
                    torch.set_num_interop_threads(cpu_num_threads)
                import numba

                if numba.get_num_threads() != cpu_num_threads:
                    numba.set_num_threads(cpu_num_threads)

        self.fs = fs
        self.use_gpu = use_gpu
        self.meta_manager = MetaManager(speaker_info_dir=speaker_info_dir)
        # 正常に読み込めたモデルはすべて保持します。
        # CPU環境では多数のスタイルを導入できるため、スタイル切り替え時に重みを再利用します。
        # モデル数による追い出しは行わず、利用可能メモリを実質的な上限とします。
        self._loaded_models: Dict[ModelKey, EspnetModel] = {}
        self._current_speaker_uuid: Optional[str] = None
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
            minimum: Optional[float] = None,
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
            speaker_uuid: Optional[str] = None,
            force_reload: bool = False,
    ) -> EspnetModel:
        self._validate_speed_scale(speed_scale)
        resolved_speaker_uuid, model_paths = self.meta_manager.resolve_model_path(
            style_id=style_id,
            speaker_uuid=speaker_uuid,
        )
        model_key = (resolved_speaker_uuid, style_id)
        # 話者UUIDとスタイルIDが同じ場合だけ重みを再利用し、ファイル変更の自動検知は行いません。
        model = None if force_reload else self._loaded_models.get(model_key)
        if model is not None:
            set_speed = getattr(model, "set_speed_control_alpha", None)
            if callable(set_speed):
                set_speed(1 / speed_scale)
            self._current_speaker_uuid = resolved_speaker_uuid
            return model

        try:
            model = EspnetModel(
                model_path=model_paths.model_path,
                config_path=model_paths.config_path,
                speed_scale=1 / speed_scale,
                use_gpu=self.use_gpu,
            )
        except Exception as err:
            raise ModelLoadError(
                f"Failed to load MYCOEIROINK style {style_id} for speakerUuid "
                f"{resolved_speaker_uuid} from {model_paths.model_path}"
            ) from err

        self._loaded_models[model_key] = model
        self._current_speaker_uuid = resolved_speaker_uuid
        return model

    def initialize_speaker(
            self,
            style_id: int,
            speed_scale: float = 1.0,
            skip_reinit: bool = True,
            speaker_uuid: Optional[str] = None,
    ) -> None:
        self._validate_speed_scale(speed_scale)
        with self._synthesis_lock:
            self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
                force_reload=not skip_reinit,
            )

    def is_speaker_initialized(
            self,
            style_id: int,
            speaker_uuid: Optional[str] = None,
    ) -> bool:
        with self._synthesis_lock:
            try:
                resolved_speaker_uuid, _ = self.meta_manager.resolve_model_path(
                    style_id=style_id,
                    speaker_uuid=speaker_uuid,
                )
            except StyleNotFoundError:
                return False
            return (resolved_speaker_uuid, style_id) in self._loaded_models

    def synthesis(
            self,
            text: Union[str, List[str]],
            style_id: int,
            speed_scale: float = 1.0,
            volume_scale: float = 1.0,
            pitch_scale: float = 0,
            intonation_scale: float = 1.0,
            pre_phoneme_length: float = 0,
            post_phoneme_length: float = 0,
            output_sampling_rate: int = 44100,
            speaker_uuid: Optional[str] = None,
    ):
        self._validate_speed_scale(speed_scale)
        self._validate_finite_scale("volume_scale", volume_scale, minimum=0)
        self._validate_finite_scale("pitch_scale", pitch_scale)
        self._validate_finite_scale("intonation_scale", intonation_scale, minimum=0)
        self._validate_finite_scale(
            "pre_phoneme_length", pre_phoneme_length, minimum=0
        )
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

        # ESPnetはプロセス全体の乱数状態を変更します。
        # 並行リクエストが別スタイルを選択したり決定的な合成を干渉したりしないよう、モデル選択と推論を同じ排他区間で行います。
        with self._synthesis_lock:
            model = self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
            )
            active_speaker_uuid = self._current_speaker_uuid
            try:
                model_input = text if isinstance(text, str) else model.tokens2ids(text)
                wav = model.make_voice(model_input)
            except Exception as err:
                raise SynthesisError(
                    f"Failed to synthesize MYCOEIROINK style {style_id} for "
                    f"speakerUuid {active_speaker_uuid}"
                ) from err

        # トリミングやリサンプリングはモデル状態を触らないため、推論ロックの外で実行します。
        try:
            wav = self.trim(wav)
            if volume_scale != 1:
                wav = self.volume(wav, volume_scale)
            if pitch_scale != 0 or intonation_scale != 1:
                wav = self.pitch_intonation(
                    wav, self.fs, pitch_scale, intonation_scale
                )
            if pre_phoneme_length != 0 or post_phoneme_length != 0:
                wav = self.sil(
                    wav, self.fs, pre_phoneme_length, post_phoneme_length
                )
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
            text: Union[str, List[str]],
            style_id: int,
            speed_scale: float = 1.0,
            speaker_uuid: Optional[str] = None,
    ) -> PredictionResult:
        """トリミング前のモデル波形とトークンごとのフレーム長を返します。
        COEIROINK v2は波形の後処理前の推論結果を別APIで公開するため、この処理もCoreの排他制御下で実行します。
        """

        self._validate_speed_scale(speed_scale)
        with self._synthesis_lock:
            model = self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
            )
            active_speaker_uuid = self._current_speaker_uuid
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
            text: Union[str, List[str]],
            style_id: int,
            speed_scale: float = 1.0,
            speaker_uuid: Optional[str] = None,
    ) -> np.ndarray:
        """v2 predict API向けにトリミング前のモデル波形を返します。"""

        self._validate_speed_scale(speed_scale)
        with self._synthesis_lock:
            model = self._load_model(
                style_id=style_id,
                speed_scale=speed_scale,
                speaker_uuid=speaker_uuid,
            )
            active_speaker_uuid = self._current_speaker_uuid
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
        # ピッチを半音単位で移動します。
        if pitch_scale != 0:
            f0 *= 2 ** pitch_scale
        # 抑揚は有声フレームの分布だけを調整します。
        if intonation_scale != 1:
            # WORLDは無声音フレームをF0=0で表すため、平均値に含めず抑揚処理で有音化もしません。
            # VOICEVOX系の処理も同じ理由で有声モーラのピッチだけを調整します。
            voiced = f0 > 0
            if np.any(voiced):
                voiced_f0 = f0[voiced]
                mean = float(voiced_f0.mean())
                deviation = voiced_f0 - mean
                f0[voiced] = mean + deviation * intonation_scale
        return pw.synthesize(f0, sp, ap, fs).astype(np.float32)

    @staticmethod
    def sil(wav, fs, pre_phoneme_length, post_phoneme_length):
        pre_pause = np.zeros(int(fs * pre_phoneme_length), dtype=wav.dtype)
        post_pause = np.zeros(int(fs * post_phoneme_length), dtype=wav.dtype)
        return np.concatenate([pre_pause, wav, post_pause], 0)

    @staticmethod
    def resampling(wav, fs, output_sampling_rate):
        # Resampyの並列カーネルは1次元波形では逐次カーネルと数値的に同じで、44.1kHz以外の応答を高速に処理できます。
        return resampy.resample(
            wav,
            fs,
            output_sampling_rate,
            filter="kaiser_fast",
            parallel=True,
        )

    # WORLD処理の基本的な呼び出し順を確認するための公開サンプルです。
    # https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/blob/3a7c99a32c717deb8e66bde64b5e60b1a4afce79/demo/demo.py
    @staticmethod
    def get_world(x, fs):
        _f0_h, t_h = pw.harvest(x, fs)
        f0_h = pw.stonemask(x, _f0_h, t_h, fs)
        sp_h = pw.cheaptrick(x, f0_h, t_h, fs)
        ap_h = pw.d4c(x, f0_h, t_h, fs)
        return f0_h, sp_h, ap_h
