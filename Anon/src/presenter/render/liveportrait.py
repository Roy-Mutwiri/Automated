"""LivePortrait backend: renders an `AvatarPose` as a photoreal frame.

## Why this works without a driving video

LivePortrait's motion representation is explicit:

    x_d = s_d · (x_c · R_d + δ_d) + t_d

`x_c` canonical keypoints, `R` head rotation, `δ` expression deltas, `s` scale,
`t` translation. Normally `R`, `δ`, `s`, `t` are extracted from each frame of a
driving video. Here they are **synthesised** from the behaviour engine, so
there is no driving clip and therefore nothing that can loop. That property is
the reason this backend was chosen over the faster diffusion alternatives.

## What runs once vs. per frame

Once, at startup: face detection, cropping, appearance-feature extraction and
source keypoint extraction. Per frame: build the driving keypoints, stitch, and
warp/decode. The expensive identity work is paid for exactly once, which is
what makes real-time feasible and also what keeps identity fixed - `f_s` and
`x_c` never change, so the face cannot drift.

## Detection: MediaPipe, not InsightFace

LivePortrait's stock cropper uses InsightFace, whose **models are licensed for
non-commercial research only** - the one real licensing hazard in the stack.
Detection runs a single time at startup, so substituting MediaPipe
(Apache-2.0) costs nothing and keeps the pipeline commercially clean.
LivePortrait's own `landmark.onnx` is used for the 203-point landmarks that
eye retargeting needs; that model is not InsightFace and carries no such
restriction.

## Calibration status

Head pose, scale and translation map onto verified quantities. Blink uses
LivePortrait's dedicated `retarget_eye` network - the correct mechanism rather
than a guess at which keypoints are eyelids. **Gaze and brow directions in
expression space are NOT verified**: LivePortrait has no native gaze control,
and the semantic meaning of individual `exp` dimensions is not documented.
Those live in `calibration.py` with measured-or-not clearly marked, and
`tools/calibrate_expression.py` probes them empirically. Do not trust an
uncalibrated gaze axis because it looks plausible in one frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ..types import AvatarPose
from .base import RendererInfo

__all__ = ["LivePortraitRenderer"]


def _add_liveportrait_to_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class LivePortraitRenderer:
    """Photoreal renderer driven by procedurally generated motion parameters."""

    def __init__(
        self,
        source_image: str | Path,
        liveportrait_root: str | Path,
        device: str = "cuda",
        use_half: bool = True,
        output_size: tuple[int, int] = (1280, 720),
        paste_back: bool = True,
    ) -> None:
        self.root = Path(liveportrait_root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"LivePortrait not found at {self.root}")
        _add_liveportrait_to_path(self.root)

        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.crop import crop_image
        from src.utils.human_landmark_runner import LandmarkRunner

        self._crop_image = crop_image
        self.device = device
        self.output_size = output_size
        self.paste_back = paste_back

        weights = self.root / "pretrained_weights"
        cfg = InferenceConfig(
            checkpoint_F=str(weights / "liveportrait/base_models/appearance_feature_extractor.pth"),
            checkpoint_M=str(weights / "liveportrait/base_models/motion_extractor.pth"),
            checkpoint_G=str(weights / "liveportrait/base_models/spade_generator.pth"),
            checkpoint_W=str(weights / "liveportrait/base_models/warping_module.pth"),
            checkpoint_S=str(weights / "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
            flag_use_half_precision=use_half,
            device_id=0,
        )
        self.cfg = cfg
        self.wrapper = LivePortraitWrapper(inference_cfg=cfg)

        self.landmark_runner = LandmarkRunner(
            ckpt_path=str(weights / "liveportrait/landmark.onnx"),
            onnx_provider="cuda" if device == "cuda" else "cpu",
            device_id=0,
        )
        self.landmark_runner.warmup()

        self._prepare_source(Path(source_image))

        self._info = RendererInfo(
            name="liveportrait",
            resolution=output_size,
            device=f"{device}{' fp16' if use_half else ' fp32'}",
            photoreal=True,
            notes=f"source={Path(source_image).name}",
        )

    # -- one-time identity preparation --------------------------------------
    def _detect_landmarks_mediapipe(self, img_rgb: np.ndarray) -> np.ndarray:
        """Initial face landmarks via MediaPipe. Replaces InsightFace detection.

        Only needs to be roughly right: the output seeds LivePortrait's own
        landmark model, which then refines to the 203-point set actually used.
        """
        import mediapipe as mp

        h, w = img_rgb.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            result = mesh.process(img_rgb)
        if not result.multi_face_landmarks:
            raise RuntimeError(
                "MediaPipe found no face in the source image. The portrait must "
                "contain one clearly visible, roughly front-facing face."
            )
        lm = result.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def _prepare_source(self, path: Path) -> None:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"could not read source image {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.source_full = img_bgr

        seed_lmk = self._detect_landmarks_mediapipe(img_rgb)
        # Refine to LivePortrait's own 203-point format in original coordinates.
        lmk = self.landmark_runner.run(img_rgb, seed_lmk)

        # Source crop parameters are LivePortrait's documented defaults for
        # still images: a wider crop than the driving crop, so pasting back
        # into the full frame has margin to work with.
        crop = self._crop_image(
            img_rgb, lmk, dsize=512, scale=2.3, vx_ratio=0.0, vy_ratio=-0.125,
        )
        self.crop_info = crop
        img_crop = crop["img_crop"]
        img_crop_256 = cv2.resize(img_crop, (256, 256), interpolation=cv2.INTER_AREA)

        # Landmarks *within* the crop, needed for the eye-close ratio.
        self.lmk_crop = self.landmark_runner.run(img_crop, lmk)

        I_s = self.wrapper.prepare_source(img_crop_256)
        self.f_s = self.wrapper.extract_feature_3d(I_s)
        self.x_s_info = self.wrapper.get_kp_info(I_s)
        self.x_c_s = self.x_s_info["kp"]
        self.x_s = self.wrapper.transform_keypoint(self.x_s_info)

        # Source neutral pose, in degrees. Behaviour deltas are applied
        # relative to this, so the avatar starts from the portrait's own
        # natural head angle rather than snapping to frontal.
        self.src_pitch = float(self.x_s_info["pitch"].item())
        self.src_yaw = float(self.x_s_info["yaw"].item())
        self.src_roll = float(self.x_s_info["roll"].item())
        self.src_scale = float(self.x_s_info["scale"].item())
        self.src_t = self.x_s_info["t"].clone()
        self.src_exp = self.x_s_info["exp"].clone()

        # Baseline eye openness of the source portrait. A blink drives toward
        # zero from here, so a subject photographed with narrow eyes does not
        # suddenly widen them.
        from src.utils.retargeting_utils import calc_eye_close_ratio

        self.src_eye_ratio = float(
            calc_eye_close_ratio(self.lmk_crop[None])[0].mean()
        )

        if self.paste_back:
            from src.utils.crop import prepare_paste_back

            self.mask_ori = prepare_paste_back(
                self.cfg.mask_crop, crop["M_c2o"],
                dsize=(img_bgr.shape[1], img_bgr.shape[0]),
            )

    # -- per-frame ----------------------------------------------------------
    def _build_driving_keypoints(self, pose: AvatarPose) -> torch.Tensor:
        """Synthesise x_d from the behaviour pose. No driving video involved."""
        from .calibration import apply_expression_deltas

        info = {
            "pitch": torch.tensor(
                [[self.src_pitch + pose.pitch]], device=self.device, dtype=torch.float32
            ),
            "yaw": torch.tensor(
                [[self.src_yaw + pose.yaw]], device=self.device, dtype=torch.float32
            ),
            "roll": torch.tensor(
                [[self.src_roll + pose.roll]], device=self.device, dtype=torch.float32
            ),
            # Breathing rides on scale and vertical translation.
            "scale": torch.tensor(
                [[self.src_scale * pose.scale]], device=self.device, dtype=torch.float32
            ),
            "t": self.src_t.clone(),
            "exp": apply_expression_deltas(self.src_exp.clone(), pose),
            "kp": self.x_c_s,
        }
        info["t"][:, 0] += pose.tx
        info["t"][:, 1] += -pose.ty

        return self.wrapper.transform_keypoint(info)

    def _apply_blink(self, x_d: torch.Tensor, pose: AvatarPose) -> torch.Tensor:
        """Close the eyelids using LivePortrait's dedicated retargeting network.

        This is the correct mechanism rather than nudging keypoints that happen
        to sit near the eye: the network was trained to produce an anatomically
        coherent lid closure, including the surrounding skin deformation that a
        raw keypoint offset would miss.
        """
        openness = 0.5 * (pose.eye_open_l + pose.eye_open_r)
        if openness >= 0.995:
            return x_d
        target = self.src_eye_ratio * openness
        combined = self.wrapper.calc_combined_eye_ratio([[target]], self.lmk_crop)
        delta = self.wrapper.retarget_eye(self.x_s, combined)
        return x_d + delta.reshape(-1, x_d.shape[1], 3)

    def render(self, pose: AvatarPose) -> np.ndarray:
        with torch.no_grad():
            x_d = self._build_driving_keypoints(pose)
            x_d = self._apply_blink(x_d, pose)
            if self.cfg.flag_stitching:
                x_d = self.wrapper.stitching(self.x_s, x_d)
            out = self.wrapper.warp_decode(self.f_s, self.x_s, x_d)
            frame = self.wrapper.parse_output(out["out"])[0]  # RGB uint8

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.paste_back:
            from src.utils.crop import paste_back

            frame_bgr = paste_back(
                frame_bgr, self.crop_info["M_c2o"], self.source_full, self.mask_ori
            )

        return self._fit_output(frame_bgr)

    def _fit_output(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox into the target 16:9 output without distorting the face."""
        target_w, target_h = self.output_size
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    @property
    def info(self) -> RendererInfo:
        return self._info

    def close(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
