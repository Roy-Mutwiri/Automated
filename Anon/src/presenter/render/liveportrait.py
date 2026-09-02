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

## Detection: no third-party detector at all

LivePortrait's stock cropper uses InsightFace, whose **models are licensed for
non-commercial research only** - the one real licensing hazard in the stack.
This backend does not use it, and does not use MediaPipe either: LivePortrait's
own `landmark.onnx` bootstraps itself in two passes (see `_detect_landmarks`).
Detection runs once at startup, so the two passes cost nothing measurable, and
the runtime ends up with **no non-commercial component anywhere in it**.

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


# Everything that belongs to *one source portrait* rather than to the renderer.
#
# Swapping outfits means swapping all of it at once and nothing else: the
# LivePortrait weights, the landmark runner and the room style are properties of
# the renderer and must survive the switch, while identity features, framing,
# masks and the composited background must not.
#
# Kept as an explicit list rather than inferred, because a silently missing
# entry produces a switch that mostly works - the new face over the old
# silhouette's mask, say - which is far harder to diagnose than a crash.
# `test_source_state_is_complete` walks the AST of the three methods that
# populate these and fails if any assignment here is unaccounted for.
_SOURCE_STATE = (
    # identity and motion baseline
    "source_full", "crop_info", "lmk_crop", "f_s", "x_s_info", "x_c_s", "x_s",
    "src_pitch", "src_yaw", "src_roll", "src_scale", "src_t", "src_exp",
    "src_eye_ratio",
    # framing and the composited background
    "frame_rect", "content_rect", "M_crop2out", "background",
    "person_matte", "subject_alpha_out", "face_luma", "background_stops",
    "desk_band", "desk_rgb_premul", "desk_inv_alpha",
    "wrap_layer", "wrap_idx", "wrap_add",
    # per-frame blend zones
    "mask_out", "mask_box", "blend_box", "opaque_mask", "feather_mask",
    "feather_alpha", "feather_bg_premul", "_opaque_frac",
)


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
        framing: str = "shoulders",
        environment: str = "streaming_room",
        room_style=None,
        room_seed: int = 7,
        neutralize_pose: float = 0.85,
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
        if framing not in ("shoulders", "close", "full"):
            raise ValueError(
                f"framing must be 'shoulders', 'close' or 'full', "
                f"got {framing!r}"
            )
        self.framing = framing
        if environment not in ("streaming_room", "source"):
            raise ValueError(
                f"environment must be 'streaming_room' or 'source', "
                f"got {environment!r}"
            )
        self.environment = environment
        self.room_style = room_style
        self.room_seed = room_seed
        self.neutralize_pose = float(min(max(neutralize_pose, 0.0), 1.0))
        self.person_matte = None
        self.subject_alpha_out = None
        self.desk_band = None
        self.wrap_layer = None
        self.wrap_idx = None
        # Measured at startup: how far the plate sits under the face, in stops.
        self.face_luma = None
        self.background_stops = None

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
            onnx_provider="cpu",  # runs once at startup; CUDA EP needs CUDA 13 DLLs
            device_id=0,
        )
        self.landmark_runner.warmup()

        # Prepared sources, keyed by path. Preparing one costs a DeepLabV3
        # forward pass, an appearance-feature extraction and a full recompose,
        # so an outfit the presenter has already worn should come back
        # instantly rather than being rebuilt.
        self._source_cache: dict[str, dict] = {}
        self.source_path = Path(source_image).resolve()

        self._prepare_source(Path(source_image))
        self._source_cache[str(self.source_path)] = self._capture_source_state()

        self._info = RendererInfo(
            name="liveportrait",
            resolution=output_size,
            device=f"{device}{' fp16' if use_half else ' fp32'}",
            photoreal=True,
            notes=self._describe_source(),
        )

    # -- wardrobe -----------------------------------------------------------
    def _describe_source(self) -> str:
        notes = f"source={self.source_path.name}"
        if self.background_stops is not None:
            # Worth surfacing rather than burying: it is the one property of
            # the background that is measured, and the number that says whether
            # the subject actually separates from the room.
            notes += f" bg=-{self.background_stops:.2f} stops"
        return notes

    def _capture_source_state(self) -> dict:
        return {k: getattr(self, k, None) for k in _SOURCE_STATE}

    def set_source(self, source_image: str | Path, cache_limit: int = 8) -> bool:
        """Switch the presenter to a different source portrait.

        This is how the wardrobe works. An outfit cannot be composited on at
        runtime - the torso is static source pixels and the head is warped from
        source appearance features - so changing clothes means changing the
        image the whole pipeline was prepared from. See `render/wardrobe.py`.

        Returns True if the source had to be prepared, False if it came back
        from cache. A fresh preparation runs a DeepLabV3 matte and an appearance
        extraction and takes a noticeable moment; a cached one is a dict copy,
        which is why switching back and forth is free after the first visit.

        `cache_limit` bounds the memory: each entry holds a composited
        background, a matte and a light-wrap layer, on the order of tens of
        megabytes, so an unbounded cache would grow without limit across a long
        session of outfit changes. Eviction is oldest-first and the *current*
        source is never evicted.
        """
        path = Path(source_image).resolve()
        if path == self.source_path:
            return False
        if not path.exists():
            raise FileNotFoundError(f"no portrait for this outfit: {path}")

        key = str(path)
        cached = self._source_cache.get(key)
        if cached is not None:
            for name, value in cached.items():
                setattr(self, name, value)
        else:
            self._prepare_source(path)
            self._source_cache[key] = self._capture_source_state()

        self.source_path = path
        self._info.notes = self._describe_source()

        while len(self._source_cache) > max(cache_limit, 1):
            for stale in list(self._source_cache):
                if stale != key:
                    del self._source_cache[stale]
                    break
            else:
                break
        return cached is None

    # -- one-time identity preparation --------------------------------------
    def _detect_landmarks(self, img_rgb: np.ndarray) -> np.ndarray:
        """Locate the face using only LivePortrait's own landmark model.

        Neither InsightFace nor MediaPipe is used. `LandmarkRunner.run` called
        without a seed force-resizes the whole image to 224x224 - which
        upstream marks "NOT RECOMMEND" for arbitrary photos, and rightly, since
        a small face in a wide shot would be destroyed by that resize. For a
        framed head-and-shoulders portrait the face already fills most of the
        frame, so the first pass lands close enough to define a crop, and a
        second pass on that crop refines to full accuracy.

        This removes the last third-party detector from the pipeline. The
        licensing consequence is the point: InsightFace's models are
        non-commercial-only, and with this two-pass approach nothing in the
        runtime depends on them.
        """
        h, w = img_rgb.shape[:2]

        # Master-frame sources break the two-pass bootstrap. In a room-scale
        # 1344x768 photograph the face is ~150 px, and the coarse pass resizes
        # the *whole frame* to 224x224 - shrinking that face to roughly 25 px,
        # which is precisely the situation upstream marks as unreliable.
        #
        # So find the person first, and give the coarse pass a crop in which
        # the head actually fills a useful fraction of the image. Person
        # segmentation is already needed elsewhere, costs one startup call, and
        # is far more robust at this scale than face landmarks are.
        search_rgb, offset, scale = img_rgb, (0.0, 0.0), 1.0
        if self.framing == "full":
            search_rgb, offset, scale = self._head_region(img_rgb)

        coarse = self.landmark_runner.run(search_rgb)
        refined = self.landmark_runner.run(search_rgb, coarse)

        # Map back into full-frame coordinates.
        refined = refined.astype(np.float32) / scale
        refined[:, 0] += offset[0]
        refined[:, 1] += offset[1]

        xs, ys = refined[:, 0], refined[:, 1]
        if not (0 <= xs.mean() <= w and 0 <= ys.mean() <= h):
            raise RuntimeError(
                "Landmark detection did not converge on a face. The source "
                "image must contain one clearly visible, roughly front-facing "
                "face."
            )
        span = max(xs.max() - xs.min(), ys.max() - ys.min())
        # The floor is relative to the *head region* for master frames, since a
        # face is legitimately a small fraction of a room-scale photograph.
        floor = 60.0 if self.framing == "full" else 0.12 * min(h, w)
        if span < floor:
            raise RuntimeError(
                f"Detected face spans only {span:.0f}px in a {w}x{h} image, "
                f"below the {floor:.0f}px floor. There is not enough face to "
                "animate; use a source framed closer to the subject."
            )
        return refined

    def _head_region(self, img_rgb: np.ndarray):
        """Crop a head-sized search window around the person's head.

        Returns ``(rgb_crop, (offset_x, offset_y), scale)`` so landmarks found
        in the crop can be mapped back to full-frame coordinates.

        Uses person segmentation to locate the subject, then takes the upper
        portion of that silhouette - where a seated person's head is - and
        upsamples it so the landmark model sees a face at a scale it was
        trained for.
        """
        h, w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        matte = self._person_matte(img_bgr)

        ys, xs = np.where(matte > 0.5)
        if len(xs) < 500:
            # Segmentation found nothing usable; fall back to the whole frame
            # and let the span check decide.
            return img_rgb, (0.0, 0.0), 1.0

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        person_h = y1 - y0
        person_w = x1 - x0

        # Head occupies roughly the top quarter of a seated silhouette. Take a
        # generous square around it rather than a tight box - the landmark
        # model wants context.
        head_cy = y0 + person_h * 0.16
        head_cx = x0 + person_w * 0.5
        half = max(person_w * 0.42, person_h * 0.24, 96.0)

        cx0 = int(np.clip(head_cx - half, 0, w - 1))
        cx1 = int(np.clip(head_cx + half, cx0 + 32, w))
        cy0 = int(np.clip(head_cy - half, 0, h - 1))
        cy1 = int(np.clip(head_cy + half, cy0 + 32, h))

        crop = img_rgb[cy0:cy1, cx0:cx1]
        # Upsample small crops so the face reaches a size the model handles.
        scale = 1.0
        if max(crop.shape[:2]) < 512:
            scale = 512.0 / max(crop.shape[:2])
            crop = cv2.resize(
                crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        self._cached_matte = matte
        return crop, (float(cx0), float(cy0)), scale

    def _prepare_source(self, path: Path) -> None:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"could not read source image {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.source_full = img_bgr

        lmk = self._detect_landmarks(img_rgb)

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

        self._prepare_compositing(img_bgr, lmk, crop)

    def _segmenter(self):
        """DeepLabV3, kept resident once loaded.

        This used to be constructed and thrown away inside every matte, which
        was defensible while a matte happened once per process. It does not
        happen once any more: preparing a source calls it twice - once to find
        the head, once to composite - and the wardrobe prepares a source every
        time the presenter changes into something new. Building a ResNet-101
        and loading its weights four times to change a shirt is most of what
        made the first switch take six seconds.

        The cost of keeping it is ~230 MB of VRAM against 14.7 GB free, and
        `close()` releases it.
        """
        if getattr(self, "_seg", None) is None:
            import torchvision

            weights = (torchvision.models.segmentation
                       .DeepLabV3_ResNet101_Weights.DEFAULT)
            self._seg = (
                torchvision.models.segmentation
                .deeplabv3_resnet101(weights=weights)
                .eval().to(self.device),
                weights,
            )
        return self._seg

    def _person_matte(self, img_bgr: np.ndarray) -> np.ndarray:
        """Segment the presenter from the source portrait.

        Uses torchvision's DeepLabV3 (BSD-licensed weights, already available
        with torch) rather than adding a matting dependency. Its output is
        coarse around hair, which would normally be disqualifying for a
        portrait with this much of it - but two things make it sufficient here:

        * The replacement background is heavily defocused, so there is no sharp
          detail behind the edge for a ragged matte to contrast against.
        * The matte is dilated and feathered before use, so the transition band
          carries a few pixels of the *original* background - itself soft, warm
          bokeh - which blends into the generated room rather than cutting
          against it.

        The dilation has a second job: the head moves a few pixels as it turns,
        and a matte computed from a static frame would otherwise clip the
        silhouette on the leading edge.
        """
        import torch

        model, weights = self._segmenter()

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        batch = weights.transforms()(
            torch.from_numpy(rgb).permute(2, 0, 1)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(batch)["out"][0]
        # Class 15 is "person" in the VOC label set these weights use.
        probs = logits.softmax(0)[15].cpu().numpy()

        h, w = img_bgr.shape[:2]
        probs = cv2.resize(probs.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)

        # Refine the semantic mask against the image's own colour statistics.
        # The raw network output is a smooth blob - correct about *where* the
        # person is, wrong about exactly where they end, and it drifts several
        # dozen pixels into the background around hair. Composited directly it
        # leaves a halo of the original backdrop, which against a replaced
        # background reads as a cut-out.
        #
        # GrabCut, seeded from the confident interior and exterior, snaps the
        # boundary onto the real edge. It will not resolve individual strands
        # of hair - no colour-model method does - but it recovers the hair
        # *mass*, and against a defocused background that is the difference
        # that matters.
        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc[probs > 0.55] = cv2.GC_PR_FGD
        kernel = np.ones((25, 25), np.uint8)
        gc[cv2.erode((probs > 0.90).astype(np.uint8), kernel) > 0] = cv2.GC_FGD
        gc[cv2.dilate((probs < 0.10).astype(np.uint8), kernel) > 0] = cv2.GC_BGD
        try:
            cv2.grabCut(img_bgr, gc, None,
                        np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64),
                        4, cv2.GC_INIT_WITH_MASK)
            matte = np.where(
                (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1.0, 0.0
            ).astype(np.float32)
        except cv2.error:
            # Seeds can be degenerate on an unusual portrait; the unrefined
            # mask is worse but still usable, so do not fail the whole run.
            matte = (probs > 0.5).astype(np.float32)

        if matte.mean() < 0.03:
            raise RuntimeError(
                "person segmentation found almost nothing - the source "
                "portrait may not contain a recognisable person"
            )

        # Keep only the largest connected region; stray specks elsewhere in the
        # frame would punch holes in the new background.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            matte.astype(np.uint8), 8
        )
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            matte = (labels == largest).astype(np.float32)

        # Close notches enclosed by the subject before growing the matte.
        #
        # A seated pose leaves a gap between the arm and the torso. The
        # segmentation is right to call that background - but what is visible
        # *through* it is the presenter's own desk, a foot behind his arm, not
        # the far wall. Replacing the background there punches a hard dark
        # wedge through his chest, which is the most conspicuous thing in the
        # frame.
        #
        # Closing the notch keeps the source pixels in that gap, so it shows the
        # desk it actually showed. The kernel is deliberately large - the gap is
        # substantial - but closing cannot bridge genuinely separate regions, so
        # the silhouette outline is unaffected.
        close = max(int(0.075 * min(h, w)) | 1, 9)
        matte = cv2.morphologyEx(
            matte, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8)
        )

        grow = max(int(0.005 * min(h, w)), 3)
        matte = cv2.dilate(matte, np.ones((grow, grow), np.uint8))
        feather = max(int(0.008 * min(h, w)) | 1, 5)
        matte = cv2.GaussianBlur(matte, (feather, feather), 0)
        return np.clip(matte, 0.0, 1.0)

    def _build_fill(self, img_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """Fill the 16:9 canvas either side of a taller-than-16:9 framing.

        Keeping the shoulders means the framing is taller than the output
        aspect, which leaves margins. Black bars would waste a third of the
        frame and look like a mistake; the broadcast convention is a blurred,
        over-scaled copy of the same image behind the sharp one.

        It suits this pipeline particularly well: the fill is derived from the
        source's own out-of-focus background, so the colour and lighting match
        exactly, and being computed once it is perfectly static - it cannot
        introduce the background wobble the brief rules out.
        """
        h, w = img_bgr.shape[:2]
        cover = max(out_w / w, out_h / h) * 1.18   # over-scale, then crop
        fw, fh = int(w * cover), int(h * cover)
        filled = cv2.resize(img_bgr, (fw, fh), interpolation=cv2.INTER_LINEAR)
        x0 = max((fw - out_w) // 2, 0)
        y0 = max((fh - out_h) // 2, 0)
        filled = filled[y0:y0 + out_h, x0:x0 + out_w]
        if filled.shape[0] != out_h or filled.shape[1] != out_w:
            filled = cv2.resize(filled, (out_w, out_h))

        # Heavy blur so no edge in the fill competes with the face, then darken
        # slightly so the sharp content reads as the subject.
        k = max(31, (min(out_w, out_h) // 12) | 1)
        filled = cv2.GaussianBlur(filled, (k, k), 0)
        return cv2.convertScaleAbs(filled, alpha=0.72, beta=0)

    # -- output framing and compositing -------------------------------------
    def _prepare_compositing(self, img_bgr, lmk, crop) -> None:
        """Precompute a single affine from the 512 crop straight to the output.

        The naive path - paste the crop back into the full 1024x1024 source,
        then letterbox that into 1280x720 - measured **44 ms per frame**, more
        than a third of the total budget, and every millisecond of it is spent
        recomputing a composite whose background never changes.

        Two things are wrong with it. It warps and alpha-blends the full
        1024x1024 frame when only the face region changes, and it then throws
        most of those pixels away: a square source letterboxed into 16:9 leaves
        the picture as bars either side.

        Instead: choose a 16:9 framing rectangle in source coordinates with
        proper headroom and eyeline, compose the *static* background into the
        output canvas once, and per frame warp only the generated crop directly
        into output space with a precomputed mask. One warp, one blend, no
        wasted pixels - and it fixes the composition requirement at the same
        time.
        """
        from src.utils.crop import prepare_paste_back

        h, w = img_bgr.shape[:2]
        out_w, out_h = self.output_size

        # Landmark set is LivePortrait's 203-point format; the eye region is
        # its first ~48 points.
        eye_y = float(np.mean(lmk[:48, 1]))
        face_w = float(lmk[:, 0].max() - lmk[:, 0].min())
        face_h = float(lmk[:, 1].max() - lmk[:, 1].min())
        face_cx = float((lmk[:, 0].max() + lmk[:, 0].min()) * 0.5)
        head_top = float(lmk[:, 1].min())

        if self.framing == "full":
            # Master-frame mode. The source is already a composed photograph of
            # the presenter in his room, so there is nothing to reframe: any
            # anatomical crop would throw away the room that is the whole point
            # of the image.
            #
            # Take the largest centred rectangle matching the *output* aspect
            # rather than fitting the whole frame. SDXL's nearest 16:9 bucket is
            # 1344x768 (1.75), a hair squarer than 16:9, and fitting that would
            # pillarbox by a few pixels and expose the blurred fill down both
            # edges - a visible border on an image that should be full-bleed.
            # Trimming ~6 px off the top and bottom instead costs nothing.
            frame_w = min(float(w), float(h) * out_w / out_h)
            frame_h = frame_w * out_h / out_w
            if frame_h > h:
                frame_h = float(h)
                frame_w = frame_h * out_w / out_h
        elif self.framing == "close":
            frame_w = min(float(w), face_w * 3.4)
            frame_h = frame_w * out_h / out_w
        else:
            # Head and shoulders. Forcing a 16:9 rectangle out of a square
            # source is what produced the tight close-up: the aspect ratio can
            # only be satisfied by discarding vertical extent, and the
            # shoulders are the first thing to go.
            #
            # So choose the framing from anatomy instead of from the output
            # aspect, and let it be taller than 16:9. Headroom above the crown,
            # and far enough below the eyeline to include the shoulder line.
            top = head_top - face_h * 0.42
            bottom = eye_y + face_h * 2.9
            frame_h = min(float(h), bottom - top)
            frame_w = min(float(w), frame_h * out_w / out_h)

        if self.framing == "full":
            left = (w - frame_w) * 0.5
            top = (h - frame_h) * 0.5
        else:
            left = face_cx - frame_w * 0.5
            if self.framing == "close":
                top = eye_y - frame_h * 0.38  # eyes above centre -> headroom
            else:
                top = head_top - face_h * 0.42
        left = float(np.clip(left, 0, max(w - frame_w, 0)))
        top = float(np.clip(top, 0, max(h - frame_h, 0)))
        self.frame_rect = (left, top, frame_w, frame_h)

        # Fit the framing inside the output, preserving aspect. When it is
        # taller than 16:9 the sides are filled rather than left black - see
        # _build_fill.
        s = min(out_w / frame_w, out_h / frame_h)
        content_w, content_h = frame_w * s, frame_h * s
        off_x = (out_w - content_w) * 0.5
        off_y = (out_h - content_h) * 0.5
        self.content_rect = (off_x, off_y, content_w, content_h)

        # Source pixels -> output canvas.
        M_src2out = np.array(
            [[s, 0.0, off_x - s * left], [0.0, s, off_y - s * top]], np.float32
        )
        # Crop pixels -> source pixels (LivePortrait gives a 3x3).
        M_c2o = crop["M_c2o"][:2, :] if crop["M_c2o"].shape[0] == 3 else crop["M_c2o"]
        M_c2o_h = np.vstack([M_c2o, [0.0, 0.0, 1.0]]).astype(np.float32)
        M_src2out_h = np.vstack([M_src2out, [0.0, 0.0, 1.0]]).astype(np.float32)
        self.M_crop2out = (M_src2out_h @ M_c2o_h)[:2, :].astype(np.float32)

        # Static background, composed once.
        if self.environment == "streaming_room":
            from .environment import (
                RoomStyle,
                fit_exposure,
                key_luminance,
                light_wrap,
                render_desk_foreground,
                render_mic_foreground,
                render_streaming_room,
            )

            style = self.room_style or RoomStyle()

            # The generated room replaces the portrait's own background
            # everywhere, so the fill and the area behind the subject are the
            # same image - no seam between them by construction.
            fill = render_streaming_room(
                out_w, out_h, style=self.room_style, seed=self.room_seed
            )

            # Fit the plate to the face rather than to taste.
            #
            # Everything else about this background is a judgement call; this
            # one is a number. The key-lit skin is measured inside the landmark
            # bounding box (see key_luminance for why that is a percentile and
            # not a median) and the room is scaled to sit the conventional 1-2
            # stops beneath it. Tuning the wall colours by eye instead would
            # bake this source portrait's exposure into the constants and
            # silently mis-expose the room for any other one.
            fx0, fx1 = int(lmk[:, 0].min()), int(lmk[:, 0].max()) + 1
            fy0, fy1 = int(lmk[:, 1].min()), int(lmk[:, 1].max()) + 1
            face = img_bgr[max(fy0, 0):min(fy1, h), max(fx0, 0):min(fx1, w)]
            self.face_luma = key_luminance(face) if face.size else 128.0
            fill, exposure_scale, self.background_stops = fit_exposure(
                fill, self.face_luma, style.exposure_ratio, style.exposure_limits
            )
            fill = fill.astype(np.float32)

            # The desk is lit by the same room, so it takes the same
            # correction. Exposing the planes independently is how a composite
            # ends up looking like three images that happen to be adjacent.
            desk_bgr, desk_alpha = render_desk_foreground(
                out_w, out_h, style=self.room_style
            )
            mic_bgr, mic_alpha = render_mic_foreground(
                out_w, out_h, style=self.room_style
            )
            desk_bgr = np.clip(
                desk_bgr.astype(np.float32) * exposure_scale, 0, 255
            )
            mic_bgr = np.clip(
                mic_bgr.astype(np.float32) * exposure_scale, 0, 255
            )

            # Flatten desk and mic into one foreground plane at startup. Both
            # are in front of the presenter and both are static, so blending
            # them separately every frame would cost twice for no difference.
            # The mic goes over the desk - it hangs in front of it.
            fg = desk_bgr * desk_alpha
            fg_alpha = desk_alpha.copy()
            fg = fg * (1.0 - mic_alpha) + mic_bgr * mic_alpha
            fg_alpha = fg_alpha * (1.0 - mic_alpha) + mic_alpha

            rows = np.where(fg_alpha[:, :, 0].max(axis=1) > 0.003)[0]
            if len(rows):
                y0d = int(rows.min())
                self.desk_band = (y0d, out_h)
                self.desk_rgb_premul = fg[y0d:]
                self.desk_inv_alpha = 1.0 - fg_alpha[y0d:]
            else:
                self.desk_band = None

            matte = self._person_matte(img_bgr)
            self.person_matte = matte
            subject = img_bgr.astype(np.float32) * matte[..., None]
            content = cv2.warpAffine(
                subject, M_src2out, (out_w, out_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            alpha_out = cv2.warpAffine(
                matte, M_src2out, (out_w, out_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )

            # Fade the subject out at the edge of the source frame.
            #
            # The portrait is a fixed crop, so the torso is simply cut off at
            # the image boundary. Against the original background that was
            # invisible - the cut coincided with the frame edge. Against a
            # replaced background it becomes a hard vertical line where the
            # body stops in mid-air, which is far more obvious than the missing
            # shoulders themselves.
            #
            # Feathering the alpha at that boundary turns the cut into a
            # falloff into shadow, which in a dimly lit room reads as the body
            # continuing out of the light. It is the difference between
            # "cropped" and "unlit", and it is what lets the desk sit low
            # enough to leave the shoulders visible.
            edge = np.ones((out_h, out_w), np.float32)
            ex0, ey0, ew, eh = self.content_rect
            band = max(int(min(ew, eh) * 0.13), 24)
            xs_e = np.arange(out_w, dtype=np.float32)
            ys_e = np.arange(out_h, dtype=np.float32)
            fx = np.clip((xs_e - ex0) / band, 0, 1) * np.clip(
                (ex0 + ew - 1 - xs_e) / band, 0, 1
            )
            fy = np.clip((ys_e - ey0) / band, 0, 1) * np.clip(
                (ey0 + eh - 1 - ys_e) / band, 0, 1
            )
            # Top edge is left alone: the crown of the head genuinely reaches
            # the frame top and fading it would erase the hair.
            fy = np.maximum(fy, np.clip((ey0 + eh - 1 - ys_e) / band, 0, 1))
            edge = np.minimum(fx[None, :], fy[:, None])
            alpha_out = (alpha_out * edge)[..., None]

            # Light wrap. The subject was not lit by this room, so without it
            # the silhouette is an algebraic cut: a hard boundary between two
            # images that a photograph would never produce, because in a real
            # room the background's light spills around the edge of the head.
            #
            # It is kept as its own layer and deliberately *not* folded in
            # here. The per-frame path rebuilds every pixel of the mask - an
            # integer copy over the opaque interior, an alpha blend over the
            # feather band - and both land squarely on the wrap band, which by
            # construction sits just inside the silhouette. Baking it in now
            # would mean the copy erases it and the blend delivers it at
            # (1 - alpha) strength, which is near zero precisely at the edge
            # it exists to soften. _finish_blend_setup adds it back afterwards.
            self.wrap_layer = light_wrap(
                fill, alpha_out[..., 0], style.wrap_width, style.wrap_strength
            )
            self.background = np.clip(
                content + fill * (1.0 - alpha_out), 0, 255
            ).astype(np.uint8)
            # Per-frame, the generated crop must be confined to the subject:
            # LivePortrait regenerates the whole crop including the portrait's
            # original background, and letting that through would paint the old
            # bokeh back over the room in a rectangle around the head.
            self.subject_alpha_out = alpha_out.astype(np.float32)
            self._finish_blend_setup(crop, M_src2out, w, h, out_w, out_h)
            return

        self.person_matte = None
        self.subject_alpha_out = None
        self.wrap_layer = None
        fill = self._build_fill(img_bgr, out_w, out_h).astype(np.float32)
        content = cv2.warpAffine(
            img_bgr, M_src2out, (out_w, out_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        ).astype(np.float32)

        # A hard edge between sharp content and blurred fill announces itself as
        # a composite. Feathering the join over a couple of dozen pixels reads
        # as depth of field instead.
        coverage = np.zeros((out_h, out_w), np.float32)
        cx0, cy0, cw, ch = self.content_rect
        cv2.rectangle(
            coverage,
            (int(round(cx0)), int(round(cy0))),
            (int(round(cx0 + cw)) - 1, int(round(cy0 + ch)) - 1),
            1.0, -1,
        )
        feather = max(9, (min(out_w, out_h) // 40) | 1)
        coverage = cv2.GaussianBlur(coverage, (feather, feather), 0)[..., None]
        self.background = np.clip(
            content * coverage + fill * (1.0 - coverage), 0, 255
        ).astype(np.uint8)

        self._finish_blend_setup(crop, M_src2out, w, h, out_w, out_h)

    def _finish_blend_setup(self, crop, M_src2out, w, h, out_w, out_h) -> None:
        """Precompute the per-frame blend mask and its zones."""
        from src.utils.crop import prepare_paste_back

        # Feather mask in output space, also once.
        mask = prepare_paste_back(
            self.cfg.mask_crop, crop["M_c2o"], dsize=(w, h)
        )
        mask_out = cv2.warpAffine(mask, M_src2out, (out_w, out_h))
        if mask_out.ndim == 2:
            mask_out = mask_out[..., None]
        self.mask_out = mask_out.astype(np.float32)
        if self.mask_out.max() > 1.5:
            self.mask_out /= 255.0

        # With a replaced background, restrict the generated crop to the
        # subject. Without this the crop's own (original) background is pasted
        # back as a rectangle of old bokeh sitting over the new room.
        if self.subject_alpha_out is not None:
            self.mask_out = self.mask_out * self.subject_alpha_out

        # Blend only inside the mask's bounding box - outside it the output is
        # exactly the precomputed background, so those pixels never need
        # touching again.
        ys, xs = np.where(self.mask_out[:, :, 0] > 0.003)
        if len(xs):
            self.blend_box = (
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            )
        else:
            self.blend_box = (0, 0, out_w, out_h)
        x0, y0, x1, y1 = self.blend_box
        self.mask_box = np.ascontiguousarray(self.mask_out[y0:y1, x0:x1])

        # Split the mask into three zones. The crop scale is wide, so the mask
        # covers most of the frame and a single float blend over all of it
        # wastes most of its work on pixels that are fully opaque or fully
        # transparent. Only the feather band actually needs per-pixel alpha:
        #
        #   opaque (>= 0.997)  -> straight copy, integer
        #   clear  (<= 0.003)  -> leave the precomputed background
        #   feather            -> float blend, a thin band
        alpha = self.mask_box[:, :, 0]
        self.opaque_mask = alpha >= 0.997
        self.feather_mask = (alpha > 0.003) & (alpha < 0.997)
        self.feather_alpha = self.mask_box[self.feather_mask]
        bg_box = self.background[y0:y1, x0:x1].astype(np.float32)
        self.feather_bg_premul = (
            bg_box[self.feather_mask] * (1.0 - self.feather_alpha)
        )
        self._opaque_frac = float(self.opaque_mask.mean())

        # The light wrap, added after everything above has snapshotted the
        # background without it.
        #
        # feather_bg_premul is taken from the wrap-free background on purpose:
        # the per-frame blend contributes it at (1 - alpha), which is close to
        # zero over most of the feather band, so a wrap folded in there would
        # be delivered at a fraction of its strength exactly where it is
        # supposed to work. Instead the band is re-added at full strength each
        # frame over both rebuilt zones, and baked into the static background
        # for the pixels the per-frame path never touches.
        self.wrap_idx = None
        wrap = self.wrap_layer
        if wrap is not None:
            wl = wrap[y0:y1, x0:x1]
            # Below ~0.75/255 the wrap cannot change the output byte, so those
            # pixels would be pure cost.
            band = (wl.max(axis=2) > 0.75) & (self.opaque_mask | self.feather_mask)
            if band.any():
                # Stored as integer indices, not as the boolean mask. The band
                # is a few thousand pixels scattered around a perimeter, and
                # indexing by mask makes numpy walk the entire box to find
                # them: measured 0.89 ms per frame against 0.11 ms for the
                # identical work addressed directly.
                self.wrap_idx = np.nonzero(band)
                self.wrap_add = wl[band]
            self.background = np.clip(
                self.background.astype(np.float32) + wrap, 0, 255
            ).astype(np.uint8)

    # -- per-frame ----------------------------------------------------------
    def _build_driving_keypoints(self, pose: AvatarPose) -> torch.Tensor:
        """Synthesise x_d from the behaviour pose. No driving video involved."""
        from .calibration import apply_expression_deltas

        # Pull the head toward frontal before adding behaviour deltas.
        #
        # A portrait is rarely shot dead-on: this source sits at yaw -10 deg,
        # pitch +8 deg. Applying behaviour on top of that leaves the presenter
        # permanently addressing a point off to one side, which undoes the eye
        # contact the whole design depends on.
        #
        # Not neutralised all the way by default. LivePortrait has to
        # hallucinate whatever the rotation reveals, and the further it is
        # driven from the source pose the more that shows - especially along a
        # jaw or a headdress edge. Around 0.7 removes most of the off-axis look
        # while staying inside the range the model reproduces cleanly.
        keep = 1.0 - self.neutralize_pose
        info = {
            "pitch": torch.tensor(
                [[self.src_pitch * keep + pose.pitch]],
                device=self.device, dtype=torch.float32,
            ),
            "yaw": torch.tensor(
                [[self.src_yaw * keep + pose.yaw]],
                device=self.device, dtype=torch.float32,
            ),
            "roll": torch.tensor(
                [[self.src_roll * keep + pose.roll]],
                device=self.device, dtype=torch.float32,
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
        return self._composite(frame_bgr)

    def _composite(self, generated: np.ndarray) -> np.ndarray:
        """Warp the generated crop into the output canvas and blend.

        One warpAffine straight into output space and one alpha blend confined
        to the mask's bounding box. Everything else was precomputed at startup.
        """
        out_w, out_h = self.output_size
        x0, y0, x1, y1 = self.blend_box

        warped = cv2.warpAffine(
            generated, self.M_crop2out, (out_w, out_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

        frame = self.background.copy()
        box = frame[y0:y1, x0:x1]
        warped_box = warped[y0:y1, x0:x1]

        # Opaque interior: integer copy, no float conversion.
        np.copyto(box, warped_box, where=self.opaque_mask[..., None])

        # Feather band only: per-pixel alpha over a thin strip.
        if self.feather_alpha.size:
            blended = (
                warped_box[self.feather_mask].astype(np.float32) * self.feather_alpha
                + self.feather_bg_premul
            )
            box[self.feather_mask] = blended.astype(np.uint8)

        # Light wrap over the silhouette edge the two blends above just rebuilt.
        if self.wrap_idx is not None:
            wr, wc = self.wrap_idx
            box[wr, wc] = np.clip(
                box[wr, wc].astype(np.float32) + self.wrap_add, 0, 255
            ).astype(np.uint8)

        # Foreground desk, composited last so it occludes the presenter. Only
        # the bottom band is touched, and the desk colour is pre-multiplied at
        # startup, so this is one multiply-add over ~17% of the frame.
        if getattr(self, "desk_band", None) is not None:
            dy0, dy1 = self.desk_band
            region = frame[dy0:dy1].astype(np.float32)
            frame[dy0:dy1] = np.clip(
                region * self.desk_inv_alpha + self.desk_rgb_premul, 0, 255
            ).astype(np.uint8)

        return frame

    @property
    def info(self) -> RendererInfo:
        return self._info

    def close(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
