"""
E7 Context Engine Orchestrator.

Pipeline:
RGBA Object
    ↓
Scale
    ↓
Lighting
    ↓
Color Temperature
    ↓
Reflection
    ↓
Occlusion
    ↓
Compositor
    ↓
ContextResult
        ├── image
        ├── placement
        └── object_mask
"""

from dataclasses import dataclass

import numpy as np
import cv2

from .color_temperature import ColorTemperatureEngine
from .compositor import Compositor
from .lighting import LightingEngine
from .occlusion import OcclusionEngine
from .reflection import ReflectionEngine
from .scale import Placement, RandomScaleStrategy


@dataclass
class ContextResult:
    """
    Result produced by the E7 Context Engine.

    Attributes:
        image:
            Final composited BGR image.

        placement:
            Position and size where the transformed object
            was placed on the background.

        object_mask:
            Final binary mask of the transformed object
            before compositing.

            Shape:
                (placement.height, placement.width)

            Values:
                0   -> transparent/background
                255 -> visible object
    """

    image: np.ndarray
    placement: Placement
    object_mask: np.ndarray


class ContextEngine:
    """
    Orchestrates the E7 Context Engine pipeline.

    The Context Engine transforms an RGBA object according to
    the target background and composites it onto that background.

    Transformations:
        1. Scale
        2. Lighting
        3. Color Temperature
        4. Optional Reflection
        5. Optional Occlusion
        6. Composition
    """

    def __init__(
        self,
        scale_strategy: RandomScaleStrategy,
        lighting: LightingEngine,
        color_temperature: ColorTemperatureEngine,
        reflection: ReflectionEngine,
        occlusion: OcclusionEngine,
        compositor: Compositor,
        seed: int | None = None,
    ):
        self.scale_strategy = scale_strategy
        self.lighting = lighting
        self.color_temperature = color_temperature
        self.reflection = reflection
        self.occlusion = occlusion
        self.compositor = compositor

        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        background: np.ndarray,
        object_rgba: np.ndarray,
        enable_reflection: bool = False,
        enable_occlusion: bool = False,
    ) -> ContextResult:
        """
        Generate one contextualized object on a background.

        Args:
            background:
                BGR background image with shape (H, W, 3).

            object_rgba:
                RGBA object image with shape (H, W, 4).

            enable_reflection:
                Whether reflection transformation is applied.

            enable_occlusion:
                Whether artificial occlusion is applied.

        Returns:
            ContextResult containing:
                - final composited image
                - object placement
                - final transformed object mask
        """

        if background.ndim != 3 or background.shape[2] != 3:
            raise ValueError(
                "Background must be a BGR image with shape (H, W, 3)."
            )

        if object_rgba.ndim != 3 or object_rgba.shape[2] != 4:
            raise ValueError(
                "Object must be an RGBA image with shape (H, W, 4)."
            )

        bg = background.copy()
        obj = object_rgba.copy()

        # ---------------------------------------------------------
        # 1. Scale and placement
        # ---------------------------------------------------------

        placement = self.scale_strategy.calculate(
            obj_w=obj.shape[1],
            obj_h=obj.shape[0],
            img_w=bg.shape[1],
            img_h=bg.shape[0],
            rng=self.rng,
        )

        # ---------------------------------------------------------
        # 2. Estimate scene brightness
        # ---------------------------------------------------------

        brightness = self.lighting.estimate_scene_brightness(bg)

        # ---------------------------------------------------------
        # 3. Lighting
        # ---------------------------------------------------------

        obj = self.lighting.apply(
            rgba=obj,
            brightness_factor=brightness,
        )

        # ---------------------------------------------------------
        # 4. Color temperature
        # ---------------------------------------------------------

        temperature = int(
            self.rng.integers(-20, 21)
        )

        obj = self.color_temperature.apply(
            rgba=obj,
            temperature=temperature,
        )

        # ---------------------------------------------------------
        # 5. Optional reflection
        # ---------------------------------------------------------

        if enable_reflection:
            obj = self.reflection.apply(
                rgba=obj,
                strength=0.25,
            )

        # ---------------------------------------------------------
        # 6. Optional occlusion
        # ---------------------------------------------------------

        if enable_occlusion:
            ratio = float(
                self.rng.uniform(0.10, 0.35)
            )

            obj = self.occlusion.apply(
                rgba=obj,
                ratio=ratio,
            )

        # ---------------------------------------------------------
        # 7. Resize object to calculated placement
        #
        # We need the transformed mask at exactly the same
        # resolution that will be composited.
        # ---------------------------------------------------------

        

        object_resized = cv2.resize(
            obj,
            (placement.width, placement.height),
            interpolation=cv2.INTER_AREA,
        )

        # ---------------------------------------------------------
        # 8. Extract final object mask from alpha channel
        # ---------------------------------------------------------

        alpha = object_resized[:, :, 3]

        object_mask = np.where(
            alpha > 0,
            255,
            0,
        ).astype(np.uint8)

        # ---------------------------------------------------------
        # 9. Composite onto background
        # ---------------------------------------------------------

        result_image = self.compositor.composite(
            background=bg,
            object_rgba=obj,
            placement=placement,
        )

        return ContextResult(
            image=result_image,
            placement=placement,
            object_mask=object_mask,
        )