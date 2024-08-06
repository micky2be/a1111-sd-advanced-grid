# Python
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from dataclasses import field as set_field
from typing import TYPE_CHECKING

# SD-WebUI
from modules import sd_models, sd_vae

# Local
from sd_advanced_grid.utils import clean_name, get_closest_from_list, logger, parse_range_float, parse_range_int

# ################################### Types ################################## #

if TYPE_CHECKING:
    from collections.abc import Callable

    from modules.processing import StableDiffusionProcessing as SD_Proc

# ################################# Constants ################################ #

SHARED_OPTS = [
    "CLIP_stop_at_last_layers",
    "code_former_weight",
    "face_restoration_model",
    "eta_noise_seed_delta",
    "sd_vae",
    "sd_model_checkpoint",
    "uni_pc_order",
    "use_scale_latent_for_hires_fix",
]

# ######################### Axis Modifier Interpreter ######################## #


@dataclass
class AxisOption:
    label: str
    type: type[str | int | float | bool]
    _: KW_ONLY
    field: str | None = None
    min: float = 0.0
    max: float = 1.0
    choices: Callable[..., list[str]] | None = None
    toggles: str | None = None
    cost: float = 0.2
    _valid: list[bool] = set_field(init=False, default_factory=list)
    _values: list[str] | list[int] | list[float] | list[bool] = set_field(init=False, default_factory=list)
    _index: int = set_field(init=False, default=0)

    @staticmethod
    def apply_to(field: str, value: AxisOption.type, proc: SD_Proc):
        if field in SHARED_OPTS:
            proc.override_settings[field] = value
        else:
            setattr(proc, field, value)

    def _apply(self, proc: SD_Proc):
        value = self._values[self._index]
        if self.type is None:
            return
        if self.toggles is None or value != "Default":
            AxisOption.apply_to(self.id, value, proc)

        if self.toggles:
            if self.choices:
                AxisOption.apply_to(self.toggles, value != "None", proc)
            else:
                AxisOption.apply_to(self.toggles, True, proc)

    def apply(self, proc: SD_Proc):
        """tranform the value on the Processing job with the current selected value"""
        if self._valid[self._index] is False:
            raise RuntimeError(f"Value not valid for {self.label}: {self.value}")
        try:
            self._apply(proc)
        except Exception as exc:
            raise RuntimeError(f"{self.value} could not be applied on {self.label}") from exc

    def next(self):
        if self._index + 1 < self.length:
            self._index += 1
            return True
        self._index = 0
        return False

    @property
    def id(self):  # pylint: disable=invalid-name
        return self.field if self.field is not None else clean_name(self.label)

    @property
    def length(self):
        return len(self._values)

    @property
    def values(self):
        """list of possible value"""
        return self._values.copy()

    @property
    def value(self):
        """value to be applied"""
        return self._values[self._index]

    @property
    def is_valid(self):
        if not self._valid:
            return None
        return all(self._valid)

    @property
    def index(self):
        return self._index

    def dict(self):
        return {"label": self.label, "param": self.id, "values": self.values}

    def set(self, values: str = "") -> AxisOption:
        """format input from a string to a list of value"""
        has_double_pipe = "||" in values
        value_list = [val.strip() for val in values.split("||" if has_double_pipe else ",") if val.strip()]
        if self.type == int:
            self._values = parse_range_int(value_list)
        elif self.type == float:
            self._values = parse_range_float(value_list)
        else:
            self._values = [value for value in map(self._format_value, value_list) if value not in {"", None}]  # type: ignore
        return self

    def unset(self):
        self._index = 0
        self._values = list()
        self._valid = list()

    def _format_value(self, value: str) -> AxisOption.type:
        cast_value = None
        if self.type == int:
            cast_value = int(value)

        elif self.type == float:
            cast_value = round(float(value), 8)

        elif self.type == bool:
            cast_value = str(value).lower()
            if cast_value in {"true", "yes", "1", "on"}:
                cast_value = True
            elif cast_value in {"false", "no", "0", "off"}:
                cast_value = False

        elif self.type == str and self.choices is not None:
            valid_list = self.choices()
            cast_value = get_closest_from_list(value, valid_list)
        else:
            cast_value = value

        return cast_value

    def validate(self, value: AxisOption.type) -> None:
        """raise an error if the data type is incorrect"""
        same_type = isinstance(value, self.type)
        if self.type in (int, float):
            if not same_type:
                raise RuntimeError(f"Must be a {self.type} number")
            if self.min is not None and value < self.min:  # type: ignore
                raise RuntimeError(f"Must be at least {self.min}")
            if self.max is not None and value > self.max:  # type: ignore
                raise RuntimeError(f"Must not exceed {self.max}")

        if self.type == bool and not same_type:
            raise RuntimeError("Must be either 'True' or 'False'")

        if self.type == str and self.choices is not None and (not same_type or not value):
            raise RuntimeError("Not found in the list")

        if not same_type:
            raise RuntimeError("Must be a valid type")

    def validate_all(self, quiet: bool = True, **_):
        def validation(value):
            try:
                self.validate(value)
            except RuntimeError as err:
                return f"'{err} for: {value}'"
            return None

        result = [validation(value) for value in self._values]

        if any(result):
            errors = [err for err in result if err]
            if not quiet:
                raise RuntimeError(f"Invalid parameters in {self.label}: {errors}")
            logger.warn(f"Invalid parameters in {self.label}", errors)
        self._valid = [err is None for err in result]


@dataclass
class AxisNothing(AxisOption):
    type: None = None

    def _apply(self, _):
        return

    @property
    def is_valid(self):
        return True


@dataclass
class AxisModel(AxisOption):
    _: KW_ONLY
    cost: float = 1.0  # change of checkpoints is too heavy, do it less often

    def validate(self, value: str):
        info = sd_models.get_closet_checkpoint_match(value)
        if info is None:
            raise RuntimeError("Unknown checkpoint")


@dataclass
class AxisVae(AxisOption):
    _: KW_ONLY
    cost: float = 0.7

    def validate(self, value: str):
        if value in {"None", "Automatic"}:
            return
        if sd_vae.vae_dict.get(value, None) is None:
            raise RuntimeError("Unknown VAE")


@dataclass
class AxisReplace(AxisOption):
    _: KW_ONLY
    cost: float = 0.5  # to allow prompt to be replaced before string manipulation
    _values: list[str] = set_field(init=False, default_factory=list)
    __tag: str = set_field(init=False, default="")

    def _apply(self, proc):
        """tranform the value on the Processing job"""
        value = str(self._values[self._index])
        proc.prompt = proc.prompt.replace(self.__tag, value)
        proc.negative_prompt = proc.negative_prompt.replace(self.__tag, value)

    def validate_all(self, quiet: bool = True, **kwargs):
        proc = kwargs.pop("proc", None)
        if proc is None:
            return
        error = ""
        if not self.__tag:
            error = "Values not set or invalid format"

        elif self.__tag not in proc.prompt and self.__tag not in proc.negative_prompt:
            error = f"Tag '{self.__tag}' not found in all prompts"

        if error:
            if quiet:
                logger.warn(error)
            else:
                raise RuntimeError(error)
        else:
            self._valid = [True] * self.length

    def set(self, values: str = "") -> AxisOption:
        """
        Promt_replace can handle different format sunch as:
        - 'one, two, three' => ['one=one', 'one=two', 'one=three']
        - 'TAG=one, two, three' => ['TAG=one', 'TAG=two', 'TAG=three']
        - 'TAG=one, TAG=two, TAG=three' => ['TAG=one', 'TAG=two', 'TAG=three']
        - 'TAG-one || TAG=two, three || TAG=four' => ['TAG=one', 'TAG=two, three', 'TAG=four']
        """
        has_double_pipe = "||" in values
        value_list = [val.strip() for val in values.split("||" if has_double_pipe else ",")]
        for value_pair in value_list:
            value = [string.strip() for string in value_pair.split("=", maxsplit=1)]
            if len(value) == 1 and value[0]:
                tag = self.__tag or value[0]
                self._values.append(value[0])
            elif value[0]:
                tag = self.__tag or value[0]
                self._values.append(value[1])
            else:
                continue
            self.__tag = tag
        self.label = self.label.replace("TAG", self.__tag)
        return self

    def unset(self):
        self._index = 0
        self._values = list()
        self._valid = list()
        self.__tag = ""
        self.label = "TAG"
