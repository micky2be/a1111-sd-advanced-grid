from __future__ import annotations
from typing import Type, Callable, Optional

from dataclasses import dataclass, field as set_field, KW_ONLY

from modules import shared, sd_models, sd_vae
from modules.processing import StableDiffusionProcessing as SDP

from sd_advanced_grid.utils import get_closest_from_list, clean_name
from sd_advanced_grid.utils import parse_range_int, parse_range_float
from sd_advanced_grid.utils import TRUTHY, FALSY


######################### Constants #########################
SHARED_OPTS = [
    "CLIP_stop_at_last_layers",
    "code_former_weight",
    "face_restoration_model",
    "eta_noise_seed_delta",
    "sd_vae",
    "sd_model_checkpoint",
    "uni_pc_order",
    "use_scale_latent_for_hires_fix"
]

######################### Axis #########################

@dataclass
class AxisOption:
    label: str
    type: Type[str | int | float | bool]
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
    def apply_to(field:str, value:AxisOption.type, p:SDP):
        if field in SHARED_OPTS:
            p.override_settings[field] = value
        else:
            setattr(p, field, value)

    def _apply(self, p:SDP):
        value = self._values[self._index]
        if self.type == None: return
        if self.toggles is None or value != "Default":
            AxisOption.apply_to(self.id, value, p)

        if self.toggles:
            if self.choices:
                AxisOption.apply_to(self.toggles, value != "None", p)
            else:
                AxisOption.apply_to(self.toggles, True, p)

    def apply(self, p:SDP):
        """tranform the value on the Processing job with the current selected value"""
        if self._valid[self._index] == False:
            raise RuntimeError(f"Unexpected error: Values not valid for {self.label}")
        try:
            self._apply(p)
        except:
            raise RuntimeError(f"Unexpected error: {self.value} could not be applied on {self.label}")

    def next(self):
        if self._index + 1 < self.length:
            self._index += 1
            return True
        self._index = 0
        return False

    @property
    def id(self):
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
        return all(self._valid)

    def dict(self):
        return {
            "label": self.label,
            "param": self.id,
            "values": self.values
        }

    def set(self, values:str = "") -> AxisOption:
        """format input from a string to a list of value"""
        has_double_pipe = "||" in values
        value_list = [val.strip() for val in values.split("||" if has_double_pipe else ",")]
        if self.type == int:
            self._values = parse_range_int(value_list)
        elif self.type == float:
            self._values = parse_range_float(value_list)
        else:
            self._values = [self._format_value(val) for val in value_list] # type: ignore
        return self

    def unset(self):
        self._index = 0
        self._values = list()
        self._valid = []

    def _format_value(self, value: str) -> AxisOption.type:
        cast_value = ""
        if self.type == int:
            cast_value = int(value)

        elif self.type == float:
            cast_value = round(float(value), 8)

        elif self.type == bool:
            cast_value = str(value).lower().strip()
            if cast_value in ["true", "yes", "1"]:
                cast_value = True
            elif cast_value in ["false", "no", "0"]:
                cast_value = False

        elif self.type == str:
            if self.choices is not None:
                valid_list = self.choices()
                cast_value = get_closest_from_list(value, valid_list)
        else:
            cast_value = value.strip()

        return cast_value


    def validate(self, p: Optional[SDP], value) -> None:
        """raise an error if the data type is incorrect"""
        same_type = type(value) == self.type
        if self.type == int or self.type == float:
            if not same_type:
                raise RuntimeError(f"must be a {self.type} number")
            elif self.min is not None and value < self.min: # type: ignore
                raise RuntimeError(f"must be at least {self.min}")
            elif self.max is not None and value > self.max: # type: ignore
                raise RuntimeError(f"must not exceed {self.max}")

        elif self.type == bool and not same_type:
            raise RuntimeError("must be either 'True' or 'False'")

        elif self.type == str and self.choices is not None and (not same_type or value == ""):
            raise RuntimeError("not matched to any entry in the list")
            
        elif not same_type:
            raise RuntimeError(f"must be a valid type")


    def validate_all(self, p: Optional[SDP] = None, quiet: bool = True):
        def validation(value):
            try:
                self.validate(p, value)
            except RuntimeError as err:
                return f"'{value}': {err=}"

        result = list(map(validation, self._values))

        if any(result):
            errors = [err for err in result if err]
            if not quiet:
                raise RuntimeError(f"Invalid parameters in {self.label}: {errors}")
            print(f"Invalid parameters in {self.label}: {errors}")
        self._valid = [err is None for err in result]


@dataclass
class AxisNothing(AxisOption):
    type: None = None

    def _apply(self, *args, **kwargs) :
        return


@dataclass
class AxisModel(AxisOption):
    _: KW_ONLY
    cost: float = 1.0 # change of checkpoints is too heavy, do it less often
        
    def validate(self, _, value:str):
        info = sd_models.get_closet_checkpoint_match(value)
        if info is None:
            raise RuntimeError("Unknown checkpoint")


@dataclass
class AxisVae(AxisOption):
    _: KW_ONLY
    cost: float = 0.7
        
    def validate(self, _, value:str):
        if value in ["None", "Automatic"]:
            return
        if sd_vae.vae_dict.get(value, None) is None:
            raise RuntimeError(f"Unknown vae: {value}")


@dataclass
class AxisReplace(AxisOption):
    _: KW_ONLY
    cost: float = 0.5 # to allow prompt to be replaced before string manipulation
    _values: list[str] = set_field(init=False, default_factory=list)
    __tag: str = set_field(init=False, default="")

    def _apply(self, p):
        """tranform the value on the Processing job"""
        value = str(self._values[self._index])
        p.prompt = p.prompt.replace(self.__tag, value)
        p.negative_prompt = p.negative_prompt.replace(self.__tag, value)

    def validate(self, p:SDP, _):
        # need validation at runtime
        if self.__tag is None or self.__tag == "":
            raise RuntimeError(f"Values not set")

        if self.__tag not in p.prompt and self.__tag not in p.negative_prompt:
            raise RuntimeError(f"'{self.__tag}' not found in all prompts")

    def set(self, values:str) -> AxisOption:
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
            value = value_pair.split('=', maxsplit=1)
            if len(value) == 1:
                tag = self.__tag or value[0]
                self._values.append(value[0])
            else:
                tag = self.__tag or value[0].strip()
                self._values.append(value[1].strip())
            self.__tag = tag
        self.label = self.label.replace("TAG", self.__tag)
        return self

    def unset(self):
        super().unset()
        self.label = self.label.replace(self.__tag, "TAG")
        self.__tag = ""

