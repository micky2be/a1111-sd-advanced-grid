# Python
from copy import copy, deepcopy
from datetime import datetime
from typing import Any

# Lib
import gradio as gr
from typing_extensions import Unpack

# SD-WebUI
from modules import processing, scripts, sd_models, sd_vae, shared
from modules.processing import Processed
from modules.processing import StableDiffusionProcessingTxt2Img as SD_Proc
from modules.shared import opts
from modules.ui_components import ToolButton
from sd_advanced_grid.axis_options import axis_options
from sd_advanced_grid.grid_settings import SHARED_OPTS, AxisOption
from sd_advanced_grid.process_axes import generate_grid

# Local
from sd_advanced_grid.utils import logger

# ################################# Constants ################################ #

REFRESH_SYMBOL = "\U0001f504"  # 🔄
FILL_SYMBOL = "\U0001f4d2"  # 📒
STEP_FIELDS = ["steps", "hr_second_pass_steps"]
MIN_AXES = 4
MAX_AXES = 10
TEXT_PLACEHOLDER = ["Select a type", "Enter values"]

# ################################## Helpers ################################# #


class SharedOptionsCache:
    def __enter__(self):
        for key in SHARED_OPTS:
            setattr(self, key, getattr(opts, key, None))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key in SHARED_OPTS:
            setattr(opts, key, getattr(self, key))
        sd_models.reload_model_weights()
        sd_vae.reload_vae_weights()


# ########################## Gradio Event Functions ########################## #


def fill_axis(axis_index: int):
    axis = axis_options[axis_index]
    if axis.choices is not None:
        return axis.choices()
    return gr.update()


def update_input(axis_index: int):
    # TODO: set an input mode system (link or not to axis.type)
    has_type = axis_index > 0
    axis_type = axis_options[axis_index]
    has_choices = False
    choices = []
    if axis_type.choices is not None:
        choices = axis_type.choices()
        has_choices = True

    is_bool = axis_type.type == bool and not has_choices

    return [
        gr.Textbox.update(
            visible=not (has_choices or is_bool),
            interactive=has_type,
            value="",
            placeholder=TEXT_PLACEHOLDER[int(has_type)],
        ),
        gr.Dropdown.update(visible=has_choices, choices=choices, value=[]),
        gr.CheckboxGroup.update(visible=is_bool, value=[]),
        gr.Button.update(interactive=has_choices),
    ]


def populate_input(axis_index, values: list[str]):
    """Update the main input (Textbox) to rely on a single source"""
    axis_type = axis_options[axis_index]
    if axis_type.choices is not None:
        choices = axis_type.choices()
        values = list(filter(lambda x: x in choices, values))
    return ", ".join(values)


def update_axis(is_selected: bool):
    return [
        gr.Row.update(visible=is_selected),
        gr.Dropdown.update() if is_selected else gr.Dropdown.update(value=axis_options[0].label),
    ]


def change_button_status(nb_axis):
    return [gr.Button.update(interactive=nb_axis < MAX_AXES), gr.Button.update(interactive=nb_axis > MIN_AXES)]


def axis_display(nb_axis):
    return [gr.Checkbox.update(value=pos < nb_axis) for pos in range(MAX_AXES)]


def nb_axes_changed(nb_axis: int):
    cur_axes = nb_axis
    if nb_axis < MIN_AXES:
        cur_axes = MIN_AXES
    elif nb_axis > MAX_AXES:
        cur_axes = MAX_AXES
    return cur_axes


# ############################### Script Class ############################### #


class ScriptGrid(scripts.Script):
    # pylint: disable=too-many-locals
    BASEDIR = scripts.basedir()

    def title(self):
        return "Advanced Grid"

    def show(self, is_img2img: bool):
        return not is_img2img

    def ui(self, _):  # pylint: disable=invalid-name
        axes_ctrl: list[gr.components.Component] = []
        axes_selection: list[gr.components.Component] = []

        def build_axis_selection(axis_count: int):
            with gr.Row(visible=axis_count <= MIN_AXES, elem_classes="axis-row").style(equal_height=True) as axis_row:
                row_visibility = gr.Checkbox(value=axis_count <= MIN_AXES, visible=False)
                row_type = gr.Dropdown(
                    label=f"Axis {axis_count} Type",
                    choices=[axis.label for axis in axis_options],
                    value=axis_options[0].label,
                    type="index",
                )
                row_value = gr.Textbox(
                    label=f"Axis {axis_count} Values", interactive=False, lines=1, placeholder=TEXT_PLACEHOLDER[0]
                )
                # TODO: add an edit/save option for the multiselect to easily change the order
                row_value_list = gr.Dropdown(label=f"Axis {axis_count} Values", visible=False, multiselect=True)
                row_value_bool = gr.CheckboxGroup(
                    label=f"Axis {axis_count} Values", visible=False, choices=["On", "Off"]
                )
                fill_row_button = ToolButton(value=FILL_SYMBOL, interactive=False, visible=True)
                # row_value could be based on input Axis
                # e.g. numbers: from, to, and steps or increment
                #      AxisReplace: Texbox with a token system (need new Gradio component)
            row_visibility.change(update_axis, inputs=[row_visibility], outputs=[axis_row, row_type])  # type: ignore
            row_type.change(
                update_input, inputs=[row_type], outputs=[row_value, row_value_list, row_value_bool, fill_row_button]
            )
            row_value_list.input(populate_input, inputs=[row_type, row_value_list], outputs=[row_value])
            row_value_bool.input(populate_input, inputs=[row_type, row_value_bool], outputs=[row_value])
            fill_row_button.click(fill_axis, inputs=[row_type], outputs=[row_value_list]) \
                .then(populate_input, inputs=[row_type, row_value_list], outputs=[row_value])
            axes_selection.extend([row_type, row_value])
            return row_visibility

        with gr.Column(elem_id="sd_adv_grid_settings", variant="box"):
            with gr.Row():
                grid_name = gr.Textbox(
                    value="", placeholder="Enter grid name", label="Output folder name (if blank uses current date)"
                )

            for i in range(MAX_AXES):
                axes_ctrl.append(build_axis_selection(i + 1))

            with gr.Row(elem_id=self.elem_id("axis-control")):
                # Column(variant="box")
                gr.Button(value="Clear all")
                add_button = gr.Button(value="Add an Axis")
                del_button = gr.Button(value="Remove last Axis", interactive=False)
                nb_axes = gr.Number(value=MIN_AXES, precision=0, visible=False)

            with gr.Row(variant="compact", elem_classes="extra-options").style(equal_height=True):
                do_overwrite = gr.Checkbox(
                    value=False, label="Overwrite", info="Overwrite existing images (for updating grids)"
                )
                test_run = gr.Checkbox(value=False, label="Dry run", info="Do a dry run to validate your grid")
                allow_batches = gr.Checkbox(
                    value=False, label="Use batches", info="Will run in batches where possible"
                )
                force_vae = gr.Checkbox(
                    value=False,
                    label="Force VAE",
                    info="Force selected VAE\n(ignores checkpoint matching if any)",
                    visible=False,
                )
                for_web = gr.Checkbox(
                    value=False,
                    label="For Web",
                    info="Create Web Interface",
                )
                # fixed seed option?

        # fmt: off
        nb_axes.change(nb_axes_changed, inputs=[nb_axes], outputs=[nb_axes]) \
            .success(axis_display, inputs=[nb_axes], outputs=axes_ctrl) \
            .success(change_button_status, inputs=[nb_axes], outputs=[add_button, del_button])
        # fmt: on

        # clear_button.click(lambda: 0, outputs=[nb_axes]).then(axis_display, inputs=[nb_axes], outputs=axes_ctrl)
        add_button.click(lambda nb: nb + 1, inputs=[nb_axes], outputs=[nb_axes])
        del_button.click(lambda nb: nb - 1, inputs=[nb_axes], outputs=[nb_axes])

        return [grid_name, do_overwrite, allow_batches, test_run, force_vae, for_web] + axes_selection

    def run(
        self,
        sd_processing: SD_Proc,
        grid_name: str,
        overwrite: bool,
        allow_batches: bool,
        test_run: bool,
        force_vae: bool,
        for_web: bool,
        *axes_selection: Unpack[tuple[Any, ...]],
    ) -> Processed:
        if not grid_name:
            grid_name = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

        # Clean up default params
        adv_proc = copy(sd_processing)
        processing.fix_seed(adv_proc)
        adv_proc.override_settings_restore_afterwards = False
        adv_proc.n_iter = 1
        adv_proc.do_not_save_grid = True
        adv_proc.do_not_save_samples = True
        batches = adv_proc.batch_size if allow_batches else 1
        adv_proc.batch_size = adv_proc.batch_size if allow_batches else 1

        if force_vae:
            # adv_proc.override_settings["sd_vae_as_default"] = False
            pass

        total_steps: list[int] = [adv_proc.steps, 0]
        variation = 1
        if adv_proc.enable_hr:
            total_steps[1] = adv_proc.hr_second_pass_steps or adv_proc.steps

        # regroup pairs and filter out "Nothing" axes and empty values
        axes_settings: list[AxisOption] = []
        for i in range(0, len(axes_selection), 2):
            axis_index: int
            axis_values: str
            axis_index, axis_values = axes_selection[i : i + 2]
            if not axis_index or not axis_values:
                continue
            axis = deepcopy(axis_options[axis_index])
            axes_settings.append(axis.set(axis_values))

            index = STEP_FIELDS.index(axis.id) if axis.id in STEP_FIELDS else None
            if index is not None:
                total_steps[index] = sum(axis.values)  # type: ignore
            else:
                variation *= axis.length
            axis.validate_all(proc=adv_proc)
            if not axis.is_valid:
                logger.warn(f"{axis.label} might contain invalid values")

        if not test_run:
            shared.total_tqdm.updateTotal(sum(total_steps) * variation)

        with SharedOptionsCache():
            result = generate_grid(adv_proc, grid_name, overwrite, batches, test_run, axes_settings, for_web)

        for axis in axes_settings:
            axis.unset()

        logger.info("Done!")
        return result
