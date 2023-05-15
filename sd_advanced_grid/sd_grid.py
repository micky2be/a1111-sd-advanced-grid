"""Entry point of the extension"""
from datetime import datetime
from copy import copy

import gradio as gr

from modules import shared, sd_models, sd_vae, scripts, processing
from modules.processing import Processed, StableDiffusionProcessing, StableDiffusionProcessingTxt2Img
from modules.shared import opts
from modules.ui_components import ToolButton

from sd_advanced_grid.grid_settings import AxisOption, SHARED_OPTS
from sd_advanced_grid.process_axes import generate_grid
from sd_advanced_grid.axis_options import axis_options

## Constants
refresh_symbol = '\U0001f504'  # 🔄
fill_values_symbol = "\U0001f4d2"  # 📒


## Helpers

class SharedOptionsCache(object):
    def __enter__(self):
        for key in SHARED_OPTS:
            setattr(self, key, getattr(opts, key, None))
  
    def __exit__(self, exc_type, exc_value, tb):
        for key in SHARED_OPTS:
            setattr(opts, key, getattr(self, key))
        sd_models.reload_model_weights()
        sd_vae.reload_vae_weights()


## Script class

class ScriptGrid(scripts.Script):
    BASEDIR = scripts.basedir()

    def title(self):
        return "Advanced Grid"

    def show(self, is_img2img: bool):
        return not is_img2img

    def ui(self, is_img2img):
        max_axes = 10
        min_axes = 4
        cur_axes = min_axes
        axes_ctrl: list[gr.Checkbox] = []
        axes_selection: list[gr.components.Component] = []

        def fill_axis(axis_index: int):
            axis = axis_options[axis_index]
            if axis.type == bool:
                return "true, false"
            elif axis.choices is not None:
                return ", ".join(list(axis.choices()))
            return gr.update()

        def on_axis_change(axis_index: int, axis_value):
            axis_type = axis_options[axis_index]
            return [
                gr.Textbox.update(interactive=axis_index!=0, value=""),
                gr.Button.update(interactive=axis_type.choices is not None or axis_type.type == bool)
            ]

        def update_axis(is_selected: bool):
            return [
                gr.Row.update(visible=is_selected),
                gr.Dropdown.update() if is_selected else gr.Dropdown.update(value=axis_options[0].label)
            ]

        def build_axis_selection(axis_count: int):
            with gr.Row(visible=axis_count <= min_axes).style(equal_height=True) as axis_row:
                row_status = gr.Checkbox(value=axis_count <= min_axes, visible=False)
                row_type = gr.Dropdown(label=f"Axis {axis_count} Type", choices=[axis.label for axis in axis_options], value=axis_options[0].label, type="index")
                row_value = gr.Textbox(label=f"Axis {axis_count} Values", interactive=False, lines=1)
                fill_row_button = ToolButton(value=fill_values_symbol, interactive=False)
                # row_value could be based on input Axis
                # e.g. multi select for choices
                #      double checkboxes for boolean
                #      sliders and steps for numbers
            row_status.change(update_axis, inputs=[row_status], outputs=[axis_row, row_type]) #type: ignore
            row_type.change(on_axis_change, inputs=[row_type, row_value], outputs=[row_value, fill_row_button])
            fill_row_button.click(fill_axis, inputs=[row_type], outputs=[row_value])
            axes_selection.extend([row_type, row_value])
            return row_status

        with gr.Column(elem_id="sd_adv_grid_settings", variant="box"):
            with gr.Row():
                grid_name = gr.Textbox(value="", placeholder="Enter grid name", label="Output folder name (if blank uses current date)")

            for i in range(max_axes):
                axes_ctrl.append(build_axis_selection(i + 1))

            with gr.Row(elem_id=self.elem_id("axis-control")):
                # Column(variant="box")
                clear_button = gr.Button(value="Clear all")
                add_button = gr.Button(value="Add an Axis")
                del_button = gr.Button(value="Remove last Axis", interactive=False)

            with gr.Row(variant="compact", elem_classes="extra-options").style(equal_height=True):
                do_overwrite = gr.Checkbox(value=False, label="Overwrite", info="Overwrite existing images (for updating grids)")
                test_run = gr.Checkbox(value=False, label="Dry run", info="Do a dry run to validate your grid")
                allow_batches = gr.Checkbox(value=False, label="Use batches", info="Will run in batches where possible", visible=False)
                force_vae = gr.Checkbox(value=False, label="Force VAE", info="Force selected VAE\n(ignores checkpoint matching if any)", visible=False)
                # fixed seed option?

        def button_status():
            return {
                add_button: gr.Button.update(interactive=cur_axes < max_axes),
                del_button: gr.Button.update(interactive=cur_axes > min_axes)
            }

        def axis_display(nb_axis):
            nonlocal cur_axes
            if nb_axis < min_axes:
                cur_axes = min_axes
            elif nb_axis > max_axes:
                cur_axes = max_axes
            else:
                cur_axes = nb_axis

            return [gr.Checkbox.update(value=True)]*cur_axes + [gr.Checkbox.update(value=False)]*(max_axes-cur_axes)

        def attach_event(target: gr.Button, mod: int | None = None):
            if mod is None:
                # clear all from content, even the minimum displayed axis
                dep = target.click(lambda: axis_display(0), outputs=axes_ctrl).then(lambda: axis_display(min_axes), outputs=axes_ctrl) # type: ignore
            else:
                dep = target.click(lambda: axis_display(cur_axes + mod), outputs=axes_ctrl) # type: ignore
            return dep.success(button_status, outputs=[add_button, del_button])

        attach_event(clear_button)
        attach_event(add_button, 1)
        attach_event(del_button, -1)

        return [grid_name, do_overwrite, allow_batches, test_run, force_vae] + axes_selection

    def run(self, sd_processing: StableDiffusionProcessingTxt2Img, grid_name, overwrite, allow_batches, test_run, force_vae, *axes_selection) -> Processed:
        if grid_name is None or grid_name == "":
            grid_name = f"adv_grid_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

        # Clean up default params
        adv_proc = copy(sd_processing)
        adv_proc.override_settings_restore_afterwards = False
        adv_proc.n_iter = 1
        adv_proc.batch_size = 1
        adv_proc.do_not_save_grid = True
        adv_proc.do_not_save_samples = True
        batches = adv_proc.batch_size if allow_batches else 1

        adv_proc.override_settings["save_images_add_number"] = False
        # adv_proc.override_settings["sd_vae_as_default"] = False
        processing.fix_seed(adv_proc)

        STEPS_MOD = ["steps", "hr_second_pass_steps"]
        total_steps: list[int] = [adv_proc.steps, 0]
        variation = 1
        if adv_proc.enable_hr:
            total_steps[1] = adv_proc.hr_second_pass_steps or adv_proc.steps

        # regroup pairs and filter out "Nothing" axes and empty values
        axes_settings: list[AxisOption] = []
        for i in range(0, len(axes_selection), 2):
            axis_index, axis_values = axes_selection[i:i+2]
            if axis_index is None or axis_index == 0 or axis_values == "":
                continue
            axis = axis_options[axis_index]
            axes_settings.append(axis.set(axis_values))

            index = STEPS_MOD.index(axis.id) if axis.id in STEPS_MOD else None
            if index is not None:
                total_steps[index] = sum(axis.values) # type: ignore
            else:
                variation *= axis.length
            axis.validate_all(adv_proc)
            if not axis.is_valid:
                print(f"\n**WARNING** {axis.label} might contain invalid values")

        shared.total_tqdm.updateTotal(sum(total_steps)*variation)

        with SharedOptionsCache():
            result = generate_grid(adv_proc, grid_name, overwrite, batches, test_run, axes_settings)

        for axis in axes_settings:
            axis.unset()

        return result
