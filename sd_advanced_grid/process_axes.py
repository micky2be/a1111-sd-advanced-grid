import os
import math
from copy import copy
from functools import reduce
import re
import hashlib
import json

from PIL import Image
from dataclasses import dataclass, field
from typing import ClassVar, Any

from modules import processing, images, shared
from modules.processing import Processed, StableDiffusionProcessingTxt2Img as SDPT2I

from sd_advanced_grid.grid_settings import AxisOption

PROB_PATTERNS = ["date", "datetime", "job_timestamp", "batch_number", "generation_number", "seed"]

def generate_filename(proc:SDPT2I, cell_id:int, axis_set, keep_origin:bool = False):
    file_name = ""
    if keep_origin:
        # use pattern defined by the user
        re_pattern = re.compile(r"(\[([^\[\]<>]+)(?:<.+>|)\])");
        width, height = proc.width, proc.height
        namegen = images.FilenameGenerator(proc, proc.seed, proc.prompt, {"width": width, "height": height })
        filename_pattern = shared.opts.samples_filename_pattern or "[seed]-[prompt_spaces]"
        # remove patterns that may prevent existance detection
        for match in re_pattern.finditer(filename_pattern):
            pattern, keyword = match.groups()
            if keyword in PROB_PATTERNS:
                filename_pattern = filename_pattern.replace(pattern, "")

        file_name = namegen.apply(filename_pattern)
    else:
        # in JS: md5(JSON.stringify(axis_set, Object.keys(axis_set).sort(), 2))
        encoded = json.dumps(axis_set, sort_keys=True, indent=2).encode("utf-8")
        dhash = hashlib.md5(encoded)
        file_name = dhash.hexdigest()

    return f"adv_cell-{cell_id:03}-{file_name}"


@dataclass
class GridCell:
    # init
    proc: SDPT2I
    axis_set: dict[str, tuple[str, Any]]
    processed: Processed = field(init=False)
    __file_name: str = field(init=False)
    cell_id: int = field(init=False)
    job_count: int = field(init=False, default=1)
    skipped: bool = field(init=False, default=False)
    failed: bool = field(init=False, default=False)
    cell_counter: ClassVar[int] = 0

    def __post_init__(self):
        GridCell.cell_counter += 1
        self.cell_id = GridCell.cell_counter
        self.__file_name = generate_filename(self.proc, self.cell_id, self.axis_set)
        if self.proc.enable_hr:
            self.job_count *= 2

    def run(self, save_to: str, overwrite:bool = False, for_web:bool = False):
        file_ext = shared.opts.samples_format
        file_exist = os.path.exists(os.path.join(save_to, "images", f"{self.__file_name}.{file_ext}"))

        total_steps = self.proc.steps + ((self.proc.hr_second_pass_steps or self.proc.steps) if self.proc.enable_hr else 0)

        if file_exist and not overwrite:
            self.skipped = True
            if shared.total_tqdm._tqdm:
                # update console progessbar
                shared.total_tqdm._tqdm.update(total_steps)
            shared.state.nextjob()
            if self.proc.enable_hr:
                shared.state.nextjob()
            print(f"\nSkipping cell #{self.cell_id}, file already exist.")
            return

        print(f"\n\nRunning image generation for cell {self.cell_id} with the following attributes:")
        for label, value in self.axis_set.values():
            print(f" * {label}: {value}")
        print("")

        try:
            processed = processing.process_images(self.proc)
        except Exception as err:
            print(f"\nUnexpected {err=}, {type(err)=}")
            self.failed = True
            return

        if shared.state.interrupted:
            return

        if shared.state.skipped:
            self.skipped = True
            shared.state.skipped = False
            print(f"\nSkipping cell #{self.cell_id}, requested by the system.")
            return

        if not processed.images or not any(processed.images):
            print(f"\nNo images were generated for cell #{self.cell_id}")
            self.failed = True
            return

        info_text = processing.create_infotext(self.proc, self.proc.all_prompts, self.proc.all_seeds, self.proc.all_subseeds)
        processed.infotexts[0] = info_text
        image: Image.Image = processed.images[0]

        images.save_image(
            image,
            path=os.path.join(save_to, "images"),
            basename="",
            info=info_text,
            forced_filename=self.__file_name,
            extension=file_ext,
            save_to_dirs=False
        )

        self.processed = processed
        # image.thumbnail((512, 512))

        if for_web:
            # create and save thumbnail
            thumb = image.copy()
            thumb.thumbnail((512, 512))
            os.makedirs(os.path.join(save_to, "thumbnails"), exist_ok=True)
            thumb.save(os.path.join(save_to, "thumbnails", f"{self.__file_name}.png"))



def apply_axes(set_proc:SDPT2I, axes: list[AxisOption]):
    excs: list[Exception] = []
    axis_set: dict[str, tuple[str, Any]] = {}
    should_iter = True
    # self.proc.styles = self.proc.styles[:] # allows for multiple styles axis
    for axis in axes:
        try:
            axis.apply(set_proc)
        except Exception as err:
            print(f"\nUnexpected {err=}, {type(err)=}")
            excs.append(err)
        else:
            axis_set[axis.id] = (axis.label, axis.value)
        if should_iter:
            should_iter = not axis.next()
    return axis_set, excs

def prepare_jobs(adv_proc:SDPT2I, axes: list[AxisOption], job_count: int, grid_name:str, batches: int = 1):
    cells: list[GridCell] = []
    for x in range(job_count):
        set_proc = copy(adv_proc)
        processing.fix_seed(set_proc)
        set_proc.override_settings = copy(adv_proc.override_settings)
        set_proc.extra_generation_params = copy(set_proc.extra_generation_params)
        set_proc.extra_generation_params['Adv. Grid'] = grid_name
        axis_set, errors = apply_axes(set_proc, axes)
        if errors:
            continue
        cell = GridCell(set_proc, axis_set)
        cells.append(cell)

    return cells

def combine_processed(processed_result:Processed, processed: Processed):
    if processed_result.index_of_first_image == 0:
        # Use our first processed result object as a template container to hold our full results
        processed_result.info = processed.info
        processed_result.images = []
        processed_result.all_prompts = []
        processed_result.all_negative_prompts = []
        processed_result.all_seeds = []
        processed_result.all_subseeds = []
        processed_result.infotexts = []
        processed_result.index_of_first_image = 1

    if processed.images:
        # Non-empty list indicates some degree of success.
        processed_result.images.extend(processed.images)
        processed_result.all_prompts.extend(processed.all_prompts)
        processed_result.all_negative_prompts.extend(processed.all_negative_prompts)
        processed_result.all_seeds.extend(processed.all_seeds)
        processed_result.all_subseeds.extend(processed.all_subseeds)
        processed_result.infotexts.extend(processed.infotexts)

    return processed_result


def generate_grid(adv_proc:SDPT2I, grid_name:str, overwrite:bool, batches: int, test_run: bool, axes_settings: list[AxisOption]):
    axes_processing = axes_settings.copy()
    axes_processing.sort(key=lambda axis: axis.cost) # reorder to avoid heavy changes
    grid_path = os.path.join(adv_proc.outpath_grids, grid_name)
    GridCell.cell_counter = 0

    processed = Processed(adv_proc, [], adv_proc.seed, "", adv_proc.subseed)

    # note: batches are possible but only with prompt, negative_prompt, seeds, or subseed
    aprox_jobs = math.prod([axis.length for axis in axes_settings])
    cells = prepare_jobs(adv_proc, axes_processing, aprox_jobs, grid_name)

    os.makedirs(grid_path, exist_ok=True)
    grid_data = {
        "name": grid_name,
        "params": json.loads(processed.js()),
        "axis": [axis.dict() for axis in axes_processing]
    }

    with open(os.path.join(grid_path, "config.json"), "w") as file:
        file.write(json.dumps(grid_data, indent=2))

    if test_run:
        return processed

    shared.state.job_count = sum([cell.job_count for cell in cells])
    shared.state.processing_has_refined_job_count = True

    print(f"\nStarting generation of {len(cells)} variants")

    for x, cell in enumerate(cells):
        job_info = f"Generating variant #{x + 1} out of {len(cells)} - "
        shared.state.textinfo = job_info # type: ignore
        shared.state.job = job_info # seems to be unused
        cell.run(save_to=grid_path, overwrite=overwrite)
        cell.proc.close()
        if shared.state.interrupted:
            print(f"\nProcess interupted. Cancelling all jobs.\n")
            break
        if not cell.skipped and not cell.failed:
            combine_processed(processed, cell.processed)

    return processed
