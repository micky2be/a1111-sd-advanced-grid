# Python
import math
import re
import hashlib
import json
import string
from copy import copy
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

# Lib
from PIL import Image

# SD-WebUI
from modules import processing, images, shared
from modules.processing import Processed, StableDiffusionProcessingTxt2Img as SDP

# Local
from sd_advanced_grid.grid_settings import AxisOption
from sd_advanced_grid.utils import clean_name

# ################################# Constants ################################ #

CHAR_SET = string.digits + string.ascii_letters
PROB_PATTERNS = ["date", "datetime", "job_timestamp", "batch_number", "generation_number", "seed"]

# ############################# Helper Functions ############################# #


def convert(num: int):
    """convert a decimal number into an alphanumerical value"""
    base = len(CHAR_SET)
    converted = ""
    while num:
        digit = num % base
        converted += CHAR_SET[digit]
        num //= base
    return converted[::-1].zfill(2)

def generate_filename(proc:SDP, axis_set, keep_origin:bool = False):
    """generate a filename for each images based on data to be processed"""
    file_name = ""
    if keep_origin:
        # use pattern defined by the user
        # FIXME: how will fontend know about the filename?
        re_pattern = re.compile(r"(\[([^\[\]<>]+)(?:<.+>|)\])")
        width, height = proc.width, proc.height
        namegen = images.FilenameGenerator(proc, proc.seed, proc.prompt, {"width": width, "height": height })
        filename_pattern = shared.opts.samples_filename_pattern or "[seed]-[prompt_spaces]"
        # remove patterns that may prevent existance detection
        for match in re_pattern.finditer(filename_pattern):
            pattern, keyword = match.groups()
            if keyword in PROB_PATTERNS:
                filename_pattern = filename_pattern.replace(pattern, "")

        file_name = f"{namegen.apply(filename_pattern)}"
    else:
        # in JS: md5(JSON.stringify(axis_set, Object.keys(axis_set).sort(), 2))
        encoded = json.dumps(axis_set, sort_keys=True, indent=2).encode("utf-8")
        dhash = hashlib.md5(encoded)
        file_name = f"{dhash.hexdigest()}"

    return file_name


def apply_axes(set_proc:SDP, axes_settings: list[AxisOption]):
    """
    run through each axis to apply current active values,
    then select next available value on an axis
    """
    axes = axes_settings.copy()
    axes.sort(key=lambda axis: axis.cost) # reorder to avoid heavy changes

    excs: list[Exception] = []
    axis_set: dict[str, tuple[str, Any]] = {}
    axis_code = ["00"] * len(axes_settings)
    should_iter = True

    # self.proc.styles = self.proc.styles[:] # allows for multiple styles axis
    for axis in axes:
        axis_code[axes_settings.index(axis)] = convert(axis.index + 1)
        try:
            axis.apply(set_proc)
        except RuntimeError as err:
            print(f"\nUnexpected {err=}")
            excs.append(err)
        else:
            axis_set[axis.id] = (axis.label, axis.value)
        if should_iter:
            should_iter = not axis.next()
    return axis_set, "".join(axis_code[::-1]), excs


def prepare_jobs(adv_proc:SDP, axes_settings: list[AxisOption], job_count: int, grid_name:str, batches: int = 1):
    """create a dedicated processing instance for each variation with different axes values"""

    if batches > 1:
        # note: batches are possible but only with prompt, negative_prompt, seeds, or subseed
        pass

    cells: list[GridCell] = []

    for _ in range(job_count):
        set_proc = copy(adv_proc)
        processing.fix_seed(set_proc)
        set_proc.override_settings = copy(adv_proc.override_settings)
        set_proc.extra_generation_params = copy(set_proc.extra_generation_params)
        set_proc.extra_generation_params["Adv. Grid"] = grid_name
        axis_set, axis_code, errors = apply_axes(set_proc, axes_settings)
        if errors:
            # TODO: option to break here
            continue
        cell = GridCell(axis_code, set_proc, axis_set)
        cells.append(cell)

    return cells


def combine_processed(processed_result:Processed, processed: Processed):
    """combine all processed data to allow a single disaply in SD WebUI"""
    if processed_result.index_of_first_image == 0:
        # Use our first processed result object as a template container to hold our full results
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

def file_exist(folder:Path, base_name:str, cell_id:str):
    files = sorted(folder.glob(f"adv_cell-*{base_name}.*"))
    if len(files) > 1:
        # SD pattern might create collision, adding cell_id to pinpoint existance (still not guaranteed)
        files = sorted(folder.glob(f"adv_cell-{cell_id}-{base_name}.*"))
    return any(files)

# ####################### Logic For Individual Variant ####################### #
@dataclass
class GridCell:
    # init
    cell_id: str
    proc: SDP
    axis_set: dict[str, tuple[str, Any]]
    processed: Processed = field(init=False)
    job_count: int = field(init=False, default=1)
    skipped: bool = field(init=False, default=False)
    failed: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.proc.enable_hr:
            # NOTE: there might be some extensions that add jobs
            self.job_count *= 2

    def run(self, save_to: Path, overwrite:bool = False, for_web:bool = False):
        base_name = generate_filename(self.proc, self.axis_set)
        file_name = f"adv_cell-{self.cell_id}-{base_name}"
        file_ext = shared.opts.samples_format
        file_path = save_to.joinpath(f"{file_name}.{file_ext}")

        total_steps = self.proc.steps + ((self.proc.hr_second_pass_steps or self.proc.steps) if self.proc.enable_hr else 0)

        if file_exist(save_to, base_name, self.cell_id) and not overwrite:
            # pylint: disable=protected-access
            self.skipped = True
            if shared.total_tqdm._tqdm:
                # update console progessbar
                shared.total_tqdm._tqdm.update(total_steps)
            shared.state.nextjob()
            if self.proc.enable_hr:
                # NOTE: not sure if this is needed or automatic, progressbar update is finicky
                shared.state.nextjob()
            print(f"\nSkipping cell #{self.cell_id}, file already exist.")
            return

        print(f"\n\nRunning image generation for cell {self.cell_id} with the following attributes:")
        for label, value in self.axis_set.values():
            print(f" * {label}: {value}")
        print("")

        # All the magic happens here
        processed = processing.process_images(self.proc)

        if shared.state.interrupted:
            return

        if shared.state.skipped:
            # pylint: disable=protected-access
            self.skipped = True
            shared.state.skipped = False
            if shared.total_tqdm._tqdm:
                # update console progessbar (to be tested)
                shared.total_tqdm._tqdm.update(total_steps - shared.state.sampling_step)
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
            path=str(file_path.parent),
            basename="",
            info=info_text,
            forced_filename=file_path.stem,
            extension=file_path.suffix[1:],
            save_to_dirs=False
        )

        self.processed = processed
        # image.thumbnail((512, 512)) # could be useful to reduce memory usage (need testing)

        if for_web:
            # create and save thumbnail
            file_path = save_to.parent.joinpath("thumbnails", f"{file_name}.png")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            thumb = image.copy()
            thumb.thumbnail((512, 512))
            thumb.save(file_path)

        print(f"\nCell {self.cell_id} saved as {file_path.stem}")


# ########################## Generation Entry Point ########################## #

def generate_grid(adv_proc:SDP, grid_name:str, overwrite:bool, batches: int, test_run: bool, axes_settings: list[AxisOption]):
    grid_path = Path(adv_proc.outpath_grids, f"adv_grid_{clean_name(grid_name)}")

    processed = Processed(adv_proc, [], adv_proc.seed, "", adv_proc.subseed)

    aprox_jobs = math.prod([axis.length for axis in axes_settings])
    cells = prepare_jobs(adv_proc, axes_settings, aprox_jobs, grid_name, batches)

    grid_path.mkdir(parents=True, exist_ok=True)
    grid_data = {
        "name": grid_name,
        "params": json.loads(processed.js()),
        "axis": [axis.dict() for axis in axes_settings]
    }

    with grid_path.joinpath("config.json").open(mode="w", encoding="UTF-8") as file:
        file.write(json.dumps(grid_data, indent=2))

    if test_run:
        return processed

    shared.state.job_count = sum(cell.job_count for cell in cells)
    shared.state.processing_has_refined_job_count = True

    print(f"\nStarting generation of {len(cells)} variants")

    for i, cell in enumerate(cells):
        job_info = f"Generating variant #{i + 1} out of {len(cells)} - "
        shared.state.textinfo = job_info # type: ignore
        shared.state.job = job_info # seems to be unused
        cell.run(save_to=grid_path.joinpath("images"), overwrite=overwrite)
        cell.proc.close()
        if shared.state.interrupted:
            print("\nProcess interupted. Cancelling all jobs.\n")
            break
        if not cell.skipped and not cell.failed:
            combine_processed(processed, cell.processed)

    return processed
