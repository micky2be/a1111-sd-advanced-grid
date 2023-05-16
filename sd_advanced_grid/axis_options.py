# SD-WebUI
from modules import shared, sd_models, sd_vae, sd_samplers

# Local
from sd_advanced_grid.grid_settings import AxisOption, AxisModel, AxisVae, AxisReplace, AxisNothing

# TODO: create a system to easily add options and refer to it by field name

# ############################## Default Options ############################# #

axis_options: list[AxisOption] = [
    ## Common to txt2img and img2img
    AxisNothing("Nothing"),
    AxisModel("Checkpoint",                 type=str,                   field="sd_model_checkpoint",        choices=lambda: list(sd_models.checkpoints_list)),
    AxisVae("VAE",                          type=str,                   field="sd_vae",                     choices=lambda: list(sd_vae.vae_dict) + ["None", "Automatic"]),
    AxisOption("Seed",                      type=int,                   field="seed"),
    AxisOption("Steps",                     type=int,   max=200,        field="steps"),
    AxisOption("ClipSkip",                  type=int,   min=1,  max=12, field="CLIP_stop_at_last_layers"),
    AxisOption("Sampler",                   type=str,                   field="sampler_name",               choices=lambda: [x.name for x in sd_samplers.all_samplers]),
    AxisOption("CFG Scale",                 type=float, max=30,         field="cfg_scale"),

    AxisOption("Restore Faces",             type=str,                   field="face_restoration_model",     toggles="restore_faces",    choices=lambda: list(map(lambda m: m.name(), shared.face_restorers)) + ["None", "Default"]),
    AxisOption("CodeFormer Weight",         type=float,                 field="code_former_weight",         toggles="restore_faces"),
    AxisOption("Tiling",                    type=bool,                  field="tiling"),
    # AxisOption("Width",                     type=int,                   field="width"),
    # AxisOption("Height",                    type=int,                   field="height"),

    AxisReplace("Replace TAG",              type=str),
    # AxisOption("Prompt",                    type=str,                   field="prompt"),
    # AxisOption("Negative Prompt",           type=str,                   field="negative_prompt"),

    AxisOption("Var Seed",                  type=int,                   field="subseed"),
    AxisOption("Var Strength",              type=float,                 field="subseed_strength"),

    # Sampler parameters
    AxisOption("ETA",                       type=float,                 field="eta"),
    AxisOption("ETA Noise Seed Delta",      type=int,                   field="eta_noise_seed_delta"),
    AxisOption("Sigma Churn",               type=float,                 field="s_churn"),
    AxisOption("Sigma TMin",                type=float,                 field="s_tmin"),
    AxisOption("Sigma TMax",                type=float,                 field="s_tmax"),
    AxisOption("Sigma Noise",               type=float,                 field="s_noise"),
    AxisOption("UniPC Order",               type=int,                   field="uni_pc_order",               cost=0.5),

    ## txt2img
    AxisOption("HighRes Upscaler",          type=str,                   field="hr_upscaler",                toggles="enable_hr",    choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]  + ["None"]),
    AxisOption("HighRes Scale",             type=float, min=1,  max=4,  field="hr_scale",                   toggles="enable_hr"),
    AxisOption("HighRes Steps",             type=int,   max=200,        field="hr_second_pass_steps",       toggles="enable_hr"),
    AxisOption("Denoising",                 type=float,                 field="denoising_strength",         toggles="enable_hr"),

    # -> opts.use_old_hires_fix_width_height
    # AxisOption("HighRes Resize Width",      type=int,                   field="hr_resize_x"),
    # AxisOption("HighRes Resize Height",     type=int,                   field="hr_resize_y"),

    # -> custom size
    # AxisOption("HighRes Upscale to Width",  type=int,                   field="hr_upscale_to_x"),
    # AxisOption("HighRes Upscale to Height", type=int,                   field="hr_upscale_to_y"),

    ## img2img
    # AxisOption("Sampler",                   type=str,                   field="sampler_name",                     choices=lambda: [x.name for x in sd_samplers.all_samplers]),
    # AxisOption("Denoising",                 type=float,                 field="denoising_strength"),
    # AxisOption("Image Mask Weight",         type=float,                 field="inpainting_mask_weight"),
    # AxisOption("Image CFG Scale",           type=float, max=3,          field="image_cfg_scale"),
]
