import os
import torch
from .mmaudio.eval_utils import ModelConfig, generate, load_video, make_video
from .mmaudio.model.flow_matching import FlowMatching
from .mmaudio.model.networks import MMAudio, get_my_mmaudio
from .mmaudio.model.utils.features_utils import FeaturesUtils
import folder_paths

if not "mmaudio" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("mmaudio", os.path.join(folder_paths.models_dir, "mmaudio"))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# === Preload everything globally ===
model: ModelConfig = ModelConfig(
    model_name='large_44k_v2',
    model_path=folder_paths.get_full_path_or_raise('mmaudio', 'weights', 'mmaudio_large_44k_v2.pth'),
    vae_path=folder_paths.get_full_path_or_raise('mmaudio', 'ext_weights', 'v1-44.pth'),
    bigvgan_16k_path=None,
    mode = '44k',
    synchformer_ckpt=folder_paths.get_full_path_or_raise('mmaudio', 'ext_weights', 'synchformer_state_dict.pth'),
)
seq_cfg = model.seq_cfg

net: MMAudio = get_my_mmaudio(model.model_name).to('cuda', torch.float32).eval()
net.load_weights(torch.load(model.model_path, map_location='cuda', weights_only=True))    

feature_utils = FeaturesUtils(
    tod_vae_ckpt=model.vae_path,
    synchformer_ckpt=model.synchformer_ckpt,
    enable_conditions=True,
    mode=model.mode,
    bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
    need_vae_encoder=False
).to('cuda', torch.float32).eval()

fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=25)
rng = torch.Generator(device='cuda').manual_seed(42)

class MMAudioSamplerMod:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False} ),
                "duration": ("FLOAT", {"default": 5, "step": 0.01, "tooltip": "Duration of the audio in seconds"}),
                "cfg": ("FLOAT", {"default": 4.5, "step": 0.1, "tooltip": "Strength of the conditioning"}),
                "prompt": ("STRING", {"default": "", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True} )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sample"
    CATEGORY = "MMAudio"

    def sample(self, video_path, duration, cfg, prompt, negative_prompt, output_dir: str = 'output/mmaudio'):
        os.makedirs(output_dir, exist_ok=True)

        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        duration = video_info.duration_sec

        seq_cfg.duration = duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
        
        audios = generate(
            clip_frames, sync_frames, [prompt],
            negative_text=[negative_prompt],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg
        )

        audio = audios.float().cpu()[0]
        video_save_path = os.path.join(output_dir, os.path.basename(video_path))
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)

        return (video_save_path, ) if video_save_path else ('', )
        
NODE_CLASS_MAPPINGS = {
    "MMAudioSamplerMod": MMAudioSamplerMod
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMAudioSamplerMod": "MMAudio Sampler Mod"
}
