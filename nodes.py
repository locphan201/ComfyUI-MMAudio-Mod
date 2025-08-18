from pathlib import Path
from typing import Optional
import torch
from mmaudio.eval_utils import (
    ModelConfig, all_model_cfg, generate, load_video, make_video
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils
import os

#region sampling
class MMAudioSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False} ),
                "duration": ("FLOAT", {"default": 5, "step": 0.01, "tooltip": "Duration of the audio in seconds"}),
                "steps": ("INT", {"default": 25, "step": 1, "tooltip": "Number of steps to interpolate"}),
                "cfg": ("FLOAT", {"default": 4.5, "step": 0.1, "tooltip": "Strength of the conditioning"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sample"
    CATEGORY = "MMAudio"

    def sample(self, video_path, duration, steps, cfg, seed, prompt, negative_prompt):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # === Preload everything globally ===
        model: ModelConfig = all_model_cfg['large_44k_v2']
        model.download_if_needed()
        seq_cfg = model.seq_cfg

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        DTYPE = torch.bfloat16  # Set to torch.float32 for full precision if needed

        net: MMAudio = get_my_mmaudio(model.model_name).to(DEVICE, DTYPE).eval()
        net.load_weights(torch.load(model.model_path, map_location=DEVICE, weights_only=True))

        feature_utils = FeaturesUtils(
            tod_vae_ckpt=model.vae_path,
            synchformer_ckpt=model.synchformer_ckpt,
            enable_conditions=True,
            mode=model.mode,
            bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
            need_vae_encoder=False
        ).to(DEVICE, DTYPE).eval()

        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=steps)
        rng = torch.Generator(device=DEVICE).manual_seed(seed)
        
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_path.expanduser()
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
        video_save_path = None
        video_save_path = output_dir / f'{video_path.stem}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)

        return (video_save_path.absolute(), ) if video_save_path else ('', )
        
NODE_CLASS_MAPPINGS = {
    "MMAudioSampler": MMAudioSampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMAudioSampler": "MMAudio Sampler"
}
