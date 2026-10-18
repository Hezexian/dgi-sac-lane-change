"""Record a trained policy on one scenario as an MP4 video."""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root

from dgi_sac.env import make_env
from dgi_sac.evaluation import load_checkpoint, resolve_checkpoint

DISPLAY = dict(screen_width=1920, screen_height=640, scaling=16.0, centering_position=[0.3, 0.5],
               offscreen_rendering=True)
ROAD_Y = (-4.0, 16.0)  # metres shown vertically (four 4-m lanes and margins)
CAPTION = 40           # pixels


class _FrameHook:
    """highway-env calls this between policy steps, once per simulator step."""

    def __init__(self, env, on_frame):
        self.env, self.on_frame = env, on_frame

    def _capture_frame(self):
        self.on_frame(self.env.unwrapped.render())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint")
    parser.add_argument("--method", choices=["dgi_sac", "sac", "mlp", "td3", "ddpg"])
    parser.add_argument("--seed", type=int, help="paper seed 1-15 (TD3 and DDPG 1-10) of a released model")
    parser.add_argument("--models-dir", default="models", help="the extracted 'models' folder of the model archive")
    parser.add_argument("--scenario", type=int, default=380000)
    parser.add_argument("--output", help="MP4 path (default: videos/<name>_scenario<scenario>.mp4)")
    args = parser.parse_args()

    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    checkpoint, name = resolve_checkpoint(args)
    agent, config = load_checkpoint(checkpoint)
    env = make_env(config["env_name"], config["env"], render_mode="rgb_array")
    base = env.unwrapped
    base.config.update(DISPLAY)
    fps = int(base.config["simulation_frequency"])
    width, crop = DISPLAY["screen_width"], int(round((ROAD_Y[1] - ROAD_Y[0]) * DISPLAY["scaling"]))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    output = Path(args.output or f"videos/{name}_scenario{args.scenario}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (width, crop + CAPTION), fps=fps, codec="libx264",
                                         quality=8, pix_fmt_out="yuv420p", macro_block_size=8)
    writer.send(None)

    def on_frame(rgb):
        top = int(round((ROAD_Y[0] - base.viewer.sim_surface.origin[1]) * DISPLAY["scaling"]))
        canvas = Image.new("RGB", (width, crop + CAPTION), (255, 255, 255))
        canvas.paste(Image.fromarray(np.ascontiguousarray(rgb[top:top + crop])), (0, CAPTION))
        speed = base.vehicle.speed * math.cos(base.vehicle.heading)
        text = f"{name} · scenario {args.scenario} · t = {base.steps / fps:5.2f} s · ego speed {speed:4.1f} m/s"
        ImageDraw.Draw(canvas).text((14, CAPTION // 2), text, fill=(30, 30, 30), font=font, anchor="lm")
        writer.send(np.asarray(canvas).tobytes())

    try:
        random.seed(args.scenario)
        np.random.seed(args.scenario)
        torch.manual_seed(args.scenario)
        state, _ = env.reset(seed=args.scenario)
        env.action_space.seed(args.scenario)
        base._record_video_wrapper = _FrameHook(env, on_frame)
        on_frame(base.render())
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, _, terminated, truncated, _ = env.step(action)
            on_frame(base.render())
            done = terminated or truncated
    finally:
        writer.close()
        env.close()
    print(f"saved {output}")


if __name__ == "__main__":
    main()
