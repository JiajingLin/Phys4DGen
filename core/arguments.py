import os
import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser(description='NeRF2Physics')
    
    # General arguments
    parser.add_argument('--scene_dir', type=str, default="data/demo",
                        help='path to scene (default: ./data/flower/)')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device for torch (default: cuda)')
    
    # Material Grouping
    parser.add_argument('--root_dir', type=str, default="./data",
                      help='Path to root directory')
    parser.add_argument('--checkpoint', type=str, default="/sda2/jiajinglin/model/SAM2/sam2.1_hiera_large.pt",
                      help='Path to SAM model checkpoint')
    parser.add_argument('--config', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                      help='Path to model configuration file')
    parser.add_argument('--port', type=int, default=7860,
                      help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true',
                      help='Create public Gradio link')
    parser.add_argument('--mask-save-dir', type=str, default=None,
                      help='Directory to save output masks and videos')
    
    # Material proposal
    parser.add_argument('--gpt_model_name', type=str, default="gpt-3.5-turbo",
                        help='GPT model name (default: gpt-3.5-turbo)')
    
    # GS Simulation
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true", default=True)
    parser.add_argument("--compile_video", action="store_true", default=True)
    parser.add_argument("--white_bg", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=True)  

    # Visualization
    parser.add_argument('--show', type=int, default=1,
                        help='whether to show interactive viewer (default: 1)')
    parser.add_argument('--compositing_alpha', type=float, default=0.2,
                        help='alpha for compositing with RGB image (default: 0.2)')
    parser.add_argument('--cmap_min', type=float, default=500,
                        help='minimum physical property value for colormap (default: 500)')
    parser.add_argument('--cmap_max', type=float, default=3500,
                        help='maximum physical property value for colormap (default: 3500)')
    parser.add_argument('--viz_save_name', type=str, default="tmp",
                        help='visualization save name (default: tmp)')
    
    args = parser.parse_args()
    
    args.scene_name = os.path.basename(args.scene_dir)

    args.yaml_file = os.path.join("config", "yaml", args.scene_name + "_config.yaml")
    with open(args.yaml_file, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in yaml_data.items():
        setattr(args, key, value)

    return args

if __name__ == "__main__":
    args = get_args()
    print(args)