import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import subprocess
import colorsys
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import cv2
import gradio as gr
import imageio.v2 as iio
from loguru import logger as guru
from pathlib import Path
import traceback
import shutil
import tempfile
from torch.cuda.amp import autocast  # Ensure autocast is imported
from core.clip_fusion import *
from core.arguments import get_args

# Set the default torch dtype to float32
#torch.set_default_dtype(torch.bfloat16)

def setup_gpu():
    if not torch.cuda.is_available():
        return

    # Disable TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

@dataclass
class InferenceState:
    """Stores the state of model inference"""
    video_path: str
    features: Optional[torch.Tensor] = None
    current_masks: Dict[int, np.ndarray] = None
    current_logits: Dict[int, np.ndarray] = None

class VideoSegmentation:
    """Main class for video and image segmentation with SAM model"""
    def __init__(self, args, checkpoint_dir: str, model_cfg: str, save_dir: str = None):
        self.args = args
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_cfg = model_cfg
        self.save_dir = Path(save_dir) if save_dir else None
        self.sam_model = None
        self.inference_state = None
        
        # Point selection state
        self.selected_points: List[List[float]] = []
        self.selected_labels: List[float] = []
        self.current_label = 1.0
        
        # Frame and mask state
        self.frame_index = 0
        self.current_image = None
        self.current_mask_index = 0 
        self.masks: Dict[int, Dict[int, np.ndarray]] = {}
        self.logits: Dict[int, Dict[int, np.ndarray]] = {}
        
        # Output state
        self.index_masks: List[np.ndarray] = []
        self.color_masks: List[np.ndarray] = []
        
        # Image loading state
        self.image_dir = ""
        self.image_paths: List[Path] = []
        
        # Processing mode
        self.mode = "video"  # "video" or "image"
        
        
        # self.scene_name = args.scene_name
        self.root_dir = args.root_dir
        self.scene_name = None
        self.selected_scene_name = gr.State(value='None')
        
        self._initialize_model()

    def set_mode(self, mode: str, scene_name: str):
        """Set processing mode (video/image)"""
        self.mode = mode
        self.scene_name = scene_name
        self.reset_state()
    
    def process_single_image(self, image_path: str) -> Tuple[np.ndarray, str]:
        """Process single image for segmentation"""
        try:
            # Read and preprocess image
            image = iio.imread(image_path)
            if image.ndim == 3 and image.shape[2] > 3:
                image = image[:, :, :3]
            
            self.current_image = image
            self.mode = "image"
            
            # Create temporary directory for the single image
            temp_dir = Path("temp_image")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            
            # Save image to temporary directory with numeric name
            temp_image_path = temp_dir / "00000.jpg"  # 숫자 형식의 파일명 사용
            iio.imwrite(str(temp_image_path), image)
            
            # Set image directory and paths
            
            
            
            self.image_dir = str(temp_dir)
            self.image_paths = [temp_image_path]
            self.frame_index = 0  # 명시적으로 프레임 인덱스 설정
            
            return image, "Image loaded successfully. Initialize SAM to begin segmentation."
            
        except Exception as e:
            guru.error(f"Error loading image: {e}")
            return None, f"Error loading image: {str(e)}"
        
    def save_single_mask(self) -> str:
        """Save mask for single image"""
        try:
            if not self.index_masks:
                return "No mask to save. Please create a mask first."

            # Use custom save directory if specified, otherwise use default
            
            # output_dir = self.save_dir if self.save_dir else Path(self.image_dir).parent / "frame_result"
            output_dir = self.save_dir if self.save_dir else Path(self.root_dir, self.scene_name) / "sam_result"
            output_dir.mkdir(exist_ok=True, parents=True) 

            # Use the original image name for the mask files
            # original_name = self.image_paths[0].stem
            original_name = "canonical_view"

            # Save color mask
            if self.color_masks:
                color_mask_path = output_dir / f"{original_name}_color.png"
                iio.imwrite(str(color_mask_path), self.color_masks[0])

            # Save index mask
            if self.index_masks:
                index_mask_path = output_dir / f"{original_name}_index.npy"
                np.save(str(index_mask_path), self.index_masks[0])

            # Clean up temporary directory
            temp_dir = Path("temp_image")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            return f"Saved masks to {output_dir}"

        except Exception as e:
            guru.error(f"Error saving mask: {e}")
            return f"Error saving mask: {e}"
            
    def initialize_sam_for_image(self) -> Tuple[str, np.ndarray]:
        """Initialize SAM model for single image"""
        try:
            if self.current_image is None:
                return "No image loaded. Please load an image first.", None
            
            # Initialize SAM using the temporary directory
            self.inference_state = self.sam_model.init_state(video_path=self.image_dir)
            # 명시적으로 초기화
            self.sam_model.reset_state(self.inference_state)
            return "SAM initialized. Click points to create mask.", self.current_image
            
        except Exception as e:
            guru.error(f"Failed to initialize SAM for image: {e}")
            return f"Error initializing SAM: {str(e)}", None
    
    def _initialize_model(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.sam_model = build_sam2_video_predictor(self.model_cfg, str(self.checkpoint_dir))

            guru.info(f"Loaded model from {self.checkpoint_dir}")
        except Exception as e:
            guru.error(f"Failed to load model: {e}")
            raise

    def clear_points(self) -> Tuple[None, None, str]:
        """Clear all selected points"""
        self.selected_points.clear()
        self.selected_labels.clear()
        return None, None, "Points cleared. Select new points to update mask."

    def add_new_mask(self) -> Tuple[None, str]:
        """Create a new mask"""
        # Check if tracking has started
        if self.inference_state and hasattr(self.inference_state, 'has_started_tracking') and self.inference_state.has_started_tracking:
            self.reset_state()
            message = "Tracking has been reset. Starting new mask."
        else:
            message = f"Creating new mask {self.current_mask_index + 1}"

        self.current_mask_index += 1
        self.clear_points()
        return None, message

    def _create_index_mask(self, masks: Dict[int, np.ndarray]) -> np.ndarray:
        """Create an index mask from multiple binary masks"""
        if not masks:
            return np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8)

        base_mask = np.zeros_like(next(iter(masks.values())), dtype=np.uint8) 
        for idx, mask in masks.items():
            base_mask[mask] = idx + 1 
        return base_mask

    def reset_state(self) -> str:
        """Reset all state variables for both video and image modes"""
        try:
            # Common state reset
            self.current_image = None
            self.current_mask_index = 0
            self.frame_index = 0
            self.selected_points.clear()
            self.selected_labels.clear()
            self.masks.clear()
            self.logits.clear()
            self.index_masks.clear()
            self.color_masks.clear()

            # Model state reset
            if self.inference_state:
                self.sam_model.reset_state(self.inference_state)
                self.inference_state = None

            # Mode-specific cleanup
            if self.mode == "image":
                # Clean up temporary image directory if it exists
                temp_dir = Path("temp_image")
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        guru.error(f"Error cleaning up temporary directory: {e}")
                
                # Reset image-specific variables
                self.image_dir = ""
                self.image_paths = []
                
                return "Image state reset successfully."
            else:  # video mode
                # Reset video-specific variables
                self.image_dir = ""
                self.image_paths = []
                
                # Clean up extracted frames if they exist
                frames_dir = Path("extracted_frames")
                if frames_dir.exists():
                    try:
                        shutil.rmtree(frames_dir)
                    except Exception as e:
                        guru.error(f"Error cleaning up frames directory: {e}")
                
                return "Video state reset successfully."

        except Exception as e:
            guru.error(f"Error during state reset: {e}")
            return f"Error resetting state: {str(e)}"

    def set_image_directory(self, directory: str) -> int:
        """Set the directory containing image frames"""
        # Removed self.reset_state()
        self.image_dir = Path(directory) 

        # Filter for image files
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir()
            if p.suffix.lower() in valid_extensions]
        ) 
        return len(self.image_paths)


    def set_frame(self, index: int) -> Optional[np.ndarray]:
        """Set the current frame to the given index"""
        if not 0 <= index < len(self.image_paths):
            return self.current_image

        self.clear_points()
        self.frame_index = index

        try:
            self.current_image = iio.imread(str(self.image_paths[index]))
            return self.current_image
        except Exception as e:
            guru.error(f"Failed to load image {self.image_paths[index]}: {e}")
            return None

    def initialize_sam(self) -> Tuple[str, Optional[np.ndarray]]:
        """Initialize SAM model with current video"""
        try:
            if not self.image_dir:
                return "No image directory set. Please extract frames first.", None

            self.inference_state = self.sam_model.init_state(video_path=str(self.image_dir))
            self.sam_model.reset_state(self.inference_state)
            return "SAM features extracted. Click points to update mask.", self.current_image
        except Exception as e:
            guru.error(f"Failed to initialize SAM: {str(e)}\n{traceback.format_exc()}")
            return f"Error initializing SAM: {e}", None

    def set_point_label(self, is_positive: bool) -> str:
        """Set the current point label"""
        self.current_label = 1.0 if is_positive else 0.0
        return f"Selecting {'positive' if is_positive else 'negative'} points"

    def _get_sam_mask(
        self,
        frame_idx: int,
        points: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Get mask from SAM model"""
        try:
            if self.inference_state is None:
                raise ValueError("Inference state is not initialized.")
            
            # Convert arrays to tensors and move to GPU if available
            if torch.cuda.is_available():
                points = torch.from_numpy(points).cuda()
                labels = torch.from_numpy(labels).cuda()
            
            with torch.no_grad():
                _, obj_ids, mask_logits = self.sam_model.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=self.current_mask_index, 
                    points=points,
                    labels=labels,
                ) 
                
                # Ensure mask_logits is on CPU for numpy operations
                mask_logits = mask_logits.cpu()

            return {
                obj_id: (mask_logits[i] > 0.0).squeeze().numpy()
                for i, obj_id in enumerate(obj_ids)
            }
        except Exception as e:
            guru.error(f"Error getting SAM mask: {e}")
            return {}

    def add_point(self, frame_idx: int, x: int, y: int) -> np.ndarray:
        """Add a point and get updated mask"""
        self.selected_points.append([x, y])
        self.selected_labels.append(self.current_label)

        # Get mask from SAM
        points = np.array(self.selected_points, dtype=np.float32)
        labels = np.array(self.selected_labels, dtype=np.int32)
        
        # obj_id : mask
        masks = self._get_sam_mask(0 if self.mode == "image" else frame_idx, points, labels)  # 이미지 모드에서는 항상 인덱스 0 사용
        mask = self._create_index_mask(masks) 
        
        # For single image mode, update the masks immediately
        if self.mode == "image":
            self.index_masks = [mask]
            color_mask = self._get_color_palette(mask.max() + 1)[mask]
            self.color_masks = [color_mask]
        
        return mask

    def run_tracking(self) -> Tuple[Optional[str], str]:
        
        start_time = time.time()
        try:
            if not self.image_paths:
                return None, "No images loaded. Please extract frames first."

            # Use custom save directory if specified, otherwise use default
            # output_dir = self.save_dir if self.save_dir else Path(self.image_dir).parent / "tracked_object"
            output_dir = self.save_dir if self.save_dir else Path(self.image_dir).parent
            # output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / "masked_video.mp4"
            temp_dir = None

            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                    self.sam_model = self.sam_model.cuda()

                # Load and preprocess images (keep as numpy arrays)
                images = []
                for p in self.image_paths: 
                    try:
                        img = iio.imread(str(p))
                        if img.ndim == 3:
                            img = img[:, :, :3]
                        # Keep as normalized numpy array
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                    except Exception as e:
                        return None, f"Error loading image {p}: {str(e)}"

                # Run tracking with autocast
                video_segments = {}
                with autocast():  # Use autocast from torch.cuda.amp
                    with torch.no_grad():
                        for frame_idx, obj_ids, mask_logits in self.sam_model.propagate_in_video(
                            self.inference_state,
                            start_frame_idx=0
                        ):
                            # Ensure mask_logits is on CPU for numpy operations
                            mask_logits = mask_logits.cpu()

                            masks = {
                                obj_id: (mask_logits[i] > 0.0).squeeze().numpy()
                                for i, obj_id in enumerate(obj_ids)
                            }
                            video_segments[frame_idx] = masks

                
                end_time = time.time()
                print(f"Tracking time: {end_time - start_time} seconds")

                # Process masks
                if not video_segments:
                    return None, "No masks were generated during tracking."

                self.index_masks = [
                    self._create_index_mask(v) for k, v in sorted(video_segments.items())
                ] 

                # Convert images back to uint8 range using numpy operations
                images = [np.clip(img * 255.0, 0, 255).astype(np.uint8) for img in images] 
                out_frames, self.color_masks = self._colorize_masks(images, self.index_masks) 
                
                # Create temporary directory for video creation
                temp_dir = Path(tempfile.mkdtemp())

                # Save frames as images
                for i, frame in enumerate(out_frames):
                    frame_path = temp_dir / f"{i:05d}.jpg"
                    # Ensure frame is in correct format using numpy operations
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                    iio.imwrite(str(frame_path), frame, quality=95)

                # Construct FFmpeg command to create the masked video
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", "10", # 10 fps
                    "-i", str(temp_dir / "%05d.jpg"),
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_path)
                ]

                # Execute FFmpeg
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )

                # Cleanup temporary frames
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)

                if not output_path.exists():
                    return None, "Error: Output video file was not created"

                # Return the path to the masked video for Gradio to display
                return str(output_path), "Tracking complete. Masked video saved."

            except subprocess.CalledProcessError as e:
                return None, f"FFmpeg error: {e.stderr}"

            except Exception as e:
                guru.error(f"Error during tracking process: {str(e)}\n{traceback.format_exc()}")
                return None, f"Error during tracking process: {str(e)}"

            finally:
                if temp_dir and temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        guru.error(f"Error cleaning up temporary directory: {str(e)}")

        except Exception as e:
            guru.error(f"Critical error in tracking: {str(e)}\n{traceback.format_exc()}")
            return None, f"Critical error in tracking: {str(e)}"

    def save_masks(self) -> str:
        """Save masks to output directory"""
        try:
            if not self.color_masks or not self.index_masks:
                return "No masks to save. Please run tracking first."

            # Use custom save directory if specified, otherwise use default
            output_dir = self.save_dir if self.save_dir else Path(self.image_dir).parent / "sam_result"
            output_dir.mkdir(exist_ok=True, parents=True)

            for img_path, color_mask, index_mask in zip(
                self.image_paths,
                self.color_masks,
                self.index_masks
            ):
                # Use the original image name for the mask files
                mask_name = img_path.name

                # Save color mask
                color_path = output_dir / mask_name
                iio.imwrite(str(color_path), color_mask)

                # Save index mask
                index_path = output_dir / f"{img_path.stem}.npy"
                np.save(str(index_path), index_mask)

            ########################### clip fusion ##########################
            print("[INFO] CLIP Fusion")
            gr.Info("CLIP Fusion is now running. Please wait...", duration=5)
            clip_fusion(self.args, str(Path(self.image_dir).parent))
            gr.Info("Successfully Save!", duration=5)
            print("[INFO] CLIP Fusion Completed")
            ########################### clip fusion ##########################
            
            

            return f"Saved masks to {output_dir}"

        except Exception as e:
            guru.error(f"Error saving masks: {e}")
            return f"Error saving masks: {e}"

    @staticmethod
    def _get_color_palette(
        n_colors: int,
        lightness: float = 0.5,
        saturation: float = 0.7
    ) -> np.ndarray:
        """Generate color palette for visualization"""
        hues = np.linspace(0, 1, int(n_colors))
        colors = [(0.0, 0.0, 0.0)]  # Start with black

        for h in hues[1:]:
            colors.append(colorsys.hls_to_rgb(h, lightness, saturation))

        return (255 * np.array(colors)).astype(np.uint8)

    def _colorize_masks(
        self,
        images: List[np.ndarray],
        index_masks: List[np.ndarray],
        alpha: float = 0.5
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        max_idx = max(mask.max() for mask in index_masks) 
        palette = self._get_color_palette(max_idx + 1) 

        color_masks = [] 
        output_frames = [] 

        for img, mask in zip(images, index_masks):
            color_mask = palette[mask.astype(int)] 
            color_masks.append(color_mask)

            # Blend image and mask
            blended = cv2.addWeighted(
                img.astype(np.uint8), alpha,
                color_mask.astype(np.uint8), 1 - alpha,
                0
            ) 
            output_frames.append(blended) 

        return output_frames, color_masks

import shutil

def create_gui(vs: VideoSegmentation, args):
    """Create Gradio interface"""
    def process_input(
        input_file = None,
        input_type: str = "video",
        start_time: float = 0,
        end_time: float = 0,
        fps: float = 30,
        height: int = 540
    ) -> Tuple[str, gr.update, Optional[np.ndarray], gr.update]:
        """Process input file (video or image)"""
        try:
            # if input_file is None:
            #     return "Please upload a file.", gr.update(), None, gr.update()
            
            # if input_file is None and input_type == "image":
            #     return "Please upload a file.", gr.update(), None, gr.update()
            
            # Set processing mode
            scene_name = vs.selected_scene_name
            vs.set_mode(input_type, scene_name)
            
            if input_type == "video":
                # Please complete the canonical-view image segmentation before using video mode
                if not os.path.exists(os.path.join(args.root_dir, scene_name, "sam_result", "canonical_view_index.npy")):
                    raise gr.Error(
                        "Please complete the canonical-view image segmentation before using video mode.",
                        duration=5
                    )
                output_dir = os.path.join(args.root_dir, scene_name, "sam_frames") 
                frame_files = sorted(os.listdir(output_dir))
                if not frame_files:
                    return "Failed to extract frames.", gr.update(), None, gr.update()
                
                num_frames = len(frame_files)
                vs.set_image_directory(str(output_dir)) 
                first_frame = vs.set_frame(0)
                
                # Show tracking controls, hide save single mask
                return (
                    f"Extracted {num_frames} frames.",
                    gr.update(
                        minimum=0,
                        maximum=max(0, num_frames - 1),
                        value=0,
                        step=1,
                        label="Frame Index",
                        visible=True
                    ),
                    first_frame,
                    gr.update(visible=True)  # Show tracking controls
                )
            
            else:  # image
                # Process single image
                
                scene_name = vs.selected_scene_name
                input_file = os.path.join(args.root_dir, scene_name, "canonical_view.png")
                image, msg = vs.process_single_image(input_file)
                if image is None: 
                    return msg, gr.update(visible=False), None, gr.update(visible=False)
                
                # Hide frame slider and tracking controls
                return (
                    msg,
                    gr.update(visible=False),
                    image,
                    gr.update(visible=False)  # Hide tracking controls
                )
                
        except Exception as e:
            guru.error(f"Error processing input: {e}")
            # return f"Error processing input: {str(e)}", gr.update(), None, gr.update()
            raise gr.Error(f"Error processing input: {str(e)}", duration=5)
        
    def handle_sam_init():
        if vs.mode == "image":
            return vs.initialize_sam_for_image()
        return vs.initialize_sam()

    def handle_save() -> str:
        if vs.mode == "image":
            return vs.save_single_mask()
        return vs.save_masks()
    
    def handle_point_label_pos() -> str:
        return vs.set_point_label(True)

    def handle_point_label_neg() -> str:
        return vs.set_point_label(False)
        
    def handle_point_selection(image, evt: gr.SelectData):
        if image is None:
            return gr.update(value=None)

        try:
            x, y = evt.index
            mask = vs.add_point(vs.frame_index, x, y)
            result = draw_points_on_image(image, vs.selected_points, vs.selected_labels, mask)
            return gr.update(value=result)
        except Exception as e:
            guru.error(f"Error handling point selection: {e}")
            return gr.update(value=image)

    def draw_points_on_image(image, points, labels, mask):
        """Draw points and mask on image"""
        if mask is None or len(points) == 0:
            return image

        try:
            palette = vs._get_color_palette(mask.max() + 1)
            color_mask = palette[mask.astype(int)]
            result = cv2.addWeighted(image.astype(np.uint8), 0.7, color_mask.astype(np.uint8), 0.3, 0)

            for point, label in zip(points, labels):
                x, y = int(point[0]), int(point[1])
                color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
                cv2.circle(result, (x, y), 5, color, -1)
                cv2.circle(result, (x, y), 6, (255, 255, 255), 1)

            return result
        except Exception as e:
            guru.error(f"Error drawing points: {e}")
            return image
        
        
    def update_scene_name(scene_name):
        vs.selected_scene_name = scene_name

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Video and Image Segmentation Tool")
        
        # Status message
        status_msg = gr.Textbox(
            value="Welcome! Start by uploading a video or image.",
            label="Status",
            interactive=False
        )
        
        
        with gr.Row():
            
            
            root_dir_textbox = gr.Text(value=args.root_dir, label="Root Directory", interactive=True)
            
            scene_names_dropdown = gr.Dropdown(choices=os.listdir(args.root_dir) + ["None"], label="Scene Name", interactive=True)

        # Input section
        with gr.Row():
            with gr.Column():
                input_type = gr.Radio(
                    choices=["video", "image"],
                    value="video",
                    label="Input Type"
                )
                input_file = gr.File(label="Input File", type="filepath")
                
                # Video specific controls
                with gr.Group(visible=True) as video_controls:
                    start_time = gr.Number(0, label="Start Time (s)")
                    end_time = gr.Number(0, label="End Time (s)")
                    fps = gr.Number(value=30, label="FPS")
                    height = gr.Number(value=540, label="Height")
                
                process_btn = gr.Button("Process Input")

            with gr.Column():
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=0,
                    value=0,
                    step=1,
                    label="Frame Index",
                    interactive=True,
                    visible=True
                )
                current_frame = gr.Image(label="Current Frame/Image", interactive=True)
                init_sam_btn = gr.Button("Initialize SAM")
                reset_btn = gr.Button("Reset")

                # Point selection
                with gr.Row():
                    pos_btn = gr.Button("Positive Point")
                    neg_btn = gr.Button("Negative Point")
                clear_btn = gr.Button("Clear Points")

            with gr.Column():
                output_display = gr.Image(label="Current Selection")
                new_mask_btn = gr.Button("New Mask")
                
                # Video specific outputs
                with gr.Group(visible=True) as tracking_outputs:
                    track_btn = gr.Button("Track Object")
                    output_video = gr.Video(label="Result")
                
                # Save controls
                # Removed save_dir and dir_button
                save_btn = gr.Button("Save Result")

        # Update visibility of video controls
        def update_controls(input_type):
            return {
                video_controls: gr.update(visible=input_type == "video"),
                tracking_outputs: gr.update(visible=input_type == "video")
            }
            
        

        # Connect components
        input_type.change(
            update_controls,
            inputs=[input_type],
            outputs=[video_controls, tracking_outputs]
        )

        process_btn.click(
            process_input,
            inputs=[
                input_file,
                input_type,
                start_time,
                end_time,
                fps,
                height
            ],
            outputs=[
                status_msg,
                frame_slider,
                current_frame,
                tracking_outputs
            ]
        )

        frame_slider.change(
            vs.set_frame,
            inputs=[frame_slider],
            outputs=[current_frame]
        )

        current_frame.select(
            handle_point_selection,
            inputs=[current_frame],
            outputs=[output_display] 
        )

        init_sam_btn.click(
            handle_sam_init,
            outputs=[status_msg, current_frame]
        )

        reset_btn.click(
            vs.reset_state,
            outputs=[status_msg]
        )

        pos_btn.click(
            handle_point_label_pos,
            outputs=[status_msg]
        )

        neg_btn.click(
            handle_point_label_neg,
            outputs=[status_msg]
        )

        clear_btn.click(
            vs.clear_points,
            outputs=[output_display, output_video, status_msg]
        )

        new_mask_btn.click(
            vs.add_new_mask,
            outputs=[output_display, status_msg]
        )

        track_btn.click(
            vs.run_tracking,
            outputs=[output_video, status_msg]
        )

        # Removed Directory selection and adjusted Save button
        save_btn.click(
            handle_save,
            outputs=[status_msg]
        )
        
        scene_names_dropdown.change(
            update_scene_name,
            inputs=[scene_names_dropdown]
        )

    return demo

def main():
    """Main entry point"""
    
    args = get_args()

    # Setup GPU if available
    setup_gpu()

    try:
        # Initialize video segmentation with save directory
        vs = VideoSegmentation(
            args=args,
            checkpoint_dir=args.checkpoint,
            model_cfg=args.config,
            save_dir=args.mask_save_dir
        )

        # Create and launch GUI
        demo = create_gui(vs, args)
        demo.queue()
        demo.launch(
            server_port=args.port,
            share=args.share,
            server_name="0.0.0.0"
        )

    except Exception as e:
        guru.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()