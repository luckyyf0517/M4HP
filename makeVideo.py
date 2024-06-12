from moviepy.editor import ImageSequenceClip, VideoFileClip
import glob

# image_folder = './data/HuPR/2024-05-30-17-59-43-324699/camera' 
# image_folder = './viz/2024-05-30-17-59-43-324699/pointcloud_proj' 
image_folder = './viz/run_hupr_action_only_no_wbce_no_pe/seq_0411/pred_result' 
video_name = 'pred_result.mp4'
images = sorted(glob.glob(f"{image_folder}/*.png"))  
clip = ImageSequenceClip(images, fps=10)
clip.write_videofile(video_name)
