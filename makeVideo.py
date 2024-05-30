from moviepy.editor import ImageSequenceClip, VideoFileClip
import glob

image_folder = 'viz/2024-05-30-17-58-57-870565/heatmap' 
# image_folder = 'viz/2024-05-30-17-59-43-324699/heatmap' 
video_name = 'pred_result.mp4'

images = sorted(glob.glob(f"{image_folder}/*.png"))  

clip = ImageSequenceClip(images, fps=10)

clip.write_videofile(video_name)
