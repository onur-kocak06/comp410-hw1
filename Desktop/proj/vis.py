from vedo import Mesh, Plotter, Video
import numpy as np

def render_rotating_video(obj_path, output_video="rotation.mp4", duration=5, fps=30):
    obj = Mesh(obj_path)
    obj.c('w')  # ensure white edges if needed

    n_frames = duration * fps
    video = Video(name=output_video, duration=duration, fps=fps)

    plotter = Plotter(offscreen=True, bg='white', size=(800, 800))
    plotter.show(obj, resetcam=True)

    for i in range(n_frames):
        obj.rotateY(360 / n_frames)
        plotter.show(obj, interactive=False)
        video.addFrame(plotter.screenshot(asarray=True))

    video.close()
    plotter.close()

# Usage
render_rotating_video("outputs/colored_ulker.obj")
