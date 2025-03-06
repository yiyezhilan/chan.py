import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from Chan import CChan
from .PlotDriver import CPlotDriver
import io
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime, timedelta


def figure_to_array(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    img.close()   # 关闭 PIL Image
    buf.close()   # 关闭 BytesIO 缓冲区
    return img_array


class CAnimateDriver:
    def __init__(
        self,
        chan: CChan,
        plot_config=None,
        plot_para=None,
        output_filename="output.mp4",
        fps=30,
        dpi=300,
    ):
        if plot_config is None:
            plot_config = {}
        if plot_para is None:
            plot_para = {}

        start_date = datetime.strptime(chan.begin_time, "%Y-%m-%d").date()
        if chan.end_time is None:
            end_time = datetime.now().date() - timedelta(days=1)
        else:
            end_time = datetime.strptime(chan.end_time, "%Y-%m-%d").date()
        total_days = (end_time - start_date).days * 0.65

        imgs = []
        for _ in tqdm(chan.step_load(), total=total_days, desc="Generating Images"):
            g = CPlotDriver(chan, plot_config, plot_para)
            # 将 Figure 转换为图像数组，并保存到列表中
            imgs.append(figure_to_array(g.figure, dpi=dpi))
            plt.close("all")

        # 创建一个新 Figure 用于写入视频帧
        fig, ax = plt.subplots()
        writer = FFMpegWriter(
            fps=fps, metadata=dict(artist="Your Name", title="Animation")
        )
        with writer.saving(fig, output_filename, dpi=dpi):
            for img in tqdm(imgs, desc="Writing Video"):
                ax.clear()  # 清空上一次的内容
                ax.imshow(img)  # 显示当前图像数组
                ax.axis("off")  # 再次确保没有坐标轴（防止干扰）
                writer.grab_frame()  # 捕获当前帧
