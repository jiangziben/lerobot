import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class RealTimeForceVisualizer:
    def __init__(self, arrow_scale=1.0):
        self.force = np.array([0.0, 0.0, 0.0])
        self.arrow_scale = arrow_scale

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 初始三个箭头：分别是 x, y, z 方向分量
        self.quiver_fx = self.ax.quiver(0, 0, 0, 0, 0, 0, color='red', label='Fx')
        self.quiver_fy = self.ax.quiver(0, 0, 0, 0, 0, 0, color='green', label='Fy')
        self.quiver_fz = self.ax.quiver(0, 0, 0, 0, 0, 0, color='blue', label='Fz')

        # 坐标轴设置
        self.ax.set_xlim([-500, 500])
        self.ax.set_ylim([-500, 500])
        self.ax.set_zlim([-500, 500])
        self.ax.set_xlabel('Fx')
        self.ax.set_ylabel('Fy')
        self.ax.set_zlabel('Fz')
        self.ax.set_title("Real-Time Force Vector Components")
        self.ax.legend()

        self.ani = FuncAnimation(self.fig, self._update_plot, interval=100, blit=False)

    def update_force(self, force_xyz):
        """外部调用：更新力的三个分量"""
        self.force = np.array(force_xyz)

    def _update_plot(self, frame):
        # 清除旧箭头
        self.quiver_fx.remove()
        self.quiver_fy.remove()
        self.quiver_fz.remove()

        fx, fy, fz = self.force
        scale = self.arrow_scale

        # 单独绘制每个方向的分量
        self.quiver_fx = self.ax.quiver(0, 0, 0, fx, 0, 0, color='red', length=scale * abs(fx), normalize=False)
        self.quiver_fy = self.ax.quiver(0, 0, 0, 0, fy, 0, color='green', length=scale * abs(fy), normalize=False)
        self.quiver_fz = self.ax.quiver(0, 0, 0, 0, 0, fz, color='blue', length=scale * abs(fz), normalize=False)

    def show(self):
        """启动可视化窗口"""
        plt.pause(0.01)

if __name__ == "__main__":
    import time

    vis = RealTimeForceVisualizer()

    # 模拟实时数据输入（例如来自传感器 / 仿真）
    t = 0
    while True:
        fx = np.sin(t)
        fy = np.cos(t)
        fz = 0.2 * np.sin(2 * t)
        vis.update_force([fx, fy, fz])  # 只需调用这个即可更新力
        t += 0.1
        vis.show()
        time.sleep(0.05)
        
