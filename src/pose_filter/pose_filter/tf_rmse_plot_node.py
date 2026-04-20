"""Capture map->wrist_strap_ekf and map->wrist_strap_truth via TF, overlay them in 3D, report RMSE."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return R.from_quat([x, y, z, w]).as_matrix()


def _rpy_deg(rot: np.ndarray) -> np.ndarray:
    return R.from_matrix(rot).as_euler("xyz", degrees=True)


class TfRmsePlotNode(Node):
    def __init__(self) -> None:
        super().__init__("tf_rmse_plot_node")

        self.declare_parameter("parent_frame", "map")
        self.declare_parameter("ekf_frame", "wrist_strap_ekf")
        self.declare_parameter("truth_frame", "wrist_strap_truth")
        self.declare_parameter("sample_rate_hz", 30.0)
        self.declare_parameter("plot_rate_hz", 10.0)
        self.declare_parameter("max_points", 3000)
        self.declare_parameter("frame_axis_len", 0.06)
        self.declare_parameter("trail_axis_len", 0.025)
        self.declare_parameter("orientation_trail_stride", 25)
        self.declare_parameter("show_orientation_trail", True)
        self.declare_parameter("save_figure", False)
        self.declare_parameter("save_path", "/tmp/tf_rmse_plot.png")
        self.declare_parameter("save_dpi", 300)
        self.declare_parameter("save_every_n_updates", 10)
        self.declare_parameter("save_on_shutdown", True)

        self._parent_frame = str(self.get_parameter("parent_frame").value)
        self._ekf_frame = str(self.get_parameter("ekf_frame").value)
        self._truth_frame = str(self.get_parameter("truth_frame").value)
        self._sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self._plot_rate_hz = float(self.get_parameter("plot_rate_hz").value)
        self._max_points = int(self.get_parameter("max_points").value)
        self._axis_len = float(self.get_parameter("frame_axis_len").value)
        self._trail_axis_len = float(self.get_parameter("trail_axis_len").value)
        self._trail_stride = max(1, int(self.get_parameter("orientation_trail_stride").value))
        self._show_trail = bool(self.get_parameter("show_orientation_trail").value)
        self._save_figure = bool(self.get_parameter("save_figure").value)
        self._save_path = Path(str(self.get_parameter("save_path").value)).expanduser()
        self._save_dpi = int(self.get_parameter("save_dpi").value)
        self._save_every_n_updates = max(1, int(self.get_parameter("save_every_n_updates").value))
        self._save_on_shutdown = bool(self.get_parameter("save_on_shutdown").value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._ekf_xyz: deque[np.ndarray] = deque(maxlen=self._max_points)
        self._ekf_R: deque[np.ndarray] = deque(maxlen=self._max_points)
        self._truth_xyz: deque[np.ndarray] = deque(maxlen=self._max_points)
        self._truth_R: deque[np.ndarray] = deque(maxlen=self._max_points)

        self._paired_err_xyz: deque[np.ndarray] = deque(maxlen=self._max_points)
        self._paired_err_rot_deg: deque[float] = deque(maxlen=self._max_points)

        self._warned_lookup_ekf = False
        self._warned_lookup_truth = False
        self._plot_updates = 0

        sample_period = 1.0 / max(self._sample_rate_hz, 0.1)
        plot_period = 1.0 / max(self._plot_rate_hz, 0.1)
        self._sample_timer = self.create_timer(sample_period, self._sample_tf)
        self._plot_timer = self.create_timer(plot_period, self._update_plot)

        self._fig = plt.figure("TF Trajectories: EKF vs Truth")
        self._ax = self._fig.add_subplot(111, projection="3d")
        plt.ion()
        self._fig.show()

        self.get_logger().info(
            "Listening for %s->%s and %s->%s; sample %.1f Hz, plot %.1f Hz"
            % (
                self._parent_frame, self._ekf_frame,
                self._parent_frame, self._truth_frame,
                self._sample_rate_hz, self._plot_rate_hz,
            )
        )

    def _lookup(self, child: str, warn_attr: str) -> Optional[tuple]:
        try:
            tf = self._tf_buffer.lookup_transform(self._parent_frame, child, Time())
            setattr(self, warn_attr, False)
            t = tf.transform.translation
            r = tf.transform.rotation
            xyz = np.array([t.x, t.y, t.z], dtype=np.float64)
            rot = _quat_to_matrix(r.x, r.y, r.z, r.w)
            return xyz, rot
        except Exception as exc:
            if not getattr(self, warn_attr):
                setattr(self, warn_attr, True)
                self.get_logger().warning(
                    "TF lookup %s->%s failed: %s" % (self._parent_frame, child, exc)
                )
            return None

    def _sample_tf(self) -> None:
        ekf = self._lookup(self._ekf_frame, "_warned_lookup_ekf")
        truth = self._lookup(self._truth_frame, "_warned_lookup_truth")

        if ekf is not None:
            self._ekf_xyz.append(ekf[0])
            self._ekf_R.append(ekf[1])
        if truth is not None:
            self._truth_xyz.append(truth[0])
            self._truth_R.append(truth[1])

        # Pairwise error only when we have a fresh sample for both at this tick.
        if ekf is not None and truth is not None:
            err_xyz = ekf[0] - truth[0]
            self._paired_err_xyz.append(err_xyz)
            R_err = ekf[1] @ truth[1].T
            angle = float(np.linalg.norm(R.from_matrix(R_err).as_rotvec(degrees=True)))
            self._paired_err_rot_deg.append(angle)

    def _draw_axes(
        self,
        origin: np.ndarray,
        rot: np.ndarray,
        axis_len: float,
        colors: list,
        alpha: float,
        linewidth: float = 2.0,
    ) -> None:
        for i in range(3):
            d = rot[:, i] * axis_len
            self._ax.quiver(
                origin[0], origin[1], origin[2],
                d[0], d[1], d[2],
                color=colors[i], linewidth=linewidth, alpha=alpha,
            )

    def _draw_trail(
        self,
        xyz: np.ndarray,
        rotations: list,
        colors: list,
        alpha: float,
    ) -> None:
        n = len(rotations)
        if n < 2 or xyz.shape[0] < 2:
            return
        for i in range(0, n, self._trail_stride):
            if i >= xyz.shape[0]:
                break
            self._draw_axes(
                xyz[i], rotations[i], self._trail_axis_len, colors, alpha, linewidth=1.0
            )

    def _compute_rmse_text(self) -> str:
        lines = []
        if len(self._paired_err_xyz) > 0:
            err = np.array(self._paired_err_xyz)
            rmse_xyz = np.sqrt(np.mean(err ** 2, axis=0))
            rmse_total = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
            lines.append("Position RMSE [m]")
            lines.append(
                "  x=%.4f  y=%.4f  z=%.4f  total=%.4f"
                % (rmse_xyz[0], rmse_xyz[1], rmse_xyz[2], rmse_total)
            )
        if len(self._paired_err_rot_deg) > 0:
            rot_arr = np.array(self._paired_err_rot_deg)
            rmse_rot = float(np.sqrt(np.mean(rot_arr ** 2)))
            lines.append("Rotation RMSE [deg]: %.3f" % rmse_rot)

        if len(self._ekf_R) > 0:
            rpy = _rpy_deg(self._ekf_R[-1])
            lines.append("EKF rpy: r=%.1f p=%.1f y=%.1f" % (rpy[0], rpy[1], rpy[2]))
        if len(self._truth_R) > 0:
            rpy = _rpy_deg(self._truth_R[-1])
            lines.append("GT  rpy: r=%.1f p=%.1f y=%.1f" % (rpy[0], rpy[1], rpy[2]))

        lines.append("Samples: ekf=%d truth=%d paired=%d"
                     % (len(self._ekf_xyz), len(self._truth_xyz), len(self._paired_err_xyz)))
        return "\n".join(lines)

    def _update_plot(self) -> None:
        if len(self._ekf_xyz) == 0 and len(self._truth_xyz) == 0:
            return

        self._ax.clear()
        self._ax.set_title("EKF vs Truth (frame: %s)" % self._parent_frame)
        self._ax.set_xlabel("X [m]")
        self._ax.set_ylabel("Y [m]")
        self._ax.set_zlabel("Z [m]")

        ekf_arr = np.array(self._ekf_xyz) if self._ekf_xyz else None
        truth_arr = np.array(self._truth_xyz) if self._truth_xyz else None

        truth_axis_colors = ["#08519c", "#238b45", "#6a51a3"]
        ekf_axis_colors = ["#cb181d", "#fd8d3c", "#67000d"]

        if truth_arr is not None:
            self._ax.plot(truth_arr[:, 0], truth_arr[:, 1], truth_arr[:, 2],
                          "b-", label="Truth")
            self._ax.scatter(truth_arr[-1, 0], truth_arr[-1, 1], truth_arr[-1, 2],
                             c="b", s=35)
        if ekf_arr is not None:
            self._ax.plot(ekf_arr[:, 0], ekf_arr[:, 1], ekf_arr[:, 2],
                          "r-", label="EKF")
            self._ax.scatter(ekf_arr[-1, 0], ekf_arr[-1, 1], ekf_arr[-1, 2],
                             c="r", s=35)

        bounds = []
        if ekf_arr is not None:
            bounds.append(ekf_arr)
        if truth_arr is not None:
            bounds.append(truth_arr)
        pts = np.vstack(bounds)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.6, 0.2)
        self._ax.set_xlim(center[0] - radius, center[0] + radius)
        self._ax.set_ylim(center[1] - radius, center[1] + radius)
        self._ax.set_zlim(center[2] - radius, center[2] + radius)

        if self._show_trail:
            if truth_arr is not None:
                self._draw_trail(truth_arr, list(self._truth_R), truth_axis_colors, 0.35)
            if ekf_arr is not None:
                self._draw_trail(ekf_arr, list(self._ekf_R), ekf_axis_colors, 0.22)

        if truth_arr is not None and len(self._truth_R):
            self._draw_axes(truth_arr[-1], self._truth_R[-1], self._axis_len,
                            truth_axis_colors, 0.95, linewidth=2.4)
        if ekf_arr is not None and len(self._ekf_R):
            self._draw_axes(ekf_arr[-1], self._ekf_R[-1], self._axis_len,
                            ekf_axis_colors, 0.9, linewidth=2.4)

        self._fig.text(
            0.02, 0.98, self._compute_rmse_text(), transform=self._fig.transFigure,
            fontsize=9, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )

        self._ax.legend(loc="upper right")
        self._ax.grid(True)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

        self._plot_updates += 1
        if self._save_figure and (self._plot_updates % self._save_every_n_updates == 0):
            self._save_current_figure()

    def _save_current_figure(self) -> None:
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(str(self._save_path), dpi=self._save_dpi, bbox_inches="tight")

    def destroy_node(self) -> bool:
        if self._save_figure and self._save_on_shutdown:
            try:
                self._save_current_figure()
                self.get_logger().info("Saved figure to %s" % self._save_path)
            except Exception as exc:
                self.get_logger().warning("Failed to save figure on shutdown: %s" % exc)
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = TfRmsePlotNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
