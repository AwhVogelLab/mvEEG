import numpy as np
import matplotlib.pyplot as plt
from .dataloader import DataLoader
import warnings
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.manifold import MDS as sklearn_MDS


class MDS:
    def __init__(
        self,
        root_dir: str,
        experiment_name: str,
        labels: str,
        subs: list = [],
        description: str = "RSA",
        n_components=2,
        stress_thresh=0.1,
        stress_behavior: str = "warn",
    ):
        """
        Class to calculate MDS projections from a RDM and visualize them
        inputs:

        root_dir: root BIDS directory
        experiment_name: name of the experiment
        labels: list of condition labels
        subs: list of subjects to include
        description: description keyword for data
        n_components: how many MDS dimensions to include by default. Usually 2 or 3. Most functions assume 2
        stress_thresh: at what threshold is is the stress function problematic
        stress_behavior: "warn" or "raise" - whether to raise a warning or error if stress exceeds this
        """

        dataset = DataLoader(
            root_dir=root_dir, data_type="rdms", experiment_name=experiment_name, descriptions=[description], subs=subs
        )

        self.rdms, self.t = dataset.get_data(dset=description, keys=["RDM", "times"])
        self.rdms = self.rdms.mean(1)
        self.rdms = np.moveaxis(self.rdms, 1, 3)  # move to subs x conds x conds x times
        self.nsub = self.rdms.shape[0]

        self.mds = sklearn_MDS(
            dissimilarity="precomputed", random_state=0, n_components=n_components, normalized_stress=False
        )  # instance transformer
        self.labels = labels
        self.stress_thresh = stress_thresh
        self.stress_behavior = stress_behavior
        self.stress_log = []  # log of stress values

    def check_stress(self):
        """
        Helper function that checks if the projection stress is above our threshold
        """
        if self.mds.stress_ > self.stress_thresh:
            if self.stress_behavior == "warn":
                warnings.warn(
                    f"Warning: stress for MDS projection {self.mds.stress_} is above threshold {self.stress_thresh}",
                    RuntimeWarning,
                )
            elif self.stress_behavior == "raise":
                raise RuntimeError(
                    f"Stress for MDS projection {self.mds.stress_} is above threshold {self.stress_thresh}"
                )

    def _calculate_MDS(self, t_start=500, t_stop=1500, isub=None, n_components=2):
        """
        Helper function to calculate MDS projections in a certain range.
        Arguments:
        t_start,t_stop: define window to average over

        """
        self.mds.set_params(n_components=n_components)
        if isub is None:
            tsub_rdm = self.rdms[..., np.logical_and(self.t >= t_start, self.t <= t_stop)].mean(
                (0, 3)
            )  # average over subjects and times
        else:
            tsub_rdm = self.rdms[isub, :, :, np.logical_and(self.t >= t_start, self.t <= t_stop)].mean(0)
        transform = self.mds.fit_transform(tsub_rdm)  # apply MDS scaling
        self.check_stress()
        self.stress_log.append(self.mds.stress_)  # helpful for debugging

        if transform.shape[-1] == 2:  # if 2D return both dimensions separately
            return transform[:, 0], transform[:, 1]
        elif transform.shape[-1] == 3:  # if 3D return x,y,z
            return transform[:, 0], transform[:, 1], transform[:, 2]
        else:  # otherwise return a tuple
            return transform

    def plot_MDS_2D(
        self,
        ax=None,
        t_start=200,
        t_stop=1800,
        title=None,
        xlim=None,
        ylim=None,
        hide_axes: bool = True,
        circwidth: int = 300,
        isub=None,
        colors=None,
        **kwargs,
    ):
        """
        Displays MDS projection, and labels each condition
        Arguments:
        ax: axis to plot on
        t_start,t_stop: times to average over (passed to calculate_MDS)
        title: plot title
        xlim,ylim: axis limits. always specifiy manually if you want to compare multiple graphs
        hide_axes: bool, should axes be shown?
        circwidth: width of circles. Change this if the circles at each point overlap your condition labels
        colors: can give a list of colors for each circle (if unset defaults to all black)
        kwargs: passed to ax.annotate
        """
        if ax is None:
            _, ax = plt.subplots()
        if colors is None:
            colors = ["black" for _ in range(len(self.labels))]
        x, y = self._calculate_MDS(t_start, t_stop, isub=isub, n_components=2)
        ax.scatter(x, y, facecolors="none", edgecolors=colors, s=circwidth)  # draws circles centered at points
        for i, label in enumerate(self.labels):
            # labels points with condition labels
            ax.annotate(label, (x[i], y[i]), ha="center", va="center", c=colors[i], **kwargs)

        ax.set_title(title)

        if hide_axes:
            ax.tick_params(
                left=False, right=False, labelleft=False, labelbottom=False, bottom=False
            )  # no axis labels or ticks

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    def _animation_wrapper(self, itime):
        """
        helper function for the animator
        do not manually call this
        """
        self.ani_ax.clear()
        try:  # plot projection averaged across [itime,itime+1]
            self.plot_MDS_2D(
                ax=self.ani_ax,
                t_start=self.ani_times[itime],
                t_stop=self.ani_times[itime + 1],
                title=f"{self.ani_times[itime]}<t<{self.ani_times[itime+1]}",
                xlim=self.ani_xlim,
                ylim=self.ani_ylim,
                **self.mds_args,
            )

        except ValueError as e:
            raise RuntimeError(f"i={itime},tstart={self.ani_times[itime]},tstop={self.ani_times[itime]}") from e

    def animate_MDS(
        self,
        t_start,
        t_stop,
        t_step,
        filename="./animation.gif",
        fps=1,
        xlim=(-0.005, 0.005),
        ylim=(-0.005, 0.005),
        **kwargs,
    ):
        """
        Animates a MDS projection over time as a gif

        Arguments:
        t_start,t_stop: absolute minimum and maximum times
        t_step: interval between steps (reasonable is usually 50-250 ms)
        filename: filename to save as
        fps: adjust this to control speed
        xlim,ylim: axis limits

        """
        fig, self.ani_ax = plt.subplots()
        self.ani_xlim = xlim
        self.ani_ylim = ylim

        # set up times to iterate over
        self.ani_times = np.arange(t_start, t_stop + t_step, t_step)
        self.mds_args = kwargs  # args doesn't work well, workaround
        ani = FuncAnimation(
            fig, self._animation_wrapper, frames=len(self.ani_times) - 2, interval=500, repeat=False
        )  # instance matplotlib animator

        ani.save(filename, dpi=300, writer=PillowWriter(fps=fps))  # output to file
        plt.close()
        print(f"Saved as {filename}")

    def plot_MDS_3D(
        self,
        ax=None,
        t_start=200,
        t_stop=1800,
        title=None,
        xlim=None,
        ylim=None,
        zlim=None,
        hide_axes: bool = True,
        isub=None,
        colors=None,
        point_size=10,
        view=(20, 65),
        text_kwargs=None,
        connect_pairs=None,
        connect_line_color=None,
    ):
        """
        Displays MDS projection, and labels each condition
        Arguments:
        ax: axis to plot on
        t_start,t_stop: times to average over (passed to calculate_MDS)
        title: plot title
        xlim,ylim: axis limits. always specifiy manually if you want to compare multiple graphs
        hide_axes: bool, should axes be shown?
        circwidth: width of circles. Change this if the circles at each point overlap your condition labels
        colors: can give a list of colors for each circle (if unset defaults to all black)
        kwargs: passed to ax.annotate
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        if text_kwargs is None:
            text_kwargs = {}

        ax.view_init(*view)

        if colors is None:
            colors = ["C0" for _ in range(len(self.labels))]
        x, y, z = self._calculate_MDS(t_start, t_stop, isub=isub, n_components=3)
        ax.scatter(x, y, z, alpha=1, facecolors=colors, s=point_size)  # draws circles centered at points

        for i, label in enumerate(self.labels):
            # labels points with condition labels
            ax.text(x[i], y[i], z[i], label, ha="center", va="center", zorder=999, **text_kwargs)
        ax.set_title(title)

        if connect_pairs is not None:
            # draw lines connecting all pairs
            if connect_line_color is None:
                connect_line_color = "k"
            get_coords = lambda lab: np.array(
                (x[self.labels.index(lab)], y[self.labels.index(lab)], z[self.labels.index(lab)])
            )
            for lab1, lab2 in connect_pairs:
                xs, ys, zs = [(c1, c2) for c1, c2 in zip(get_coords(lab1), get_coords(lab2))]
                ax.plot(xs, ys, zs, color=connect_line_color, linestyle="dashed", linewidth=2)

        if hide_axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
