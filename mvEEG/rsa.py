from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sista
import seaborn as sns


from .dataloader import DataLoader
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings


class RSA:
    def __init__(
        self,
        root_dir: str,
        experiment_name: str,
        condition_details: dict,
        delay_period_start: int = 200,
        extra_theoretical_models: dict = None,
        per_subject_factors: dict = None,
        descriptions: list = ["RSA"],
        subs: list = [],
    ):
        """
        Class to perform representational similarity analysis (RSA) on a dataset.

        Arguments:
        condition_details: dict of dicts with all conditions with relevant values for calculating RDMs.
        Eg: {'condition1': {'factor1': X, 'factor2': Y}}

        delay_period_start (int): time when the delay starts, default 200
        extra_theoretical_models (dict): additional theoretical models to compare to. Must be a dict of RDMs
        per_subject_factors (dict): factors that vary per subject. Give as a dict of {'model':ndarray(subjects x conditions)}

        """

        self.labels = list(condition_details.keys())

        self.ridx, self.cidx = np.triu_indices(len(self.labels), k=1)
        self.delay_period_start = delay_period_start

        ######### SET UP RDMS #########

        # Populate theoretical_models
        self.theoretical_models = defaultdict(lambda: np.zeros((len(self.labels), len(self.labels))))

        factors = np.unique([list(condition_details[key].keys()) for key in condition_details.keys()]).tolist()

        for factor in factors:  # compute RDMs
            for i, condi in enumerate(condition_details):
                for j, condj in enumerate(condition_details):
                    self.theoretical_models[factor][i, j] = np.abs(
                        condition_details[condi][factor] - condition_details[condj][factor]
                    )

        if extra_theoretical_models is not None:
            self.theoretical_models.update(extra_theoretical_models)

        # Per subject models (eg accuracy, pupil size)
        if per_subject_factors is not None:
            ncond = len(self.labels)
            self.per_subject_models = defaultdict(
                lambda: np.full((per_subject_factors.shape[0], ncond, ncond), np.nan)
            )  # subs x conds x conds
            for factor in per_subject_factors.keys():
                for i in range(ncond):
                    for j in range(ncond):
                        self.per_subject_models[factor][:, i, j] = np.abs(
                            per_subject_factors[factor][:, i] - per_subject_factors[factor][:, j]
                        )

        else:
            self.per_subject_models = None

        ######### LOAD DATA #########

        self.dataset = DataLoader(
            root_dir=root_dir, data_type="rdms", experiment_name=experiment_name, descriptions=descriptions, subs=subs
        )

        self.rdms, self.t = self.dataset.get_data(dset="RSA", keys=["RDM", "times"])
        self.rdms = self.rdms.mean(1)
        self.rdms = np.moveaxis(self.rdms, 1, 3)  # move to subs x conds x conds x times
        self.nsub = self.rdms.shape[0]
        self.delay_period_end = self.t[-1]

        if self.per_subject_models is not None:

            self.color_palette = {
                factor: sns.color_palette()[i]
                for i, factor in enumerate(list(self.theoretical_models.keys()) + list(self.per_subject_models.keys()))
            }
        else:
            self.color_palette = {
                factor: sns.color_palette()[i] for i, factor in enumerate(self.theoretical_models.keys())
            }

    ##################################
    # CALCULATE FITS
    ##################################

    def calculate_VIF(self, factor_df):
        """
        helper function to calculate VIF
        inputs: dataframe with unranked values
        """
        ranked_vals = sista.rankdata(factor_df, axis=0)

        desmat_with_intercept = pd.DataFrame(ranked_vals)
        desmat_with_intercept["intercept"] = 1
        vif_data = pd.DataFrame()
        vif_data["regressor"] = desmat_with_intercept.columns.drop("intercept")
        vif_data["VIF"] = [
            variance_inflation_factor(desmat_with_intercept.values, i)
            for i in range(len(desmat_with_intercept.columns))
            if desmat_with_intercept.columns[i] != "intercept"
        ]
        vif_data["regressor"] = factor_df.columns.tolist()

        return vif_data

    def fit_theoretical_models(self, models=None, ret_VIF=False, rank=True):
        """
        Applies a linear regression fit of specified theoretical models
        Arguments:
        models: list of models (found in self.theoretical_models) to run
        ret_VIF: returns a list of VIFs per condition
        rank: rank RDMs (empirical and theoretical) prior to running linear regression
        """
        if models is None:  # if unset use all available options
            models = list(self.theoretical_models.keys())

        self.r2 = np.full((self.nsub, len(self.t)), np.nan)

        self.factor_df = pd.DataFrame(
            np.transpose([self.theoretical_models[key][self.ridx, self.cidx] for key in models]), columns=models
        )  # convert to 1D dataframe
        self.factor_df["Intercept"] = 1

        # rank factors by relative dissimilarity

        if ret_VIF:  # calculate and return VIFs
            vif_data = self.calculate_VIF(self.factor_df)
            print(vif_data)
        partial_r_df = pd.DataFrame()

        for isub in range(self.nsub):

            if self.per_subject_models is None:
                factor_df = pd.DataFrame(
                    np.transpose([self.theoretical_models[key][self.ridx, self.cidx] for key in models]), columns=models
                )  # convert to 1D dataframe
            else:
                skip = False
                for model in self.per_subject_models.keys():
                    if np.isnan(self.per_subject_models[model][isub]).any():
                        warnings.warn(
                            f"Subject {isub} has NaNs in {model} model. Skipping this subject", RuntimeWarning
                        )
                        skip = True
                if skip:
                    continue  # skip this subject if we have NaNs in any of the RDMs

                factor_df = pd.DataFrame(
                    np.transpose(
                        [self.theoretical_models[key][self.ridx, self.cidx] for key in models]
                        + [
                            self.per_subject_models[model][isub, self.ridx, self.cidx]
                            for model in self.per_subject_models.keys()
                        ]
                    ),
                    columns=models + list(self.per_subject_models.keys()),
                )  # combination of RDMs and per subject models
            factor_df["Intercept"] = 1

            if rank:
                # Rank the RDMs across each time point by row

                ranked_vals = sista.rankdata(factor_df, axis=0)
                ranked_dists = sista.rankdata(self.rdms[isub, self.ridx, self.cidx, :], axis=0)
            else:
                ranked_vals = factor_df.to_numpy()
                ranked_dists = self.rdms[isub, self.ridx, self.cidx, :]

            r_scores = defaultdict(lambda: np.zeros((ranked_dists.shape[1])))

            for t in range(ranked_dists.shape[1]):
                curr_dists = ranked_dists[:, t]

                fitted_lm = LinearRegression().fit(ranked_vals, curr_dists)
                full_r2 = fitted_lm.score(ranked_vals, curr_dists)
                self.r2[isub, t] = full_r2
                # Fit a linear regression model and calculate the R-squared for the full model

                # Calculate partial correlation for each factor
                # Skip the intercept column
                for col in range(ranked_vals.shape[1] - 1):
                    submodel_r2 = (
                        LinearRegression()
                        .fit(np.delete(ranked_vals, col, axis=1), curr_dists)
                        .score(np.delete(ranked_vals, col, axis=1), curr_dists)
                    )
                    # Fit a linear regression model without the current factor and calculate the R-squared
                    r_scores[col][t] = np.sqrt(full_r2 - submodel_r2) * np.sign(fitted_lm.coef_[col])
                    # Calculate the partial correlation and store it in r_scores

                r_scores[ranked_vals.shape[1]][t] = np.sqrt(full_r2)
                # Store the total correlation (sqrt of R-squared) for the full model

            r_df = pd.DataFrame(r_scores)
            r_df.columns = factor_df.columns
            r_df["sid"] = isub
            r_df["timepoint"] = self.t

            sub_df = pd.melt(
                r_df,
                id_vars=["sid", "timepoint"],
                value_vars=r_df.columns[:-2],
                var_name="factor",
                value_name="semipartial correlation",
            )

            # Append the correlation dataframe
            partial_r_df = pd.concat([partial_r_df, sub_df], axis=0)

        partial_r_df = partial_r_df.reset_index(drop=True)
        self.partial_r_df = partial_r_df[partial_r_df["factor"] != "Intercept"]

    ##################################
    # VISUALIZATIONS
    ##################################

    def visualize_rdm(self, key: str = "Empirical", title="Dataset RDM", ax=None, border=False, **kwargs):
        """
        Plot a RDM
        Arguments:
        key: which RDM, one of any theoretical model or "Empirical" to plot the empirical RDM
        averaged over the delay period
        title: plot title
        ax: subplot axis. Useful for plotting multiple RDMs on 1 axis
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if key == "Empirical":
            model = self.rdms[..., self.t > self.delay_period_start].mean((0, -1))
        else:
            if key not in self.theoretical_models.keys():
                raise ValueError('Key must be one of "Empirical" or a valid theoretical model')
            model = self.theoretical_models[key]

        sns.heatmap(model, ax=ax, xticklabels=self.labels, yticklabels=self.labels, **kwargs)  # plot the RDM
        ax.set_title(title)
        if border:
            for _, spine in ax.spines.items():
                spine.set_visible(True)

    def plot_corrs(
        self,
        fac_order=None,
        y_sig=0.3,
        t_start=None,
        t_end=None,
        star_size=20,
        title="semipartial correlation of RDMs During Delay Period",
        ax=None,
        **kwargs,
    ):
        """
        Plots a barplot of partial correlations for each factor, averaged over time
        Arguments:
        fac_order: list of factors to plot, in order (default all)
        y_sig: where to put stars for significance
        t_start, t_end: time range to use, default delay_period_start and end
        title: figure title


        Returns:
        ax: axis for further modification
        stats: dataframe of test statistics

        """
        if ax is None:
            fig = plt.figure(facecolor="white", figsize=(8, 4))  # set up figure
            ax = fig.add_subplot(111)

        if fac_order is None:
            fac_order = self.partial_r_df.factor.unique()
            fac_order = fac_order[fac_order != "Intercept"]

        # default to beginning and end of delay period
        t_start = self.delay_period_start if t_start is None else t_start
        t_end = self.delay_period_end if t_end is None else t_end

        # average partial correlations over selected time
        delay_summary_df = (
            self.partial_r_df.query(f"timepoint > {t_start} & timepoint < {t_end}")
            .groupby(["sid", "factor"])
            .mean()
            .reset_index()
        )
        delay_summary_df = delay_summary_df[~(delay_summary_df.factor == "Total")]  # ignore total

        ax.hlines(0, xmin=-0.5, xmax=3.5, color="black", linestyle="--")  # 0 line
        ax = sns.barplot(
            data=delay_summary_df,
            x="factor",
            y="semipartial correlation",
            hue="factor",
            legend=False,
            errorbar=("ci", 68),
            palette=self.color_palette,
            order=fac_order,
            ax=ax,
            **kwargs,
        )  # plot correlations

        # significance testing
        stats = []

        for i, factor in enumerate(fac_order):
            x = delay_summary_df.query(f'factor=="{factor}"')["semipartial correlation"].values
            # wilcoxcon rank-signed test
            w, p = sista.wilcoxon(x=x, nan_policy="omit", alternative="greater")
            if any(np.isnan(x)):
                warnings.warn("Warning: Partial correlations contain nans. Check your data", RuntimeWarning)
            # print out test statistics and factors
            stats.append(pd.DataFrame([{"factor": factor, "mean": np.mean(x), "w": w, "p": p}]))

            ax.scatter(i, y_sig, alpha=0)  # dummy points to annotate

            # annotate spots with significance labels
            if p < 0.001:
                ax.annotate(
                    "***",
                    (i, y_sig),
                    size=star_size,
                    color=self.color_palette[factor],
                    label="p < .001",
                    horizontalalignment="center",
                )
            elif p < 0.01:
                ax.annotate(
                    "**",
                    (i, y_sig),
                    size=star_size,
                    color=self.color_palette[factor],
                    label="p < .01",
                    horizontalalignment="center",
                )
            elif p < 0.05:
                ax.annotate(
                    "*",
                    (i, y_sig),
                    size=star_size,
                    color=self.color_palette[factor],
                    label="p < .05",
                    horizontalalignment="center",
                )

        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title(title, fontsize=20, pad=20)

        stats_df = pd.concat(stats)
        return ax, stats_df

    def plot_corrs_temporal(
        self,
        title="Model Fits across time",
        stim_time=[0, 200],
        hide_stim=False,
        ax=None,
        factors: list[str] = None,
        ylim=[None, None],
        y_sig=-0.2,
        sig_size=10,
        **kwargs,
    ):
        """
        Plots correlations of empirical RDM to each factor over timepoints
        Arguments:
        title: plot title
        stim_time: to plot a gray bar over these times
        hide_stim: do not show the stimulus gray bar
        ax: axis to use (default creates a new one)
        factors: iterable of factor names to plot (default: all)
        ylim: figure y axes
        y_sig: y level to START significance dots at (will go down below this)
        sig_size: size of significance dots
        kwargs: passed to sns.lineplot


        Returns:
        ax: axis for further modification

        """

        if ax is None:
            ax = plt.subplot()

        if factors is None:
            factors = self.partial_r_df.factor.unique()
            factors = factors[factors != "Intercept"]

        if ylim[0] is not None and ylim[1] is not None:  # set ylim if specified
            ax.set_ylim(ylim)
        ax.set_xlim((self.t.min(), self.t.max()))

        ax = sns.lineplot(
            x="timepoint",
            y="semipartial correlation",
            hue="factor",
            hue_order=factors,
            data=self.partial_r_df[np.in1d(self.partial_r_df.factor, factors)],
            palette=self.color_palette,
            ax=ax,
            **kwargs,
        )  # plot relevant factors

        ax.hlines(0, xmin=self.t[0], xmax=self.t[-1], color="black", linestyle="--")  # 0 bar

        # significance testing using wilcoxcon test for each timepoint and condition
        for factor in factors:
            tmp_df = self.partial_r_df.query(f'factor=="{factor}"')
            p_values = []
            for t in self.t[self.t > 0]:
                x = tmp_df[tmp_df["timepoint"] == t]["semipartial correlation"].values
                _, p = sista.wilcoxon(x=x, nan_policy="omit", alternative="greater")
                p_values.append(p)
            # correct for n_timepoints comparisons
            _, corrected_p, _, _ = multipletests(p_values, method="fdr_bh")

            sig05 = corrected_p < 0.05
            print(f"{factor}: {sum(sig05)}/{len(sig05)} significant timepoints")


            ax.scatter(
                self.t[self.t > 0][sig05],
                np.ones(sum(sig05)) * (y_sig),
                marker="s",
                s=sig_size,
                color=self.color_palette[factor],
            )  # mark significant points on axis
            y_sig -= 0.03
            ax.get_legend().set_title(None)  # remove legend title because it gets in the way
        ax.set_title(title)

        # gray stim bar ofver stim period
        if not hide_stim:
            y_min, y_max = ax.get_ylim()
            if type(stim_time[0]) is int:
                ax.fill_between(stim_time, [y_min, y_min], [y_max, y_max], color="gray", alpha=0.5, zorder=0)
            elif type(stim_time[0]) is list or type(stim_time[0]) is tuple:
                for time in stim_time:
                    ax.fill_between(time, [y_min, y_min], [y_max, y_max], color="gray", alpha=0.5, zorder=0)

        return ax  # return the axis for further modification

    def correlate_regressors(self, x_factor: str, y_factor: str, title: str = None, xlab=None, ylab=None, ax=None):
        """
        Function to plot correlations of two factors.
        Useful for seeing if they explain similar sources of variance
        Arguments:
        x_factor, y_factor: factors on each axis
        title: plot title
        xlab,ylab: axis labels (default: factor names)
        """
        if ax is None:
            fig, ax = plt.subplots()

        delay_summary_df = (
            self.partial_r_df.query(f"timepoint > {self.delay_period_start}")
            .groupby(["sid", "factor"])
            .mean()
            .reset_index()
        )

        x_corr = delay_summary_df.query(f'factor == "{x_factor}"')["semipartial correlation"]  # pick out correlations
        y_corr = delay_summary_df.query(f'factor == "{y_factor}"')["semipartial correlation"]

        # scatterplot and linear regression
        ax = sns.regplot(x=x_corr, y=y_corr, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(xlab if xlab is not None else x_factor)
        ax.set_ylabel(ylab if ylab is not None else y_factor)

        # calculate linear regression and plot r2 and p values
        lm = sista.linregress(x_corr, y_corr)
        plt.text(
            0.99,
            0.95,
            f"r2 = {np.round(lm.rvalue**2,3)}",
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        p_text = f"p = {lm.pvalue:.2E}" if lm.pvalue < 0.001 else f"p = {round(lm.pvalue,3)}"
        plt.text(0.99, 0.9, p_text, horizontalalignment="right", verticalalignment="center", transform=ax.transAxes)
