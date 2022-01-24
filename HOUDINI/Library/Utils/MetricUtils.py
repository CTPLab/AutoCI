import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pylatex as pl


class coxsum():
    def __init__(self,
                 index,
                 params,
                 alpha=0.05,
                 file_nm='portec'):
        """This class print the summarized table of the output obtained 
        by the simple neural network supervised with the cox loss
        https://github.com/havakv/pycox.git. 

        This implementation borrows from
        https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/fitters/cox_time_varying_fitter.py.


        Args:
            index: the name of causal variable candidates, stored in 
                HOUDINI/Yaml/PORTEC.yaml
            params: the gradients of the nn w.r.t to the input 
            alpha: by default == 0.05, indicating the z value of 95 
                confidence interval (CI) 
            file_nm: file name in the path where the table is stored
        """

        self.alpha = alpha
        self.ci = 100 * (1 - self.alpha)
        self.z = self._inv_normal_cdf(1 - self.alpha / 2)
        self.index = index
        self.file_nm = file_nm
        # be careful with the axis
        # here the param is the gradients for inputs
        self.param = pd.Series(np.mean(params, axis=0),
                               name='param',
                               index=self.index)
        # standard error
        self.se = pd.Series(stats.sem(params, axis=0),
                            name='se',
                            index=self.index)
        # confidence interval
        self.conf_inv = self._compute_confidence_intervals(self.param,
                                                           self.se,
                                                           self.ci,
                                                           self.z)

    def _inv_normal_cdf(self, p) -> float:
        return stats.norm.ppf(p)

    def _compute_confidence_intervals(self,
                                      haz,
                                      se,
                                      ci,
                                      z) -> pd.DataFrame:
        conf = pd.DataFrame(np.c_[haz - z * se, haz + z * se],
                            columns=['{}% lower-bound'.format(ci),
                                     '{}% upper-bound'.format(ci)],
                            index=self.index)
        return conf

    def _compute_z_values(self):
        return self.param / self.se

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _quiet_log2(self, p):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    'divide by zero encountered in log2')
            return np.log2(p)

    def summary(self, df_path) -> pd.DataFrame:
        """The statistical summary table 
        stored in the df_path
        """

        filename = str(df_path / self.file_nm)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
            df = pd.DataFrame(index=self.index)
            df.index.name = 'features'
            df['coef'] = self.param
            df['exp(coef)'] = pd.Series(np.exp(self.param),
                                        name='exp(coef)',
                                        index=self.index)
            df['se(coef)'] = self.se
            df['{}% CI(cl)'.format(self.ci)
               ] = self.conf_inv['{}% lower-bound'.format(self.ci)]
            df['{}% CI(cu)'.format(self.ci)
               ] = self.conf_inv['{}% upper-bound'.format(self.ci)]
            df['{}% CI(el)'.format(self.ci)] = np.exp(
                self.param - self.z * self.se)
            df['{}% CI(eu)'.format(self.ci)] = np.exp(
                self.param + self.z * self.se)
            df['z'] = self._compute_z_values()
            df['p'] = self._compute_p_values()
            # avoid zero
            df.loc[df['p'] <= 1e-16, 'p'] = 1e-16
            df['-log2(p)'] = -self._quiet_log2(df['p'])
            # 4 digits after decimal
            # df.update(df.iloc[:, ].apply(
            #     lambda x: (x * 1e3).astype(int) / 1e3))

        doc = pl.Document()
        doc.packages.append(pl.Package('adjustbox'))
        with doc.create(pl.Section('Table')) as Table:
            Table.append(pl.Command('center'))
            Table.append(pl.Command('tiny'))
            Table.append(pl.NoEscape(r'\begin{adjustbox}{width=1\textwidth}'))
            with doc.create(pl.Tabular('c' * (len(df.columns) + 1))) as table:
                table.add_hline()
                table.add_row([df.index.name] + list(df.columns))
                table.add_hline()
                for row in df.index:
                    table.add_row([row] + list(df.loc[row, :]))
                table.add_hline()
            Table.append(pl.NoEscape(r'\end{adjustbox}'))

        doc.generate_pdf(filename, clean_tex=False)
        return df

    def vis_plot(self,
                 scores,
                 plot_path,
                 labels=None,
                 fig_size=4):
        """Save the boxplot of Concord_Index, Brier_Score and 
        Binomial_Log-likelihood 

        Args:
            scores: Concord_Index, Brier_Score and Binomial_Log-likelihood
                obtained during evaluation 
            plot_path: path to the output boxplot
            labels: the name of metric score stored in HOUDINI/Yaml/PORTEC.yaml 
            fig_size: figure size
        """

        _, ax = plt.subplots(1,
                             len(scores),
                             figsize=(len(scores) * fig_size,
                                      fig_size))
        for ida, axi in enumerate(ax.flat):
            axi.boxplot(scores[ida],
                        vert=True,
                        patch_artist=True,
                        labels=[labels[ida]],
                        showfliers=False)  # Do not show outliers
        plt.tight_layout()
        plt.savefig(str(plot_path / '{}.png'.format(self.file_nm)))
        plt.figure().clear()
        plt.close()


def main():
    params = np.random.rand(256, 4)
    index = ['age', 'gender', 'grade', 'preOP']
    cox = coxsum(index, params)
    df = cox.summary()
    print(df)


if __name__ == '__main__':
    main()
