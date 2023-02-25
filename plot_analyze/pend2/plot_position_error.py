# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/25 7:12 PM
@desc:
"""
import torch


def processData(error_fun, gt_data, baseline_data, HNN_data, HnnModScale_data):
    print(gt_data.shape)

    gt_data = torch.tensor(gt_data)
    baseline_data = torch.tensor(baseline_data)
    baseline_error = error_fun(gt_data, baseline_data)

    print(baseline_error.mean(0))
    pass


def plot_position_error(*args, **kwargs):
    processData(*args, **kwargs)
    # legendsize = 12
    #
    # fig, ax = plt.subplots(figsize=[8, 2.5], dpi=DPI)
    # labels = ['ModLaNet (Ours)', 'HNN', 'Baseline', 'LightHNN', 'LightBaseline']
    # lines = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 5, 1, 5, 1, 5)),
    #          (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)),
    #          (0, (3, 10, 1, 10, 1,
    #               10))]  # 'loosely dotted', 'loosely dashed', 'loosely dashdotted', 'dashdotdotted' and 'loosely dashdotdotted'
    #
    # x_array = np.array(x_list, dtype=np.float32)
    # # with sns.axes_style("darkgrid"):
    # epochs = np.linspace(t_span[0], t_span[1], end_time * timescale + 1)
    # for i in range(num_models - 1):
    #     x_error = x_array[i + 1] - x_array[0]
    #     x_error = np.linalg.norm(x_error, axis=1)
    #     meanst = np.mean(x_error, axis=0)
    #     sdt = np.std(x_error, axis=0)
    #     ax.plot(epochs, meanst, label=labels[i],linestyle=lines[i])
    #     ax.fill_between(epochs, meanst, meanst + sdt, alpha=0.3, facecolor=clrs[i])
    #
    #
    # ax.legend(fontsize=legendsize)
    # ax.set_yscale('log')
    # ax.tick_params(axis="y", direction='in')  # , length=8)
    # ax.tick_params(axis="x", direction='in')  # , length=8)
    # ax.set_ylim(top=1e3)
    # ax.set_xlim([-1, 32])
    # ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])
    # ax.annotate('$t$', xy=(0.98, -0.025), ha='left', va='top', xycoords='axes fraction')
    # ax.annotate('MSE', xy=(-0.07, 1.05), xytext=(-15, 2), ha='left', va='top', xycoords='axes fraction',
    #             textcoords='offset points')
    #
    # # ax.grid('on')
    # fig.savefig('{}/pend-2-100traj-pos.png'.format(args.result_dir))
    # # ax.set_ylabel('MSE of position ($m$)')
    # # ax.set_xlabel('Time ($s$)')
    pass
