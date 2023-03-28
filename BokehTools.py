
import bokeh.models as bmo
import matplotlib.cm as cm
import sys

if 'LielTools' in sys.modules:
    from LielTools import PlotTools
else:
    import PlotTools

def categ_colormapper_f_matplotlib_cm(categories, matplotlib_cm_name='Set1'):
    palette = cm.get_cmap(matplotlib_cm_name).colors
    palette = [PlotTools.color_rgb_to_hexa_string(rgb_tup, vals_between_0_1=True) for rgb_tup in palette]
    color_map = bmo.CategoricalColorMapper(factors=categories,
                                           palette=palette)
    return color_map

def categ_colormapper_f_color_list(categories, colors, rgb_or_hexa, rgb_vals_between_0_1=False):
    if rgb_or_hexa=='rgb':
        colors = [PlotTools.color_rgb_to_hexa_string(color, vals_between_0_1=rgb_vals_between_0_1) for color in colors]
    color_map = bmo.CategoricalColorMapper(factors=categories,
                                           palette=colors)
    return color_map


##### check out C:\Users\liel-\Dropbox\PyCharm\PycharmProjectsNew\COVID_mutations\ab ranking\2021_12_26 ab ranking.py
##### check out C:\Users\Liel\Dropbox\PyCharm\PycharmProjectsNew\COVID_pipeline\ab_plot_functions.py