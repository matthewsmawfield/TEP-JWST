import matplotlib.pyplot as plt

COLORS = {
    'primary': '#1C2E4A',      # purple-deep
    'primary_light': '#2C3E50', # purple-dark
    'secondary': '#5D6D7E',    # purple-mid
    'accent': '#84a3aa',       # purple-accent
    'highlight': '#1A5276',    # magenta (used for links/highlights)
    'text': '#17202A',         # text-dark
    'background': '#FFFFFF',   # White
    'grid': '#EBEDEF',         # gray-light
    'black': '#17202A',
    'gray': '#566573',         # gray-purple
    'teal': '#84a3aa',         # Mapped to accent
    'blue': '#1A5276',         # Mapped to highlight
}

FIG_SIZE = {
    'single_column': (3.5, 2.6),
    'double_column': (7.2, 4.5),
    'full_width': (10.0, 6.0),
    'web_standard': (7.5, 4.6),
    'web_tall': (7.5, 5.4),
    'web_two_panel': (7.5, 4.0),
    'web_three_panel': (7.5, 3.0),
    'web_quad': (7.5, 5.6),
    'web_dense': (7.5, 6.6),
    'web_grid_3x3': (7.5, 7.2),
    'web_six_panel': (7.5, 5.8),
}

WEB_TARGET_WIDTH_IN = FIG_SIZE['web_standard'][0]
FIG_SCALE = {
    'single_column': 1.0,
    'double_column': 1.0,
    'full_width': 1.0,
    'web_standard': 1.0,
    'web_tall': 1.0,
    'web_two_panel': 1.0,
    'web_quad': 1.0,
    'web_dense': 1.0,
    'web_grid_3x3': 1.0,
    'web_six_panel': 1.0,
}


def set_pub_style(scale=1.0, dpi=300, transparent=False):
    base_font = 11
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': base_font * scale,
        'axes.labelsize': (base_font + 1) * scale,
        'axes.titlesize': (base_font + 2) * scale,
        'xtick.labelsize': (base_font - 1) * scale,
        'ytick.labelsize': (base_font - 1) * scale,
        'legend.fontsize': (base_font - 1) * scale,
        'figure.titlesize': (base_font + 3) * scale,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.transparent': transparent,
        'figure.facecolor': 'white' if not transparent else 'none',
        'axes.facecolor': 'white' if not transparent else 'none',
        'axes.edgecolor': '#475569',
        'axes.labelcolor': '#0f172a',
        'xtick.color': '#0f172a',
        'ytick.color': '#0f172a',
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#475569',
        'savefig.facecolor': 'white' if not transparent else 'none',
        'savefig.edgecolor': 'white' if not transparent else 'none',
        'grid.color': '#475569',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.2,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'text.usetex': False,
        'text.color': '#0f172a',
    })
    
    import matplotlib as mpl
    # Set color cycle to blues/slate
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        '#2563eb', '#64748b', '#60a5fa', '#1e3a8a', '#b43b4e'
    ])
