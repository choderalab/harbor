import plotly.graph_objects as go
import pandas as pd
from matplotlib import colors as plt_colors


def rgb_to_rgba(rgb_str, alpha):
    # Split the RGB string into its components
    rgb_values = rgb_str.strip("rgb()").split(",")

    # Extract individual RGB values and convert them to integers
    r, g, b = map(int, rgb_values)

    # Construct the RGBA string
    rgba_str = f"rgba({r}, {g}, {b}, {alpha})"

    return rgba_str


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def update_traces(fig, labels):
    for trace in fig.data:
        if trace.name is None:
            continue
        trace.name = trace.name.replace("_", " ")
        trace.name = trace.name.replace("Split", "")
        trace.name = trace.name.replace(", ", " | ")
        trace.name = trace.name.replace("RMSD", "RMSD (Positive Control)")
        for k, v in labels.items():
            trace.name = trace.name.replace(k, v)
    return fig


def create_plot_with_error_bands_and_dual_legend(
    df,
    x,
    y,
    error_column=None,
    lower_error_column=None,
    upper_error_column=None,
    color_category_column=None,
    dash_category_column=None,
    name_column=None,
    title=None,
    x_title=None,
    y_title=None,
    width=900,
    height=600,
):
    """
    Creates a line plot with error bands and separate legends for categories and types
    from a tidy (long-format) DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Tidy DataFrame containing the data to plot
    x_column : str
        Column name for x-axis values
    y_column : str
        Column name for y-axis values
    error_column : str or None
        Column name for symmetric error values (set to None if using lower/upper error columns)
    lower_error_column : str or None
        Column name for lower error bounds (only used if error_column is None)
    upper_error_column : str or None
        Column name for upper error bounds (only used if error_column is None)
    color_category_column : str or None
        Column name for categories (for color grouping)
    dash_category_column : str or None
        Column name for types (for line style grouping)
    name_column : str or None
        Column name for trace names (if None, uses combination of color_category and type)
    title, x_title, y_title : str
        Plot titles (if None, uses column names)
    width, height : int
        Plot dimensions

    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Create figure
    fig = go.Figure()

    # Set default titles if not provided
    if x_title is None:
        x_title = x
    if y_title is None:
        y_title = y

    # If color_category or type columns not provided, create dummy ones for consistent processing
    df_plot = df.copy()

    if color_category_column is None:
        df_plot["_category"] = "Default Color Category"
        color_category_column = "_color_category"

    if dash_category_column is None:
        df_plot["_type"] = "Default Dash Category"
        dash_category_column = "_dash_category"

    # Determine trace name source
    if name_column is None:
        # Create a name based on color_category and type if not provided
        df_plot["_name"] = (
            df_plot[color_category_column] + " - " + df_plot[dash_category_column]
        )
        name_column = "_name"

    # Get unique values for grouping
    unique_names = df_plot[name_column].unique()
    unique_categories = df_plot[color_category_column].unique()
    unique_types = df_plot[dash_category_column].unique()

    # Create color and dash style mappings
    colors = {
        cat: f"rgb{tuple(int(c * 255) for c in plt_colors.to_rgb(f'C{i}'))}"
        for i, cat in enumerate(unique_categories)
    }
    dash_styles = {
        typ: style
        for typ, style in zip(
            unique_types,
            ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"][
                : len(unique_types)
            ],
        )
    }

    # Process each unique trace (combination of grouping variables)
    for name in unique_names:
        # Filter data for this trace
        trace_data = df_plot[df_plot[name_column] == name]

        if len(trace_data) == 0:
            continue

        # Get first row to determine color_category and type
        first_row = trace_data.iloc[0]
        color_category = first_row[color_category_column]
        type_val = first_row[dash_category_column]

        # Get color and dash style
        color = colors[color_category]
        dash = dash_styles[type_val]

        # Sort data by x values
        trace_data = trace_data.sort_values(by=x)

        # Get x and y values
        x_values = trace_data[x]
        y_values = trace_data[y]

        # Check if we need to draw error bands
        has_error = False
        y_upper = None
        y_lower = None

        if error_column is not None and error_column in trace_data.columns:
            # Symmetric error
            error_values = trace_data[error_column]
            y_upper = y_values + error_values
            y_lower = y_values - error_values
            has_error = True
        elif (
            lower_error_column is not None
            and upper_error_column is not None
            and lower_error_column in trace_data.columns
            and upper_error_column in trace_data.columns
        ):
            # Asymmetric error with separate bounds
            y_lower = trace_data[lower_error_column]
            y_upper = trace_data[upper_error_column]
            has_error = True

        # Add error band (if errors are available)
        if has_error:
            # Add error band as a filled area
            x_error = list(x_values) + list(x_values[::-1])
            y_error = list(y_upper) + list(y_lower[::-1])

            # Convert RGB color to RGBA with transparency
            color_parts = (
                color.replace("rgb", "").replace("(", "").replace(")", "").split(",")
            )
            rgba_color = f"rgba({color_parts[0]},{color_parts[1]},{color_parts[2]},0.3)"

            fig.add_trace(
                go.Scatter(
                    x=x_error,
                    y=y_error,
                    fill="toself",
                    fillcolor=rgba_color,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{name} Error Band",
                )
            )

        # Add line trace
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color=color, dash=dash, width=2),
                name=name,
                legendgroup=name,
                showlegend=False,  # Will hide actual traces from legend
            )
        )

    # Add "dummy" traces for color_category legend (colors)
    for color_category, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],  # No data points
                mode="lines",
                line=dict(color=color, width=2),
                name=color_category,
                legendgroup=color_category_column,
                legendgrouptitle_text=color_category_column,
                showlegend=True,
            )
        )

    # Add "dummy" traces for type legend (line styles)
    for dash_category, dash in dash_styles.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],  # No data points
                mode="lines",
                line=dict(color="black", dash=dash, width=2),
                name=dash_category,
                legendgroup=dash_category_column,
                legendgrouptitle_text=dash_category_column,
                showlegend=True,
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=width,
        height=height,
        legend=dict(groupclick="toggleitem"),
        template="plotly_white",
    )

    return fig


# Example usage
if __name__ == "__main__":
    # example data
    data = [
        {
            "Bootstraps": 1000,
            "StructureChoice": "Dock_to_All",
            "StructureChoice_Choose_N": "All",
            "Score": "POSIT_Probability",
            "Score_Choose_N": 1,
            "EvaluationMetric": "RMSD",
            "EvaluationMetric_Cutoff": 2.0,
            "Split": "SimilaritySplit",
            "N_Per_Split": -1,
            "Split_Variable": "Tanimoto",
            "PoseSelection": "Default",
            "PoseSelection_Choose_N": 1,
            "Min": 0.0,
            "Max": 0.0,
            "CI_Upper": 0.975,
            "CI_Lower": 0.025,
            "Total": 0,
            "Fraction": 0.0,
            "Similarity_Threshold": 0.0,
            "Include_Similar": False,
            "Higher_Is_More_Similar": True,
            "Aligned": True,
            "Type": "TanimotoCombo",
            "Error_Lower": -0.025,
            "Error_Upper": 0.975,
            "Engine": "POSIT",
        },
        {
            "Bootstraps": 1000,
            "StructureChoice": "Dock_to_All",
            "StructureChoice_Choose_N": "All",
            "Score": "POSIT_Probability",
            "Score_Choose_N": 1,
            "EvaluationMetric": "RMSD",
            "EvaluationMetric_Cutoff": 2.0,
            "Split": "SimilaritySplit",
            "N_Per_Split": -1,
            "Split_Variable": "Tanimoto",
            "PoseSelection": "Default",
            "PoseSelection_Choose_N": 1,
            "Min": 0.0181818181818181,
            "Max": 0.0181818181818181,
            "CI_Upper": 0.0519043474859503,
            "CI_Lower": 0.0066036075243498,
            "Total": 165,
            "Fraction": 0.0181818181818181,
            "Similarity_Threshold": 0.25,
            "Include_Similar": False,
            "Higher_Is_More_Similar": True,
            "Aligned": True,
            "Type": "TanimotoCombo",
            "Error_Lower": 0.0115782106574683,
            "Error_Upper": 0.03372252930413219,
            "Engine": "FRED",
        },
        {
            "Bootstraps": 1000,
            "StructureChoice": "Dock_to_All",
            "StructureChoice_Choose_N": "All",
            "Score": "POSIT_Probability",
            "Score_Choose_N": 1,
            "EvaluationMetric": "RMSD",
            "EvaluationMetric_Cutoff": 2.0,
            "Split": "SimilaritySplit",
            "N_Per_Split": -1,
            "Split_Variable": "Tanimoto",
            "PoseSelection": "Default",
            "PoseSelection_Choose_N": 1,
            "Min": 0.3715596330275229,
            "Max": 0.3715596330275229,
            "CI_Upper": 0.4375041030245344,
            "CI_Lower": 0.310142546161883,
            "Total": 218,
            "Fraction": 0.3715596330275229,
            "Similarity_Threshold": 0.5,
            "Include_Similar": False,
            "Higher_Is_More_Similar": True,
            "Aligned": True,
            "Type": "TanimotoCombo",
            "Error_Lower": 0.06141708686563985,
            "Error_Upper": 0.06594446999701153,
            "Engine": "POSIT",
        },
        {
            "Bootstraps": 1000,
            "StructureChoice": "Dock_to_All",
            "StructureChoice_Choose_N": "All",
            "Score": "POSIT_Probability",
            "Score_Choose_N": 1,
            "EvaluationMetric": "RMSD",
            "EvaluationMetric_Cutoff": 2.0,
            "Split": "SimilaritySplit",
            "N_Per_Split": -1,
            "Split_Variable": "Tanimoto",
            "PoseSelection": "Default",
            "PoseSelection_Choose_N": 1,
            "Min": 0.7522935779816514,
            "Max": 0.7522935779816514,
            "CI_Upper": 0.8048516150447931,
            "CI_Lower": 0.690843980335917,
            "Total": 218,
            "Fraction": 0.7522935779816514,
            "Similarity_Threshold": 0.75,
            "Include_Similar": False,
            "Higher_Is_More_Similar": True,
            "Aligned": True,
            "Type": "TanimotoCombo",
            "Error_Lower": 0.06144959764573443,
            "Error_Upper": 0.05255803706314166,
            "Engine": "FRED",
        },
    ]
    df = pd.DataFrame.from_records(data)

    # Example with asymmetric errors
    fig = create_plot_with_error_bands_and_dual_legend(
        df=df,
        x="Similarity_Threshold",
        y="Fraction",
        error_column=None,  # Don't use symmetric error
        lower_error_column="Error_Lower",  # For asymmetric error
        upper_error_column="Error_Upper",  # For asymmetric error
        color_category_column="Engine",
        dash_category_column="Score",
        title="Fraction of Ligands with RMSD < 2.0",
        x_title="TanimotoCombo Similarity",
        y_title="Fraction of Ligands with RMSD < 2.0",
    )

    fig.show()
