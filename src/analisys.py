import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import json
import os
import logging # For logging
import ipywidgets as widgets
from IPython.display import display, clear_output


# -------------------------------
# 1. Loading and Processing Data
# -------------------------------

# Load data from the CSV file (change the filename as needed)
df = pd.read_csv('circular_black.csv')

# Define a dictionary for alternative column names if the expected one is not present
alt_columns = {
    'area': 'area_mm2',
    'perimeter': 'perimeter_mm',
    'avg_defect_depth': 'avg_defect_depth_mm'
}

# List of features for analysis (names we want to use)
features = ['aspect_ratio', 'area', 'perimeter', 'extent', 
            'hu_moments_norm', 'circularity', 
            'convexity_defects_count', 'avg_defect_depth', 'avg_color_val']

# Build a mapping from feature names to actual column names in df.
feature_columns = {}
for feature in features:
    if feature in df.columns:
        feature_columns[feature] = feature
    elif feature in alt_columns and alt_columns[feature] in df.columns:
        feature_columns[feature] = alt_columns[feature]
    else:
        # If no alternative is found, default to the feature name
        feature_columns[feature] = feature

# Convert the specified numeric columns to numeric type
for feature in features:
    col = feature_columns[feature]
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute the L2 norm for the hu_moments column to obtain a single number
def compute_hu_norm(hu_str):
    try:
        hu_list = ast.literal_eval(hu_str)
        return np.linalg.norm(hu_list)
    except Exception:
        return np.nan

if 'hu_moments' in df.columns:
    df['hu_moments_norm'] = df['hu_moments'].apply(compute_hu_norm)
elif 'hu_moments_norm' not in df.columns:
    df['hu_moments_norm'] = np.nan

# Process the avg_color column: assume stored as a string, e.g., "[0, 0, 0]"
def compute_avg_color(color_str):
    try:
        color_list = ast.literal_eval(color_str)
        return np.mean(color_list)
    except Exception:
        return np.nan

if 'avg_color' in df.columns:
    df['avg_color_val'] = df['avg_color'].apply(compute_avg_color)
elif 'avg_color_val' not in df.columns:
    df['avg_color_val'] = np.nan

# -------------------------------
# 2. Preparing Interactive Widgets
# -------------------------------

# For each feature, use the actual column from df (using feature_columns)
default_ranges = {}
for feature in features:
    col = feature_columns[feature]
    if col in df.columns:
        lower_default = df[col].quantile(0.05)
        upper_default = df[col].quantile(0.95)
        min_val = df[col].min()
        max_val = df[col].max()
        default_ranges[feature] = (min_val, lower_default, upper_default, max_val)
    else:
        default_ranges[feature] = (0, 0, 0, 0)

# Create slider widgets for each feature (FloatRangeSlider)
slider_widgets = {}
for feature in features:
    min_val, lower_default, upper_default, max_val = default_ranges[feature]
    # Avoid zero-length range: if min == max, set max = min + 1 and value accordingly
    if min_val == max_val:
        slider = widgets.FloatRangeSlider(
            value=[min_val, min_val + 1],
            min=min_val,
            max=min_val + 1,
            step=0.01,
            description=feature,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f'
        )
    else:
        slider = widgets.FloatRangeSlider(
            value=[lower_default, upper_default],
            min=min_val,
            max=max_val,
            step=(max_val - min_val) / 100,
            description=feature,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f'
        )
    slider_widgets[feature] = slider

# Text field for entering the form name
form_name_widget = widgets.Text(
    value='Black_Cylindrical_Tablets',
    placeholder='Enter form name',
    description='Form:',
    disabled=False
)

# Color picker widget (default color is black)
color_widget = widgets.ColorPicker(
    concise=False,
    description='Color:',
    value='#000000',
    disabled=False
)

# Button to save the configuration
save_button = widgets.Button(
    description='Save Parameters',
    button_style='success'
)

# Output widget for displaying plots and messages
output = widgets.Output()

# Dropdown widget for selecting a feature (for histogram visualization)
feature_dropdown = widgets.Dropdown(
    options=features,
    description='Parameter:',
    value=features[0]
)

# Function to update the histogram for the selected feature
def update_histogram(change):
    feature = feature_dropdown.value
    slider = slider_widgets[feature]
    lower, upper = slider.value
    col = feature_columns.get(feature, feature)
    # Use df.get to avoid KeyError; if the column is missing, an empty Series is returned
    data = df.get(col, pd.Series([])).dropna()
    
    with output:
        clear_output(wait=True)
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(lower, color='red', linestyle='--', label='Lower bound')
        plt.axvline(upper, color='green', linestyle='--', label='Upper bound')
        plt.title(f'Distribution for {feature} (column: {col})')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

# Update the histogram when the selected feature or slider value changes
for slider in slider_widgets.values():
    slider.observe(update_histogram, names='value')
feature_dropdown.observe(update_histogram, names='value')

# Initial histogram update
update_histogram(None)

# Function to handle the "Save Parameters" button click
def on_save_button_clicked(b):
    config = {}
    # Collect slider values for each feature
    for feature in features:
        lower, upper = slider_widgets[feature].value
        config[feature] = {"min": lower, "max": upper}
    
    category_name = form_name_widget.value.strip()
    color_value = color_widget.value
    
    # Form the configuration entry for the current category
    config_entry = {category_name: {"parameters": config, "color": color_value}}
    config_filename = "identification_params.json"
    
    # Load the current configuration if file exists
    if os.path.exists(config_filename):
        with open(config_filename, "r") as f:
            existing_config = json.load(f)
    else:
        existing_config = {}
    
    # Update or add the new category
    existing_config.update(config_entry)
    
    # Save the updated configuration
    with open(config_filename, "w") as f:
        json.dump(existing_config, f, indent=4)
    
    with output:
        print(f"Parameters for '{category_name}' have been successfully saved in {config_filename}")

save_button.on_click(on_save_button_clicked)

# -------------------------------
# Layout and Display of the Interface
# -------------------------------
ui = widgets.VBox([
    form_name_widget,
    color_widget,
    widgets.Label("Set Parameter Ranges:"),
    widgets.VBox(list(slider_widgets.values())),
    save_button,
    widgets.Label("View Distribution for Selected Parameter:"),
    feature_dropdown,
    output
])

display(ui)