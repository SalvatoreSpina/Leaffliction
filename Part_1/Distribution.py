import os
import sys
import matplotlib.pyplot as plt
from collections import Counter
import argparse

def fetch_images(directory):
    """Fetch images from subdirectories."""
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                file_path = os.path.join(root, file)
                plant_type = os.path.basename(root)
                images.append((file_path, plant_type))
    return images

def analyze_images(images):
    """Analyze the image data and return a count of images per plant type."""
    plant_counter = Counter()
    for _, plant_type in images:
        plant_counter[plant_type] += 1
    return plant_counter

def plot_charts(plant_counter, directory_name, save=False, save_path=None):
    """Plot pie chart and bar chart based on the plant image distribution."""
    plant_types = list(plant_counter.keys())
    counts = list(plant_counter.values())

    # Pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=plant_types, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of Plant Types in {directory_name} - Pie Chart')
    if save:
        pie_chart_path = os.path.join(save_path, f'{directory_name}_distribution_pie_chart.png')
        plt.savefig(pie_chart_path)
        print(f"Saved pie chart to {pie_chart_path}")
    plt.show()

    # Bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(plant_types, counts, color='skyblue')
    plt.title(f'Distribution of Plant Types in {directory_name} - Bar Chart')
    plt.xlabel('Plant Type')
    plt.ylabel('Number of Images')
    if save:
        bar_chart_path = os.path.join(save_path, f'{directory_name}_distribution_bar_chart.png')
        plt.savefig(bar_chart_path)
        print(f"Saved bar chart to {bar_chart_path}")
    plt.show()

def main(directory, save, save_path):
    """Main function to execute the program."""
    directory_name = os.path.basename(directory)
    images = fetch_images(directory)
    if not images:
        print(f"No images found in the directory {directory}")
        return

    plant_counter = analyze_images(images)
    plot_charts(plant_counter, directory_name, save, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and plot the distribution of plant images in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing plant images.')
    parser.add_argument('--save', action='store_true', help='Flag to save the charts as image files.')
    parser.add_argument('--save_path', type=str, default='.', help='Path to save the charts if --save is specified. Default is the current directory.')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"The path '{args.directory}' is not a valid directory.")
        sys.exit(1)

    if args.save and not os.path.exists(args.save_path):
        print(f"The save path '{args.save_path}' does not exist.")
        sys.exit(1)

    main(args.directory, args.save, args.save_path)
