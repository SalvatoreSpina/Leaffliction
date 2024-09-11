import os
import sys
import shlex
import fnmatch
import argparse
from plantcv import plantcv as pcv

def find_matching_file(pattern, path):
    """Find a file matching the pattern in the given path."""
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None

class ImageTransformer:
    def __init__(self, source_directory, destination_directory):
        """Initialize the ImageTransformer with source and destination directories."""
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        pcv.params.debug_outdir = destination_directory

    def get_image_file_paths(self):
        """Return a list of image file paths from the source directory."""
        image_files = []
        for item in os.listdir(self.source_directory):
            if os.path.isfile(os.path.join(self.source_directory, item)):
                image_files.append(os.path.join(self.source_directory, item))
        return image_files

    def apply_gaussian_blur(self, image):
        """Apply a Gaussian blur to the input image."""
        saturation_channel = pcv.rgb2gray_hsv(rgb_img=image, channel='s')
        binary_threshold = pcv.threshold.binary(gray_img=saturation_channel, threshold=40, max_value=255, object_type='light')
        median_blurred_image = pcv.median_blur(gray_img=binary_threshold, ksize=5)
        gaussian_blurred_image = pcv.gaussian_blur(img=binary_threshold, ksize=(5, 5), sigma_x=0, sigma_y=None)
        return gaussian_blurred_image

    def create_image_mask(self, img):
        """Create a mask for the input image."""
        saturation_channel = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
        binary_threshold = pcv.threshold.binary(gray_img=saturation_channel, threshold=40, max_value=255, object_type='dark')
        median_blurred_image = pcv.median_blur(gray_img=binary_threshold, ksize=5)
        blue_channel = pcv.rgb2gray_lab(rgb_img=img, channel='b')

        blue_threshold = pcv.threshold.binary(gray_img=blue_channel, threshold=120, max_value=255, object_type='dark')
        combined_binary = pcv.logical_or(bin_img1=median_blurred_image, bin_img2=blue_threshold)
        mask = pcv.invert(combined_binary)

        masked_image = pcv.apply_mask(img=img, mask=mask, mask_color='white')

        masked_a_channel = pcv.rgb2gray_lab(rgb_img=masked_image, channel='a')
        masked_b_channel = pcv.rgb2gray_lab(rgb_img=masked_image, channel='b')

        masked_a_threshold = pcv.threshold.binary(gray_img=masked_a_channel, threshold=115, max_value=255, object_type='dark')
        masked_a_threshold1 = pcv.threshold.binary(gray_img=masked_a_channel, threshold=135, max_value=255, object_type='light')
        masked_b_threshold = pcv.threshold.binary(gray_img=masked_b_channel, threshold=128, max_value=255, object_type='light')
        combined_ab1 = pcv.logical_or(bin_img1=masked_a_threshold, bin_img2=masked_b_threshold)
        combined_ab = pcv.logical_or(bin_img1=masked_a_threshold1, bin_img2=combined_ab1)
        filled_mask = pcv.fill(bin_img=combined_ab, size=200)
        final_masked_image = pcv.apply_mask(img=masked_image, mask=filled_mask, mask_color='white')
        return final_masked_image, filled_mask

    def apply_created_mask(self, image):
        """Apply the created mask to the input image."""
        masked_image, filled_mask = self.create_image_mask(image)
        return masked_image

    def apply_roi_and_detect_objects(self, img):
        """Apply ROI and find objects in the input image."""
        saturation_channel = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
        binary_threshold = pcv.threshold.binary(gray_img=saturation_channel, threshold=40, max_value=255, object_type='dark')
        median_blurred_image = pcv.median_blur(gray_img=binary_threshold, ksize=5)
        gaussian_blurred_image = pcv.gaussian_blur(img=binary_threshold, ksize=(5, 5), sigma_x=0, sigma_y=None)
        blue_channel = pcv.rgb2gray_lab(rgb_img=img, channel='b')

        blue_threshold = pcv.threshold.binary(gray_img=blue_channel, threshold=120, max_value=255, object_type='dark')
        combined_binary = pcv.logical_or(bin_img1=median_blurred_image, bin_img2=blue_threshold)
        mask = pcv.invert(combined_binary)

        masked_image = pcv.apply_mask(img=img, mask=mask, mask_color='white')

        masked_a_channel = pcv.rgb2gray_lab(rgb_img=masked_image, channel='a')
        masked_b_channel = pcv.rgb2gray_lab(rgb_img=masked_image, channel='b')

        masked_a_threshold = pcv.threshold.binary(gray_img=masked_a_channel, threshold=115, max_value=255, object_type='dark')
        masked_a_threshold1 = pcv.threshold.binary(gray_img=masked_a_channel, threshold=135, max_value=255, object_type='light')
        masked_b_threshold = pcv.threshold.binary(gray_img=masked_b_channel, threshold=128, max_value=255, object_type='light')
        combined_ab1 = pcv.logical_or(bin_img1=masked_a_threshold, bin_img2=masked_b_threshold)
        combined_ab = pcv.logical_or(bin_img1=masked_a_threshold1, bin_img2=combined_ab1)
        clean_mask = pcv.fill(bin_img=combined_ab, size=200)
        pcv.params.debug_outdir = '.'
        objects, object_hierarchy = pcv.find_objects(img, clean_mask)

        contour, hierarchy = pcv.roi.rectangle(img, 0, 0, img.shape[0], img.shape[1])
        pcv.params.debug = 'print'
        pcv.roi_objects(img=img, roi_contour=contour, roi_hierarchy=hierarchy, object_contour=objects, obj_hierarchy=object_hierarchy, roi_type='partial')
        roi_file = find_matching_file("*_obj_on_img.png", '.')
        trash_file = find_matching_file("*_roi_mask.png", '.')
        transformed_img, path, filename = pcv.readimage(roi_file)
        os.remove(trash_file)
        os.remove(roi_file)
        os.remove('input_image.png')
        return transformed_img

    def analyze_image(self, img):
        """Analyze the input image."""
        image, mask = self.create_image_mask(img)
        objects, object_hierarchy = pcv.find_objects(img, mask)
        obj, mask = pcv.object_composition(img=img, contours=objects, hierarchy=object_hierarchy)
        return pcv.analyze_object(img, obj, mask)

    def apply_pseudolandmarks(self, img):
        """Apply pseudolandmarks to the input image."""
        pcv.params.debug_outdir = '.'
        image, mask = self.create_image_mask(img)
        objects, object_hierarchy = pcv.find_objects(img, mask)
        obj, mask = pcv.object_composition(img=img, contours=objects, hierarchy=object_hierarchy)
        pcv.params.debug = 'print'
        pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask, label="default")
        pcv.params.debug = 'None'
        pseudolandmarks_file = find_matching_file("*_pseudolandmarks.png", '.')
        transformed_img, path, filename = pcv.readimage(pseudolandmarks_file)
        os.remove(pseudolandmarks_file)
        return transformed_img

    def apply_edge_detection(self, img):
        """Apply edge detection to the input image using Canny algorithm."""
        edges = pcv.canny_edge_detect(img)
        return edges

    def save_transformed_image(self, image, transformation_name, filename):
        """Save the transformed image to the destination directory."""
        output_path = os.path.join(self.destination_directory, f"{filename}_{transformation_name}.png")
        print(output_path)
        pcv.print_image(image, output_path)

    def process_single_image(self, image_path, args):
        """Apply selected transformations to a single image."""
        image, _, filename = pcv.readimage(image_path)

        for key, transformation in TRANSFORMATIONS.items():
            if args[key] or args['apply_all']:
                pcv.params.debug = 'None'
                transformed_image = transformation(self, image)

                # If the transformation returns a tuple, unpack it to get the actual image
                if isinstance(transformed_image, tuple):
                    transformed_image = transformed_image[0]

                self.save_transformed_image(transformed_image, key, filename)

    def process_directory_of_images(self, args):
        """Process all image files in a directory by applying transformations."""
        image_files = self.get_image_file_paths()
        for image_file in image_files:
            self.process_single_image(image_file, args)

def parse_command_line_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='Image Transformation Tool',
        description='Program to apply various image transformations to a file or directory of files.'
    )
    parser.add_argument('-src', type=str, default='input_images')
    parser.add_argument('-dst', type=str, default='transformed_images')

    parser.add_argument('-blur', action='store_true')
    parser.add_argument('-mask', action='store_true')
    parser.add_argument('-roi', action='store_true')
    parser.add_argument('-analysis', action='store_true')
    parser.add_argument('-landmarks', action='store_true')
    parser.add_argument('-edges', action='store_true')

    parser.add_argument('-all', action='store_true')

    args = parser.parse_args()
    return {
        'destination_directory': args.dst,
        'source_directory': args.src,
        'blur': args.blur,
        'mask': args.mask,
        'roi': args.roi,
        'analysis': args.analysis,
        'landmarks': args.landmarks,
        'edges': args.edges,
        'apply_all': args.all
    }

TRANSFORMATIONS = {
    "blur": ImageTransformer.apply_gaussian_blur,
    "mask": ImageTransformer.create_image_mask,
    "roi": ImageTransformer.apply_roi_and_detect_objects,
    "analysis": ImageTransformer.analyze_image,
    "landmarks": ImageTransformer.apply_pseudolandmarks,
    "edges": ImageTransformer.apply_edge_detection,
}

if __name__ == '__main__':
    args = parse_command_line_arguments()

    # Initialize the image transformer with source and destination directories
    transformer = ImageTransformer(args['source_directory'], args['destination_directory'])

    if not os.path.exists(args['source_directory']):
        sys.exit('The specified source directory does not exist or is not accessible.')

    if not os.path.exists(args['destination_directory']):
        os.mkdir(args['destination_directory'])

    if os.path.isfile(args['source_directory']):
        transformer.process_single_image(args['source_directory'], args)
        # script = f"python3 tester.py --show {shlex.quote(args['source_directory'])} {shlex.quote(args['destination_directory'])}"
        # os.system(script)

    elif os.path.isdir(args['source_directory']):
        transformer.process_directory_of_images(args)
