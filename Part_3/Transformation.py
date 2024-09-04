import argparse
import os
from plantcv import plantcv as pcv
import sys


class ImageTransformer:
    def __init__(self, source_directory, destination_directory):
        """Initialize the ImageTransformer with source and destination directories."""
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        pcv.params.debug_outdir = destination_directory

    def get_image_files(self):
        """Return a list of image file paths from the source directory."""
        image_files = []
        for item in os.listdir(self.source_directory):
            if os.path.isfile(os.path.join(self.source_directory, item)):
                image_files.append(os.path.join(self.source_directory, item))
        return image_files

    def apply_gaussian_blur(self, image):
        """Apply a Gaussian blur to the input image."""
        hsv_s_channel = pcv.rgb2gray_hsv(rgb_img=image, channel='s')
        binary_s_threshold = pcv.threshold.binary(gray_img=hsv_s_channel, threshold=40, max_value=255, object_type='light')
        gaussian_blurred_image = pcv.gaussian_blur(img=binary_s_threshold, ksize=(5, 5), sigma_x=0, sigma_y=None)
        return gaussian_blurred_image

    def create_image_mask(self, image):
        """Create a mask for the input image based on HSV and LAB color spaces."""
        hsv_s_channel = pcv.rgb2gray_hsv(rgb_img=image, channel='s')
        binary_s_threshold = pcv.threshold.binary(gray_img=hsv_s_channel, threshold=40, max_value=255, object_type='dark')
        median_blur_s = pcv.median_blur(gray_img=binary_s_threshold, ksize=5)

        lab_b_channel = pcv.rgb2gray_lab(rgb_img=image, channel='b')
        binary_b_threshold = pcv.threshold.binary(gray_img=lab_b_channel, threshold=120, max_value=255, object_type='dark')

        combined_mask = pcv.logical_or(bin_img1=median_blur_s, bin_img2=binary_b_threshold)
        inverted_mask = pcv.invert(combined_mask)

        masked_image = pcv.apply_mask(img=image, mask=inverted_mask, mask_color='white')
        return masked_image  # Returning only the masked image, not the mask itself

    def apply_roi_and_find_objects(self, image):
        """Apply region of interest (ROI) objects and return the transformed image."""
        # Get the masked image from the create_image_mask method
        masked_image = self.create_image_mask(image)
        
        # Convert the masked image to grayscale and then apply binary thresholding to get a binary mask
        gray_image = pcv.rgb2gray(masked_image)
        binary_mask = pcv.threshold.binary(gray_img=gray_image, threshold=128, max_value=255, object_type='light')
        
        # Now find objects using the binary mask
        objects, object_hierarchy = pcv.find_objects(masked_image, binary_mask)
        
        # Define the region of interest (ROI) as a rectangle covering the full image
        roi_contour, roi_hierarchy = pcv.roi.rectangle(masked_image, 0, 0, masked_image.shape[0], masked_image.shape[1])

        # Apply the ROI to the objects
        pcv.roi_objects(
            img=masked_image, 
            roi_contour=roi_contour, 
            roi_hierarchy=roi_hierarchy, 
            object_contour=objects, 
            obj_hierarchy=object_hierarchy,
            roi_type='partial'
        )
        
        return masked_image


    def analyze_image(self, image):
        """Analyze the input image to identify and analyze objects."""
        masked_image = self.create_image_mask(image)
        objects, object_hierarchy = pcv.find_objects(masked_image, masked_image)  # Using masked_image as mask
        composed_object, object_mask = pcv.object_composition(
            img=masked_image, contours=objects, hierarchy=object_hierarchy
        )
        return pcv.analyze_object(masked_image, composed_object, object_mask)

    def apply_pseudolandmarks(self, image):
        """Apply pseudolandmarks to the input image."""
        masked_image = self.create_image_mask(image)
        objects, object_hierarchy = pcv.find_objects(masked_image, masked_image)  # Using masked_image as mask
        composed_object, object_mask = pcv.object_composition(
            img=masked_image, contours=objects, hierarchy=object_hierarchy
        )
        pcv.y_axis_pseudolandmarks(img=masked_image, obj=composed_object, mask=object_mask, label="default")
        return masked_image

    def create_color_histogram(self, image):
        """Generate a color histogram for the input image."""
        masked_image = self.create_image_mask(image)
        return pcv.analyze_color(rgb_img=image, mask=masked_image, colorspaces='all', label="default")

    def save_image(self, image, transformation_name, filename):
        """Save the transformed image to the destination directory."""
        output_path = os.path.join(self.destination_directory, f"{filename}_{transformation_name}.png")
        print(output_path)
        pcv.print_image(image, output_path)

    def process_single_image(self, image_path, args):
        """Apply selected transformations to a single image."""
        image, _, filename = pcv.readimage(image_path)

        if args['apply_all']:
            for key, transformation in TRANSFORMATIONS.items():
                transformed_image = transformation(self, image)

                # If the transformation returns a tuple, unpack it to get the actual image
                if isinstance(transformed_image, tuple):
                    transformed_image = transformed_image[0]

                self.save_image(transformed_image, key, filename)
        else:
            for key, transformation in TRANSFORMATIONS.items():
                if args[key]:
                    transformed_image = transformation(self, image)

                    # If the transformation returns a tuple, unpack it to get the actual image
                    if isinstance(transformed_image, tuple):
                        transformed_image = transformed_image[0]

                    self.save_image(transformed_image, key, filename)

    def process_directory_of_images(self, args):
        """Process all image files in a directory by applying transformations."""
        image_files = self.get_image_files()
        for image_file in image_files:
            self.process_single_image(image_file, args)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='Image Transformation Tool',
        description='Program to apply various image transformations to a file or directory of files.'
    )
    parser.add_argument('-dst', type=str, default='transformed_images')
    parser.add_argument('-src', type=str, default='input_images')
    parser.add_argument('-blur', action='store_true')
    parser.add_argument('-mask', action='store_true')
    parser.add_argument('-roi', action='store_true')
    parser.add_argument('-analysis', action='store_true')
    parser.add_argument('-landmarks', action='store_true')
    parser.add_argument('-colors', action='store_true')
    parser.add_argument('-all', action='store_true')
    parser.add_argument('-v', action='store_true')

    args = parser.parse_args()
    return {
        'destination_directory': args.dst,
        'source_directory': args.src,
        'apply_blur': args.blur,
        'apply_mask': args.mask,
        'apply_roi': args.roi,
        'apply_analysis': args.analysis,
        'apply_landmarks': args.landmarks,
        'apply_color_histogram': args.colors,
        'apply_all': args.all
    }


TRANSFORMATIONS = {
    "blur": ImageTransformer.apply_gaussian_blur,
    "mask": ImageTransformer.create_image_mask,
    "roi": ImageTransformer.apply_roi_and_find_objects,
    "analysis": ImageTransformer.analyze_image,
    "landmarks": ImageTransformer.apply_pseudolandmarks,
    "colors": ImageTransformer.create_color_histogram
}


if __name__ == '__main__':
    args = parse_arguments()

    # Initialize the image transformer with source and destination directories
    transformer = ImageTransformer(args['source_directory'], args['destination_directory'])

    if not os.path.exists(args['source_directory']):
        sys.exit('The specified source directory does not exist or is not accessible.')

    if not os.path.exists(args['destination_directory']):
        os.mkdir(args['destination_directory'])

    if os.path.isfile(args['source_directory']):
        transformer.process_single_image(args['source_directory'], args)
    elif os.path.isdir(args['source_directory']):
        transformer.process_directory_of_images(args)
