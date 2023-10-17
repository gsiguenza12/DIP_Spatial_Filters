"""
Resources:
https://www.askpython.com/python/examples/display-images-using-python
https://realpython.com/image-processing-with-the-python-pillow-library/
https://www.martinreddy.net/gfx/2d-hi.html
https://www.researchgate.net/figure/8-bit-256-x-256-Grayscale-Lena-Image_fig1_3935609
https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/
https://www.geeksforgeeks.org/python-opencv-getting-and-setting-pixels/
https://www.codesansar.com/numerical-methods/linear-interpolation-python.htm

Gabriel Alfredo Siguenza, CS 5550 Digital Image Processing
Hw 2
Dr. Amar Raheja
Date: 10-09-2023
Last modified: 10-16-2023
"""

# Importing Libraries
import cv2

from matplotlib import pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import matplotlib.pyplot
from matplotlib import image as mpimg

global original_image, processed_image_label


def linear_interpolation(image, zoom_factor_height):
    height, width = image.shape
    new_height = int(height * zoom_factor_height)

    # Initialize the zoomed image
    zoomed_image = np.zeros((new_height, width), dtype=np.uint8)

    for i in range(new_height):
        # Compute the corresponding row in the original image
        old_i = i / zoom_factor_height

        # Nearest neighboring rows
        y1, y2 = int(old_i), min(int(old_i) + 1, height - 1)

        # Linear interpolation along the height dimension
        dy = old_i - y1
        interpolated_row = (1 - dy) * image[y1, :] + dy * image[y2, :]

        zoomed_image[i, :] = interpolated_row

    return zoomed_image


def bilinear_interpolation(image, zoom_factor_height, zoom_factor_width):
    height, width = image.shape
    new_height = int(height * zoom_factor_height)
    new_width = int(width * zoom_factor_width)

    # Initialize the zoomed image
    zoomed_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # Compute the coordinates in the original image
            old_i = i / zoom_factor_height
            old_j = j / zoom_factor_width

            # Nearest neighboring pixel coordinates
            x1, y1 = int(old_j), int(old_i)
            x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

            # Interpolate using bilinear interpolation
            dx = old_j - x1
            dy = old_i - y1
            interpolated_value = (1 - dx) * (1 - dy) * image[y1, x1] + \
                                 dx * (1 - dy) * image[y1, x2] + \
                                 (1 - dx) * dy * image[y2, x1] + \
                                 dx * dy * image[y2, x2]

            zoomed_image[i, j] = int(interpolated_value)

    return zoomed_image


def nearest_neighbor_interpolation(image, zoom_factor_height, zoom_factor_width):
    width = int(image.shape[1])
    height = int(image.shape[0])
    new_height = int(height * zoom_factor_height)
    new_width = int(width * zoom_factor_width)

    # init zoomed img
    zoomed_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            old_i = int(i / zoom_factor_height)
            old_j = int(j / zoom_factor_width)
            zoomed_image[i, j] = image[old_i, old_j]

        # uncomment for raw matrix
        # print("zoomed_image shape:", zoomed_image.shape)
        # print("zoomed_image type:", type(zoomed_image))

    return zoomed_image



''' START OF HW2 FILTERS CODE '''


# Used to generate the kernel for each of the spatial filtering options
def generate_laplacian_kernel(resolution):
    # Ensure the resolution is odd
    if resolution % 2 == 0:
        resolution += 1

    # Generate the Laplacian kernel
    laplacian_kernel = np.ones((resolution, resolution), dtype=np.float32)
    center = resolution // 2
    laplacian_kernel[center, center] = -resolution * resolution + 1

    return laplacian_kernel


# Function to apply a smoothing filter (mean filter?)
def apply_smoothing_filter(image, mask_size):
    # Create a mask of ones with the specified size, defining filter kernel of mask x mask
    kernel = np.ones((mask_size, mask_size)) / (mask_size * mask_size)

    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Initialize an output image
    output_image = np.zeros_like(image)

    # Flip the kernel (180-degree rotation)
    kernel = np.flipud(np.fliplr(kernel))

    # Iterate through the image
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest from the image
            roi = image[i:i + kernel_height, j:j + kernel_width]

            # Ensure the ROI and kernel have the same dimensions
            if roi.shape == kernel.shape:
                output_image[i, j] = np.sum(roi * kernel)

    return output_image



# Function to apply a median filter, only removes noise as opposed to mean filter which smooths out
# the variations in the data. Useful for removing salt and pepper noise.
def apply_median_filter(image, mask_size):
    # Implement a median filter logic using the median of pixel values in the neighborhood
    filtered_image = np.zeros_like(image)

    # zeros must be padded around the row edge and the column edge.
    padding = mask_size//2

    # using the specified kernel size, list the pixel values covered by the kernel
    # determine median level, if the kernel covers an even number of pixels, the avg of two median values is used.
    # slide the kernel mask until all the pixels have been iterated through.
    for i in range(padding, image.shape[0] - padding):
        for j in range(padding, image.shape[1] - padding):
            neighborhood = image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            filtered_image[i, j] = np.median(neighborhood)

    return filtered_image


# Function to apply a sharpening Laplacian filter, derivative based filter for edge detection?
# highlight rapid intensity changes in an image, which correspond to edges.
# in other words, highlights intensity discontinuities and de-emphasizes regions with slowly varying gray levels
def apply_sharpening_laplacian_filter(image, mask):
    # Implement a sharpening filter using Laplacian
    # Generate the Laplacian kernel
    laplacian_kernel = generate_laplacian_kernel(mask)

    # Pad the image to handle the convolution at image boundaries
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Get the dimensions of the image and kernel
    rows, cols = image.shape
    krows, kcols = laplacian_kernel.shape

    # Initialize an output image
    sharpened_image = np.zeros_like(image)

    # Perform convolution (manually)
    for i in range(rows):
        for j in range(cols):
            # Extract the region of interest (ROI)
            roi = padded_image[i:i + krows, j:j + kcols]
            # Perform element-wise multiplication and sum to get the convolved value
            conv_value = np.sum(roi * laplacian_kernel)
            sharpened_image[i, j] = image[i, j] - conv_value

    # Ensure pixel values are within [0, 255] range
    sharpened_image = np.clip(sharpened_image, 0, 255)

    # Convert the image to uint8 data type
    sharpened_image = sharpened_image.astype(np.uint8)

    return sharpened_image


# Function to apply a high-boosting filter, a sharpening technique that employs Laplacian filter with modification.
# obtained by adding the amplified Laplacian scaled by A to the original image.
# A stands for amplification factor - determines then extent of sharpening or boosting applied to an image.
# When A = 1, the high boost filter becomes the traditional Laplacian
def apply_high_boost_filter(image, A, mask):
    # Implement a high-boost filter by combining the original image with a sharpened version

    # Sharpen the original image using the Laplacian filter
    # laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = apply_sharpening_laplacian_filter(image, mask)

    sharpened_image = image - (A * laplacian)

    # Ensure pixel values are within [0, 255] range
    high_boost_filtered_image = np.clip(sharpened_image, 0, 255)

    # Convert the image to uint8 data type
    high_boost_filtered_image = high_boost_filtered_image.astype(np.uint8)

    return high_boost_filtered_image


''' END OF FILTERS CODE'''


# histogram equalization, spatial domain
# range is [0, L-1]

def local_histogram_equalization(image, mask_size):
    height, width = image.shape
    half_mask = mask_size // 2
    equalized_image = np.zeros((height, width), dtype=np.uint8)

    # iterating through the pixels in the image using specified mask
    for i in range(half_mask, height - half_mask):
        for j in range(half_mask, width - half_mask):
            # Extract the local neighborhood
            local_region = image[i - half_mask:i + half_mask + 1, j - half_mask:j + half_mask + 1]

            # Compute histogram and CDF for the local neighborhood
            hist, bins = np.histogram(local_region.flatten(), bins=256, range=(0, 256), density=True)
            cdf = hist.cumsum()

            # Perform histogram equalization for the local neighborhood
            equalized_values = np.interp(local_region.flatten(), bins[:-1], cdf * 255)
            equalized_values = equalized_values.reshape(local_region.shape).astype(np.uint8)

            # Replace the center pixel with the equalized value
            equalized_image[i, j] = equalized_values[half_mask, half_mask]

    # calculate and display histogram for local equalized image
    calc_hist(equalized_image)
    return equalized_image


def histogram_equalization(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)

    # Compute the cumulative distribution function (CDF) (Sk)
    cdf = hist.cumsum()
    # normalize the CDF, (probability distribution factor (nk/n))
    cdf_normalized = cdf / cdf.max()

    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized * 255)
    equalized_image = equalized_image.reshape(image.shape).astype(np.uint8)

    # calculate anad display histogram for equalized image
    calc_hist(equalized_image)
    return equalized_image


# Calculates and plots histogram of given image
def calc_hist(image):
    # Flatten the image to 1D array of pixel intensities
    pixels = image.flatten()

    # Calculate histogram
    h, bins, _ = plt.hist(pixels, bins=256, range=(0, 256))

    # Display the histogram
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Grayscale Image')
    plt.show()


def remove_bit_planes(image, bits_to_remove):
    max_value = 255  # assuming 8-bit image
    mask = np.zeros_like(image, dtype=np.uint8)
    for bit in bits_to_remove:
        mask |= 1 << bit

    # remove specified bit planes by bitwise AND with compliment of the mask
    processed_image = image & ~mask

    return processed_image


'''
1.
Vary the spatial resolution of this image from the given scale to 512x512 and down to 32x32 and then zoom it again to 
see the loss of detail. 

Use the nearest neighbor method, linear method (both x or y) and bi-linear interpolation method 
for zooming it back the desired resolution.
'''

'''
2.	
Vary the gray level resolution of your image from 8-bit to a 1-bit image in steps of 1-bits. Let the user decide 
how many number of bits or provide a selection from a drop-down menu.
'''


# this function quantizes an image to a specified number of bits.
# Function to perform bit plane slicing for image
def reduce_gray_resolution(image, bits):
    img = np.asarray(image)

    # Extract the specified bit plane
    # max_pixel_value = (image >> bits) & 1
    L = 2 ** bits
    # mathematical relation between gray level resolution and bits per pixel
    # L = 2^k
    q = 256 / L
    q_image = (img / L).astype(np.uint8)
    q_image = ((q_image / (L - 1)) * 255).astype(np.uint8)

    # Scale the pixel values to the specified number of bits
    # reduced_resolution_image = (max_pixel_value * (image / 255)).astype(np.uint8)

    # Calculate the maximum pixel value for the specified number of bits
    # max_pixel_value = 2 ** bit_plane - 1

    # Multiply by 255 to visualize the bit plane
    # reduced_resolution_image = bit_plane_image * 255

    # return Image.fromarray(quantized_img)
    return q_image


# TODO: Fix this function

# Function to open an image and process it using bit plane slicing
def process_bit_plane_slicing():
    global original_image, processed_image_label

    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    # Load the image using OpenCV
    image = cv2.imread(file_path, 0)
    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Get the selected bit plane
    bit_plane = int(bits_var.get())
    print(bit_plane)

    # # Determine the bit depth sequence (8 to 1 bits in 1-bit steps)
    # bit_depth_sequence = range(8, 0, -1)
    #
    # # Process the image for each bit depth and display the results
    # for num_bits in bit_depth_sequence:
    #     # Reduce the gray level resolution
    reduced_resolution_image = reduce_gray_resolution(image, bit_plane)

    # Convert images to PIL format for displaying in the GUI
    original_image = ImageTk.PhotoImage(Image.fromarray(image))
    processed_image = ImageTk.PhotoImage(Image.fromarray(reduced_resolution_image))

    # Display the images in the GUI
    original_image_label.config(image=original_image)
    processed_image_label.config(image=processed_image)
    original_image_label.image = original_image
    processed_image_label.image = processed_image

    # Update the GUI to show the processed image for each bit depth
    root.update_idletasks()
    root.update()


""" The following functions should be used if you implement a button """


def process_local_histogram():
    global original_image, processed_image_label

    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not open or find the image.")
        return

    # get the mask value
    mask_size = int(mask_size_entry.get())
    local_he_image = local_histogram_equalization(image, mask_size)

    # calculate histogram for original image
    calc_hist(image)

    # Convert images to PIL format for displaying in the GUI
    original_image = ImageTk.PhotoImage(Image.fromarray(image))
    processed_image = ImageTk.PhotoImage(Image.fromarray(local_he_image))

    # Display the images in the GUI
    original_image_label.config(image=original_image)
    processed_image_label.config(image=processed_image)
    original_image_label.image = original_image
    processed_image_label.image = processed_image

    # Update the GUI to show the processed image for each bit depth
    root.update_idletasks()
    root.update()


def process_global_histogram():
    global original_image, processed_image_label

    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not open or find the image.")
        return

    he_image = histogram_equalization(image)
    # calculate histogram for original image
    calc_hist(image)

    # Convert images to PIL format for displaying in the GUI
    original_image = ImageTk.PhotoImage(Image.fromarray(image))
    processed_image = ImageTk.PhotoImage(Image.fromarray(he_image))

    # Display the images in the GUI
    original_image_label.config(image=original_image)
    processed_image_label.config(image=processed_image)
    original_image_label.image = original_image
    processed_image_label.image = processed_image

    # Update the GUI to show the processed image for each bit depth
    root.update_idletasks()
    root.update()


""" End of functions for implementing a button """


# convert and set original image and processed image in GUI
def set_images(og_image, proc_image):
    # Display the images in the GUI
    original_image_label.config(image=og_image)
    processed_image_label.config(image=proc_image)
    original_image_label.image = og_image
    processed_image_label.image = proc_image


# Function to open an image file and display it
def process_image():
    global original_image, processed_image_label

    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Get the selected interpolation method
    interpolation_method = interpolation_var.get()

    # Define the zoom factor for height (downsampling to 32 pixels in height)
    zoom_factor_height = 32 / image.shape[0]  # Scaling factor for height
    zoom_factor_width = 32 / image.shape[1]  # Scaling factor for width

    # Process the image based on the selected interpolation method
    if interpolation_method == "Nearest Neighbor":
        zoomed_image = nearest_neighbor_interpolation(image, zoom_factor_height, zoom_factor_width)
        zoomed_image = nearest_neighbor_interpolation(zoomed_image, 1 / zoom_factor_height, 1 / zoom_factor_width)
    elif interpolation_method == "Bilinear":
        zoomed_image = bilinear_interpolation(image, zoom_factor_height, zoom_factor_width)
        zoomed_image = bilinear_interpolation(zoomed_image, 1 / zoom_factor_height, 1 / zoom_factor_width)
    elif interpolation_method == "Linear":
        zoomed_image = linear_interpolation(image, zoom_factor_height)
        zoomed_image = linear_interpolation(zoomed_image, 1 / zoom_factor_height)

    # Get user input for mask size (default: 3x3)
    mask_size = input("Enter mask size (e.g., '3' for a 3x3 mask): ")
    mask_size = int(mask_size) if mask_size.isdigit() else 3

    # Get user input for A value for high-boost filter
    A = input("Enter A value for high-boost filter: ")
    A = float(A) if A.replace('.', '', 1).isdigit() else 1.0

    # Apply the respective filters

    calc_hist(zoomed_image)
    # eq_image = local_histogram_equalization(image, mask_size)  # global histogram eq
    # kernel = np.ones((3, 3)) / 9
    # smoothed_image = apply_smoothing_filter(image, mask_size)
    median_filtered_image = apply_median_filter(image, mask_size)
    # sharpened_image = apply_sharpening_laplacian_filter(image, mask_size)
    # high_boost_filtered_image = apply_high_boost_filter(image, A, mask_size)
    # lower_order_removed = remove_bit_planes(image, [7])

    """ These calls are the simplified version of what I had before """
    # Convert images to PIL format for displaying in the GUI
    original_image = ImageTk.PhotoImage(Image.fromarray(image))
    # processed_image = ImageTk.PhotoImage(Image.fromarray(zoomed_image))  # sets image to zoomed ^
    # processed_image = ImageTk.PhotoImage(Image.fromarray(eq_image))  # sets image to HE
    processed_image = ImageTk.PhotoImage(Image.fromarray(median_filtered_image))

    set_images(original_image, processed_image)




# TODO: improve GUI, clean up code for readability
# Create the main application window
root = tk.Tk()
root.title("Image Zooming with Linear Interpolation")

# Create a button to open an image
# open_button = tk.Button(root, text="Open Image", command=open_image)
# open_button.pack(pady=10)

# Create a button to open an image and process it using bit plane slicing
bit_plane_button = tk.Button(root, text="Process Image - Bit Plane Slicing", command=process_bit_plane_slicing)
bit_plane_button.pack(pady=10)

# Create a button to open an image and process it
process_button = tk.Button(root, text="Process Image - Interpolation", command=process_image)
process_button.pack(pady=10)

# Create radio buttons for interpolation selection
interpolation_var = tk.StringVar(value="Nearest Neighbor")

neighbour_button = tk.Radiobutton(root, text="Nearest Neighbor", variable=interpolation_var, value="Nearest Neighbor")
neighbour_button.pack(anchor=tk.W)

linear_button = tk.Radiobutton(root, text="Linear", variable=interpolation_var, value="Linear")
linear_button.pack(anchor=tk.W)

bilinear_button = tk.Radiobutton(root, text="Bilinear", variable=interpolation_var, value="Bilinear")
bilinear_button.pack(anchor=tk.W)

# Create a drop-down menu for selecting bit depth
bits_var = tk.StringVar(value="8")  # Default to 8 bits

bits_label = tk.Label(root, text="Select number of bits (for bit plane slicing):")
bits_label.pack(anchor=tk.W)

bits_menu = tk.OptionMenu(root, bits_var, *["1", "2", "3", "4", "5", "6", "7", "8"])
bits_menu.pack(anchor=tk.W)

# Create buttons to process image with local or global histogram equalization
local_histogram_button = tk.Button(root, text="Local Histogram Equalization", command=process_local_histogram)
local_histogram_button.pack(pady=10)

global_histogram_button = tk.Button(root, text="Global Histogram Equalization", command=process_global_histogram)
global_histogram_button.pack(pady=10)

# Create an input box for a specified mask size
mask_label = tk.Label(root, text="Specify Mask Size (for local HE and spatial filters):")
mask_label.pack(anchor=tk.W)

mask_size_entry = tk.Entry(root)
mask_size_entry.pack(anchor=tk.W)

# Create labels to display histograms
# local_histogram_label = tk.Label(root, text="Local Histogram")
# local_histogram_label.pack(side=tk.LEFT, padx=10, pady=10)

# global_histogram_label = tk.Label(root, text="Global Histogram")
# global_histogram_label.pack(side=tk.RIGHT, padx=10, pady=10)

# Create labels to display the original and processed images
original_image_label = tk.Label(root, text="Original Image")
original_image_label.pack(side=tk.LEFT, padx=10, pady=10)

processed_image_label = tk.Label(root, text="Processed Image")
processed_image_label.pack(side=tk.RIGHT, padx=10, pady=10)

# Run the main event loop
root.mainloop()
