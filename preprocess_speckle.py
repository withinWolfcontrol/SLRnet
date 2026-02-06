import os
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_image(args):
    """
    Function to process a single image, designed for multiprocessing.

    Args:
    args (tuple): A tuple containing the following elements:
        - input_path (str): Full path to the input image file.
        - output_path (str): Full path to save the processed image.
        - crop_size (int): Target size for center cropping.
        - resize_size (int): Target size for resizing.
    """
    input_path, output_path, crop_size, resize_size = args

    # If the output file already exists, skip it to avoid redundant work
    if os.path.exists(output_path):
        return (input_path, "skipped")

    try:
        # Open the image
        with Image.open(input_path) as img:
            # 1. Convert to grayscale
            gray_img = img.convert('L')
            
            # 2. Center crop
            width, height = gray_img.size
            if width < crop_size or height < crop_size:
                 # If the image itself is smaller than the crop size, skip or log a warning
                 return (input_path, "too_small")
            
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = (width + crop_size) / 2
            bottom = (height + crop_size) / 2
            
            cropped_img = gray_img.crop((left, top, right, bottom))
            
            # 3. Resize (Downsampling)
            # Use a high-quality resampling filter (LANCZOS) for anti-aliasing
            resized_img = cropped_img.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
            
            # 4. Save the image
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            resized_img.save(output_path)
            
            return (input_path, "success")
            
    except Exception as e:
        # Catch any possible errors (e.g., corrupted files)
        return (input_path, f"failed: {e}")

def batch_preprocess_images(
    input_dir, 
    output_dir, 
    crop_size, 
    resize_size, 
    output_format=None, 
    num_processes=None
):
    """
    Batch preprocess all images in a directory.

    Args:
    input_dir (str): Input directory path containing images.
    output_dir (str): Output directory path to save processed images.
    crop_size (int): Target size for center cropping.
    resize_size (int): Target size for resizing.
    output_format (str, optional): Output image format extension (e.g., 'png', 'bmp').
                                   If None, preserves the original format. Defaults to None.
    num_processes (int, optional): Number of CPU processes to use. Defaults to half of the system's CPU cores.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    # Prepare the task list
    tasks = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            
            # Construct the output filename
            base_name, old_ext = os.path.splitext(filename)
            new_ext = f".{output_format}" if output_format else old_ext
            output_filename = base_name + new_ext
            output_path = os.path.join(output_dir, output_filename)
            
            tasks.append((input_path, output_path, crop_size, resize_size))

    if not tasks:
        print("No images found in the input directory.")
        return

    # Set the number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() // 2) # Default to half of the CPU cores to prevent system freeze
    
    print(f"Found {len(tasks)} images. Starting preprocessing with {num_processes} processes...")

    # Use a multiprocessing pool for processing
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processing images"))

    # Statistics of processing results
    success_count = sum(1 for _, status in results if status == "success")
    skipped_count = sum(1 for _, status in results if status == "skipped")
    too_small_count = sum(1 for _, status in results if status == "too_small")
    failed_count = len(tasks) - success_count - skipped_count - too_small_count
    
    print("\n--- Preprocessing Summary ---")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Skipped (too small for crop): {too_small_count}")
    print(f"Failed to process: {failed_count}")
    
    if failed_count > 0:
        print("\n--- Failed Files ---")
        for path, status in results:
            if status.startswith("failed"):
                print(f"{path} -> {status}")


if __name__ == '__main__':
    # --- Configure your parameters here ---

    # 1. Input directory path (containing high-resolution original images)
    INPUT_IMAGE_DIR = r'D:\imagenet_with_video_speckle'
    
    # 2. Output directory path (for saving processed images)
    OUTPUT_IMAGE_DIR = r'D:\speckle\imagenet_with_video_group_XX\\'
    
    # 3. Target size for center cropping (e.g., crop a 768x768 square from the center)
    CROP_SIZE = 768
    
    # 4. Target size for resizing (e.g., downsample the cropped image to 200x200)
    RESIZE_SIZE = 200
    
    # 5. Output image format (extension without dot).
    #    If set to None, the original image format will be preserved.
    #    Recommended: 'png' (lossless) or 'jpg' (lossy but smaller file size).
    OUTPUT_FORMAT = 'png'
    
    # 6. Number of CPU cores to use.
    #    If set to None, the script automatically selects a reasonable value (usually half the cores).
    #    You can manually set this based on your PC specs, e.g., 8 or 12.
    NUM_PROCESSES = None
    
    # --- Configuration end, start running ---
    
    batch_preprocess_images(
        input_dir=INPUT_IMAGE_DIR,
        output_dir=OUTPUT_IMAGE_DIR,
        crop_size=CROP_SIZE,
        resize_size=RESIZE_SIZE,
        output_format=OUTPUT_FORMAT,
        num_processes=NUM_PROCESSES
    )