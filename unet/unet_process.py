import os
import cv2
import numpy as np
import torch
import tempfile
from unet.net import UNet
from unet.data import transform
from unet.utils import keep_image_size_open_rgb  # assuming this is defined in utils

def process_stream(input_files: dict) -> dict:
    """
    Processes a set of input images (provided as a dict with filenames as keys and file bytes as values)
    using the UNet model and returns a dict of processed output images (encoded as PNG bytes),
    following the same image processing pipeline as in test.py.
    """
    # Initialize network model and load weights from 'params/save_pth_770/unet.pth'
    net = UNet(2).cuda()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights = os.path.join(current_dir, 'params', 'unet.pth')
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully loaded weights')
    else:
        print('failed to load weights')

    net.eval()
    output_files = {}

    for filename, file_bytes in input_files.items():
        # Write the file bytes to a temporary file so we can use keep_image_size_open_rgb
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(file_bytes)
            temp_filename = tmp.name

        try:
            # Use the same image loading function as test.py
            img = keep_image_size_open_rgb(temp_filename)
        except Exception as e:
            print(f"Failed to process image {filename} using keep_image_size_open_rgb: {e}")
            os.remove(temp_filename)
            continue

        # Remove the temporary file
        os.remove(temp_filename)

        # Preprocess using the transform (same as test.py)
        img_data = transform(img).cuda()
        img_data = torch.unsqueeze(img_data, dim=0)

        with torch.no_grad():
            out = net(img_data)
            out = torch.argmax(out, dim=1)
            out = torch.squeeze(out, dim=0)
            out = out.unsqueeze(dim=0)
            # Permute to (H, W, 1) and convert to NumPy array
            out = out.permute((1, 2, 0)).cpu().detach().numpy()
            # Replace numpy.resize with cv2.resize for proper image scaling
            out = cv2.resize(out, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Multiply by 255.0 as in test.py and convert to uint8 type
        out = (out * 255.0).astype(np.uint8)

        # Encode the processed image as PNG
        success, encoded_image = cv2.imencode('.png', out)
        if success:
            # out_filename = os.path.splitext(filename)[0] + '.png'
            # output_files[out_filename] = encoded_image.tobytes()
            # print(f"Processed: {filename} -> {out_filename}")
            output_files[filename] = encoded_image.tobytes()
            print(f"Processed: {filename} -> {filename}")
        else:
            print(f"Failed to encode image: {filename}")

    print("Processing complete.")
    return output_files

if __name__ == '__main__':
    input_folder = 'test_input'
    if os.path.isdir(input_folder):
        output_folder = 'output_stream'
        os.makedirs(output_folder, exist_ok=True)
        input_files = {}
        # Read all valid image files from the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'rb') as f:
                    input_files[filename] = f.read()

        # Process the images in streaming mode
        result = process_stream(input_files)

        # Save each processed image to the output folder
        for out_filename, img_bytes in result.items():
            output_path = os.path.join(output_folder, out_filename)
            with open(output_path, 'wb') as f:
                f.write(img_bytes)
            print(f"Saved: {output_path}")
    else:
        print("Invalid folder path.")
