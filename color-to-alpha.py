import os
import numpy as np
from PIL import Image
import last_folder_helper

def color_to_alpha(img_array, target_color, alpha_threshold=0.02, cleanup_alpha=0.1, cleanup_distance=8):
    target = np.array(target_color[:3], dtype=np.float64) / 255.0
    rgba = img_array.astype(np.float64) / 255.0
    rgb = rgba[:, :, :3]
    result = np.zeros_like(rgba)
    alpha = np.zeros(rgb.shape[:2], dtype=np.float64)
    for c in range(3):
        channel = rgb[:, :, c]
        t = target[c]
        high_mask = channel >= t
        low_mask = ~high_mask
        with np.errstate(divide='ignore', invalid='ignore'):
            high_alpha = np.where(high_mask, (channel - t) / (1.0 - t + 1e-10), 0.0)
            low_alpha = np.where(low_mask, (t - channel) / (t + 1e-10), 0.0)
        alpha = np.maximum(alpha, high_alpha)
        alpha = np.maximum(alpha, low_alpha)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.where(alpha < alpha_threshold, 0.0, alpha)
    result[:, :, 3] = alpha
    safe_alpha = np.where(alpha > 1e-6, alpha, 1.0)
    for c in range(3):
        result[:, :, c] = np.clip((rgb[:, :, c] - target[c] * (1.0 - safe_alpha)) / safe_alpha, 0.0, 1.0)
    result[:, :, :3] = np.where(alpha[:, :, np.newaxis] > 1e-6, result[:, :, :3], 0.0)
    original_rgb_255 = img_array[:, :, :3].astype(np.float64)
    target_255 = np.array(target_color[:3], dtype=np.float64)
    color_dist = np.max(np.abs(original_rgb_255 - target_255), axis=2)
    low_alpha_mask = result[:, :, 3] < (cleanup_alpha * 255.0)
    close_color_mask = color_dist <= cleanup_distance
    cleanup_mask = low_alpha_mask & close_color_mask
    result[cleanup_mask] = 0
    return (result * 255.0).round().astype(np.uint8)

def remove_corner_color(directory):
    directory = os.path.expanduser(directory)
    output_directory = os.path.join(directory, 'processed')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.webp', '.jpg', '.jpeg')):
            input_path = os.path.join(directory, filename)
            output_name = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_directory, output_name)
            try:
                img = Image.open(input_path).convert('RGBA')
                img_array = np.array(img)
                corner_color = img_array[0, 0]
                result_array = color_to_alpha(img_array, corner_color)
                Image.fromarray(result_array, 'RGBA').save(output_path, 'PNG')
                print(f'processed {filename}')
            except Exception as e:
                print(f'error {filename}: {e}')

if __name__ == "__main__":
    default = last_folder_helper.get_last_folder()
    user_input = input(f'Input folder ({default}): ').strip()
    folder = user_input or default
    if not folder:
        folder = '.'
    last_folder_helper.save_last_folder(folder)
    remove_corner_color(folder)

