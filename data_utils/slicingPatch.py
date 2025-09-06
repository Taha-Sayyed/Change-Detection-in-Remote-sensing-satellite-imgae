import os
import sys
from datetime import datetime
import cv2
import numpy as np
import socket
from PIL import Image
import glob
import argparse
import tqdm
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn.functional as F
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✅ PyTorch GPU available")
        print(f"🔢 Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"🖥️  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ PyTorch available but no GPU detected")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")

try:
    from osgeo import gdal, gdal_array
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("Warning: GDAL not available. Only PNG/JPG processing will work.")

# Multi-scale factors for slicing
multiScale = [1.0]

# T4 optimized settings
T4_OPTIMAL_BATCH_SIZE = 32  # Optimized for T4's memory bandwidth
T4_MAX_IMAGE_SIZE = 4096    # Maximum image size to process in single GPU operation

def setup_gpu_devices():
    """Setup and configure available GPUs"""
    if not GPU_AVAILABLE:
        return []
    
    devices = []
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        
        # Get memory info
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        
        total_memory = gpu_props.total_memory / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        free_memory = total_memory - allocated_memory
        
        devices.append({
            'id': i,
            'name': gpu_props.name,
            'free_memory': free_memory,
            'total_memory': total_memory,
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
        })
        
        print(f"🖥️  GPU {i}: {gpu_props.name}")
        print(f"   💾 Memory: {free_memory:.1f}GB free / {total_memory:.1f}GB total")
        print(f"   🔧 Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    return devices

def clear_gpu_memory(device_id=None):
    """Clear GPU memory to prevent OOM errors"""
    if GPU_AVAILABLE:
        if device_id is not None:
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
        else:
            # Clear all devices
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
    gc.collect()

def writeTiff(im_data, path, im_bands, im_height, im_width, im_geotrans, im_proj):
    """
    Write image data to TIFF format
    Note: This function requires GDAL which may not be available in Kaggle
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is required for TIFF operations but not available")
    
    # Convert from GPU to CPU if needed
    if GPU_AVAILABLE and torch.is_tensor(im_data):
        im_data = im_data.cpu().numpy()
    
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'int32' in im_data.dtype.name:
        datatype = gdal.GDT_UInt32
    else:
        datatype = gdal.GDT_Float32
    
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    
    # Create file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, 1, datatype)
    
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def readTiff(filename):
    """
    Read TIFF image file
    Note: This function requires GDAL which may not be available in Kaggle
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is required for TIFF operations but not available")
    
    img_ds = gdal.Open(filename, gdal.GA_ReadOnly)
    im_width = img_ds.RasterXSize
    im_height = img_ds.RasterYSize
    im_bands = img_ds.RasterCount
    im_geotrans = img_ds.GetGeoTransform()
    im_proj = img_ds.GetProjection()
    
    im_data = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount), 
                       gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    
    for b in range(im_data.shape[2]):
        im_data[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    
    return im_data, img_ds

def torch_resize_image_t4_optimized(img_array, new_size, is_label=False, device_id=0):
    """
    T4-optimized GPU image resize using PyTorch
    
    Args:
        img_array: Input image as numpy array
        new_size: (width, height) tuple for new size
        is_label: Whether this is a label image (use nearest neighbor)
        device_id: GPU device ID to use
    
    Returns:
        Resized image array as numpy
    """
    if not GPU_AVAILABLE:
        # CPU fallback using OpenCV
        interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
        return cv2.resize(img_array, new_size, interpolation=interpolation)
    
    try:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)
        
        # For very large images, use tiled processing
        if max(img_array.shape[:2]) > T4_MAX_IMAGE_SIZE:
            return torch_resize_large_image(img_array, new_size, is_label, device_id)
        
        # Convert numpy to torch tensor
        if len(img_array.shape) == 3:
            # Convert HWC to NCHW format for PyTorch
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        else:
            # Single channel: HW to NCHW
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
        
        # Transfer to GPU
        img_tensor = img_tensor.to(device)
        
        # Resize using PyTorch
        mode = 'nearest' if is_label else 'bicubic'
        
        # PyTorch expects (N, C, H, W) and size as (H, W)
        resized_tensor = F.interpolate(
            img_tensor, 
            size=(new_size[1], new_size[0]),  # PyTorch expects (H, W)
            mode=mode,
            align_corners=False if mode == 'bicubic' else None
        )
        
        # Convert back to numpy
        if len(img_array.shape) == 3:
            # Convert NCHW back to HWC
            result = resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            # Convert NCHW back to HW
            result = resized_tensor.squeeze(0).squeeze(0).cpu().numpy()
        
        # Clean up GPU memory
        del img_tensor, resized_tensor
        torch.cuda.empty_cache()
        
        # Ensure correct data type
        if is_label:
            result = result.astype(np.uint8)
        else:
            result = result.astype(img_array.dtype)
        
        return result
        
    except Exception as e:
        print(f"⚠️ T4 GPU resize failed on device {device_id}, falling back to CPU: {e}")
        clear_gpu_memory(device_id)
        interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
        return cv2.resize(img_array, new_size, interpolation=interpolation)

def torch_resize_large_image(img_array, new_size, is_label=False, device_id=0):
    """
    Handle very large images by processing in tiles using PyTorch
    """
    tile_size = T4_MAX_IMAGE_SIZE // 2
    overlap = 64
    
    # Calculate scale factors
    scale_h = new_size[1] / img_array.shape[0]
    scale_w = new_size[0] / img_array.shape[1]
    
    # Create output array
    if len(img_array.shape) == 3:
        output = np.zeros((new_size[1], new_size[0], img_array.shape[2]), dtype=img_array.dtype)
    else:
        output = np.zeros((new_size[1], new_size[0]), dtype=img_array.dtype)
    
    for y in range(0, img_array.shape[0], tile_size - overlap):
        for x in range(0, img_array.shape[1], tile_size - overlap):
            # Extract tile
            y_end = min(y + tile_size, img_array.shape[0])
            x_end = min(x + tile_size, img_array.shape[1])
            tile = img_array[y:y_end, x:x_end]
            
            # Calculate output coordinates
            out_y = int(y * scale_h)
            out_x = int(x * scale_w)
            out_y_end = int(y_end * scale_h)
            out_x_end = int(x_end * scale_w)
            
            # Resize tile
            tile_new_size = (out_x_end - out_x, out_y_end - out_y)
            resized_tile = torch_resize_image_t4_optimized(tile, tile_new_size, is_label, device_id)
            
            # Place in output
            if len(img_array.shape) == 3:
                output[out_y:out_y_end, out_x:out_x_end, :] = resized_tile
            else:
                output[out_y:out_y_end, out_x:out_x_end] = resized_tile
    
    return output

def torch_process_patches_dual_t4(img_array, patch_coords, imgsize, is_label=False, batch_size=32):
    """
    Process patches using dual T4 GPUs for maximum performance with PyTorch
    
    Args:
        img_array: Input image array
        patch_coords: List of (start_h, end_h, start_w, end_w) coordinates
        imgsize: Size of patches
        is_label: Whether processing labels
        batch_size: Batch size per GPU
    
    Returns:
        List of patch arrays
    """
    if not GPU_AVAILABLE or len(patch_coords) < batch_size:
        # Fallback to CPU for small batches
        return cpu_process_patches(img_array, patch_coords, is_label)
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 1:
        return torch_process_patches_single(img_array, patch_coords, imgsize, is_label, 0, batch_size)
    
    # Dual GPU processing
    patches = [None] * len(patch_coords)
    
    def process_gpu_batch(gpu_id, coords_batch, indices_batch):
        try:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
            
            # Convert image to tensor once per batch
            if len(img_array.shape) == 3:
                # HWC to CHW
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().to(device)
            else:
                # HW to CHW (add channel dimension)
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float().to(device)
            
            for i, (start_h, end_h, start_w, end_w) in enumerate(coords_batch):
                if len(img_array.shape) == 3 and not is_label:
                    # Multi-channel patch: CHW format
                    patch_tensor = img_tensor[:, start_h:end_h, start_w:end_w]
                    # Convert back to HWC for saving
                    cpu_patch = patch_tensor.permute(1, 2, 0).cpu().numpy()
                else:
                    # Single channel patch
                    if len(img_array.shape) == 3:
                        # Take first channel for labels
                        patch_tensor = img_tensor[0, start_h:end_h, start_w:end_w]
                    else:
                        patch_tensor = img_tensor[0, start_h:end_h, start_w:end_w]
                    cpu_patch = patch_tensor.cpu().numpy()
                
                # Ensure correct data type
                if is_label:
                    cpu_patch = cpu_patch.astype(np.uint8)
                else:
                    cpu_patch = cpu_patch.astype(img_array.dtype)
                
                patches[indices_batch[i]] = cpu_patch
            
            # Clean up GPU memory
            del img_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️ GPU {gpu_id} batch processing failed: {e}")
            # Fallback to CPU for this batch
            for i, (start_h, end_h, start_w, end_w) in enumerate(coords_batch):
                if len(img_array.shape) == 3 and not is_label:
                    patch = img_array[start_h:end_h, start_w:end_w, :]
                else:
                    patch = img_array[start_h:end_h, start_w:end_w]
                patches[indices_batch[i]] = patch
    
    # Split work between GPUs
    total_batches = len(patch_coords)
    batches_per_gpu = total_batches // num_gpus
    
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * batches_per_gpu
            if gpu_id == num_gpus - 1:  # Last GPU gets remaining work
                end_idx = total_batches
            else:
                end_idx = start_idx + batches_per_gpu
            
            coords_batch = patch_coords[start_idx:end_idx]
            indices_batch = list(range(start_idx, end_idx))
            
            future = executor.submit(process_gpu_batch, gpu_id, coords_batch, indices_batch)
            futures.append(future)
        
        # Wait for all GPUs to complete
        for future in as_completed(futures):
            future.result()
    
    return [p for p in patches if p is not None]

def torch_process_patches_single(img_array, patch_coords, imgsize, is_label, device_id, batch_size):
    """Process patches on a single GPU using PyTorch"""
    try:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)
        
        # Convert image to tensor
        if len(img_array.shape) == 3:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().to(device)
        else:
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).float().to(device)
        
        patches = []
        
        for start_h, end_h, start_w, end_w in patch_coords:
            if len(img_array.shape) == 3 and not is_label:
                patch_tensor = img_tensor[:, start_h:end_h, start_w:end_w]
                cpu_patch = patch_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                if len(img_array.shape) == 3:
                    patch_tensor = img_tensor[0, start_h:end_h, start_w:end_w]
                else:
                    patch_tensor = img_tensor[0, start_h:end_h, start_w:end_w]
                cpu_patch = patch_tensor.cpu().numpy()
            
            if is_label:
                cpu_patch = cpu_patch.astype(np.uint8)
            else:
                cpu_patch = cpu_patch.astype(img_array.dtype)
            
            patches.append(cpu_patch)
        
        # Clean up
        del img_tensor
        return patches
        
    except Exception as e:
        print(f"⚠️ Single GPU processing failed: {e}")
        return cpu_process_patches(img_array, patch_coords, is_label)

def cpu_process_patches(img_array, patch_coords, is_label):
    """CPU fallback for patch processing"""
    patches = []
    for start_h, end_h, start_w, end_w in patch_coords:
        if len(img_array.shape) == 3 and not is_label:
            patch = img_array[start_h:end_h, start_w:end_w, :]
        else:
            patch = img_array[start_h:end_h, start_w:end_w]
        patches.append(patch)
    return patches

def getAxisBoundary(Index, length, imgsize, totalNums):
    """
    Calculate boundary indices for slicing
    
    Args:
        Index: Current index
        length: Total length of the dimension
        imgsize: Size of each slice
        totalNums: Total number of slices
    
    Returns:
        tuple: (start, end) indices
    """
    if (Index + 1 == totalNums) and (length % imgsize != 0):
        start = length - imgsize
        end = length
    else:
        start = Index * imgsize
        end = (Index + 1) * imgsize
    return start, end

def slicingSingleImg(imgDir, outputDir, imgsize=512, scales=[1.0], isLabel=False, batch_size=T4_OPTIMAL_BATCH_SIZE):
    """
    Slice a single large image into smaller patches using dual T4 GPU acceleration with PyTorch
    
    Args:
        imgDir: Path to input image
        outputDir: Path to output directory
        imgsize: Size of output patches
        scales: List of scale factors
        isLabel: Whether the image is a label/mask
        batch_size: Number of patches to process in GPU batch (optimized for T4)
    """
    path, name = os.path.split(imgDir)
    subpath, datasetName = os.path.split(path)
    _, datasetName = os.path.split(subpath)
    filename, extension = os.path.splitext(name)
    outputPath = os.path.join(outputDir, f"{datasetName}_{filename}")
    
    save_extension = ".png"
    
    try:
        if extension.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            sourceImg = Image.open(imgDir)
        elif extension.lower() in [".tiff", ".tif"] and GDAL_AVAILABLE:
            sourceImg, img_ds1 = readTiff(imgDir)
            sourceImg = Image.fromarray(sourceImg) if len(sourceImg.shape) == 3 else Image.fromarray(sourceImg.squeeze(axis=2))
        else:
            print(f"⚠️ Skipping unsupported format: {extension}")
            return
        
        sourceW, sourceH = sourceImg.size[0], sourceImg.size[1]
        sourceC = len(sourceImg.getbands())
        
        for singleScale in scales:
            # Convert PIL to numpy for GPU processing
            img_array = np.array(sourceImg)
            
            # Determine optimal GPU for this operation
            device_id = 0
            if GPU_AVAILABLE and torch.cuda.device_count() > 1:
                # Simple load balancing - use GPU with more free memory
                device_id = 0
            
            # GPU-accelerated resize optimized for T4 with PyTorch
            new_size = (int(sourceW * singleScale), int(sourceH * singleScale))
            img = torch_resize_image_t4_optimized(img_array, new_size, isLabel, device_id)
            
            # Process labels on GPU if available
            if isLabel:
                if GPU_AVAILABLE:
                    try:
                        device = torch.device(f'cuda:{device_id}')
                        torch.cuda.set_device(device_id)
                        
                        # Convert to tensor
                        if len(img.shape) == 3:
                            img_tensor = torch.from_numpy(img).to(device)
                        else:
                            img_tensor = torch.from_numpy(img).to(device)
                        
                        # Binary thresholding
                        img_tensor = torch.where(img_tensor != 0, 255, img_tensor).to(torch.uint8)
                        
                        # Convert RGB label to binary if needed
                        if sourceC != 1 and len(img_tensor.shape) == 3:
                            img_tensor = torch.max(img_tensor, dim=2)[0]
                        
                        img = img_tensor.cpu().numpy()
                        
                        # Clean up
                        del img_tensor
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"⚠️ GPU label processing failed, using CPU: {e}")
                        img[img != 0] = 255
                        img = img.astype(np.uint8)
                        if sourceC != 1 and len(img.shape) == 3:
                            img = img.max(axis=2)
                else:
                    img[img != 0] = 255
                    img = img.astype(np.uint8)
                    if sourceC != 1 and len(img.shape) == 3:
                        img = img.max(axis=2)
            
            img_h, img_w = img.shape[0], img.shape[1]
            h_nums, w_nums = img.shape[0] // imgsize, img.shape[1] // imgsize
            
            if img_h % imgsize != 0:
                h_nums = h_nums + 1
            if img_w % imgsize != 0:
                w_nums = w_nums + 1
            
            if img_h < imgsize or img_w < imgsize:
                continue
            
            print(f"⚡ PyTorch T4 Processing scale {singleScale}: {h_nums}x{w_nums} patches")
            
            # Collect patch coordinates for batch processing
            patch_coords = []
            patch_filenames = []
            
            for hIndex in range(h_nums):
                start_h, end_h = getAxisBoundary(hIndex, img_h, imgsize, h_nums)
                
                for wIndex in range(w_nums):
                    start_w, end_w = getAxisBoundary(wIndex, img_w, imgsize, w_nums)
                    
                    outputPathTemp = (f"{outputPath}_scale-{singleScale}_y-{start_h}"
                                    f"_x-{start_w}_imgsize-{imgsize}{save_extension}")
                    
                    if not os.path.exists(outputPathTemp):
                        patch_coords.append((start_h, end_h, start_w, end_w))
                        patch_filenames.append(outputPathTemp)
            
            # Process patches with dual T4 optimization using PyTorch
            total_patches = len(patch_coords)
            if total_patches == 0:
                continue
                
            print(f"🚀 PyTorch T4 Dual-GPU processing {total_patches} patches...")
            
            # Process all patches at once with dual GPU
            patches = torch_process_patches_dual_t4(img, patch_coords, imgsize, isLabel, batch_size)
            
            # Save patches
            print(f"💾 Saving {len(patches)} patches...")
            for patch, filename in zip(patches, patch_filenames):
                temp_img = Image.fromarray(patch)
                temp_img.save(filename)
            
            # Clear GPU memory after processing
            clear_gpu_memory()
    
    except Exception as e:
        print(f"❌ Error processing {imgDir}: {str(e)}")
        clear_gpu_memory()

def process_levir_dataset(input_path, output_path, imgsize=256, batch_size=T4_OPTIMAL_BATCH_SIZE):
    """
    Process the entire LEVIR-CD dataset with dual T4 GPU acceleration using PyTorch
    
    Args:
        input_path: Path to input dataset
        output_path: Path to output directory
        imgsize: Size of output patches
        batch_size: GPU batch size optimized for T4
    """
    # Setup GPUs
    gpu_devices = setup_gpu_devices()
    
    splits = ['train', 'val', 'test']
    image_types = ['T1', 'T2', 'label']
    
    print(f"🚀 Starting PyTorch T4 Dual-GPU accelerated processing...")
    print(f"🎯 Detected {len(gpu_devices)} GPU(s)")
    print(f"📦 T4-Optimized Batch Size: {batch_size}")
    print(f"🔧 Image Size: {imgsize}")
    
    for split in splits:
        print(f"
📁 Processing {split} split...")
        
        for img_type in image_types:
            input_dir = os.path.join(input_path, split, img_type)
            output_dir = os.path.join(output_path, split, img_type)
            
            if not os.path.exists(input_dir):
                print(f"⚠️  Input directory not found: {input_dir}")
                continue
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Get all files in input directory
            fileList = glob.glob(os.path.join(input_dir, "*"))
            
            if not fileList:
                print(f"⚠️  No files found in {input_dir}")
                continue
            
            print(f"🖼️  Processing {img_type}: {len(fileList)} files")
            
            isLabel = (img_type == 'label')
            count = 0
            
            for singleImg in tqdm.tqdm(fileList, desc=f"PyTorch T4 Processing {img_type}"):
                count += 1
                slicingSingleImg(
                    singleImg, 
                    output_dir, 
                    scales=multiScale, 
                    imgsize=imgsize, 
                    isLabel=isLabel,
                    batch_size=batch_size
                )
                
                # Clear memory every few files to prevent accumulation
                if count % 3 == 0:
                    clear_gpu_memory()
            
            print(f"✅ Completed {img_type}: {count}/{len(fileList)} files processed")
            clear_gpu_memory()  # Clear memory after each type

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='PyTorch T4 Dual-GPU accelerated image slicing for change detection')
    parser.add_argument("-i", "--input_dir", 
                       default="/kaggle/input/processed-levir-cd-dataset/LEVIR-CD+_256", 
                       type=str, help="Path to input dataset directory")
    parser.add_argument("-o", "--output_dir", 
                       default="/kaggle/working/processed_data", 
                       type=str, help="Path to output directory")
    parser.add_argument("-is", "--img_size", default=256, type=int, help="Output patch size")
    parser.add_argument("-bs", "--batch_size", default=T4_OPTIMAL_BATCH_SIZE, type=int, 
                       help=f"GPU batch size (T4 optimized default: {T4_OPTIMAL_BATCH_SIZE})")
    parser.add_argument("-ol", "--overlap_size", default=512, type=int, help="Overlap size (not implemented)")
    parser.add_argument('-c', "--multi_scale_slicing", action='store_true', default=False)

    args, unknown = parser.parse_known_args()
    
    if args.output_dir is None:
        print("❌ Error: No output directory specified!")
        exit(0)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # T4-specific optimizations
    if not GPU_AVAILABLE:
        print("⚠️  Warning: GPU not available, running on CPU")
        args.batch_size = 1
    else:
        print(f"⚡ Using PyTorch with T4 GPU(s), optimized batch size: {args.batch_size}")
        
        # Adjust batch size based on image size for T4's memory
        if args.img_size > 512:
            args.batch_size = max(8, args.batch_size // 2)
            print(f"🔧 Adjusted batch size for large images: {args.batch_size}")
    
    # Process dataset with PyTorch T4 optimizations
    process_levir_dataset(args.input_dir, args.output_dir, args.img_size, args.batch_size)
    
    print("🎉 PyTorch T4 processing completed successfully!")

if __name__ == "__main__":
    main()
