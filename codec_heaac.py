import os
import subprocess

def process_files(org_dir, save_compressed_basedir, save_wav_basedir, codec, bitrate_list):
    """Processes files in the specified directory and its subdirectories."""
    print(org_dir)
    for root, dirs, files in os.walk(org_dir):
        for file_name in files:
            if file_name.endswith(('.wav')):
                relative_path = os.path.relpath(root, org_dir)
                file_basename = os.path.splitext(file_name)[0]
                input_file = os.path.join(root, file_name)
                
                # Convert stereo to mono if necessary (disabled for now)
                # mono_file = os.path.join(root, f"{file_basename}_mono.wav")
                # convert_to_mono(input_file, mono_file)
                
                for bitrate in bitrate_list:
                    # Set up paths for compressed and wav output
                    compressed_dir = os.path.join(save_compressed_basedir, relative_path)
                    wav_dir = os.path.join(save_wav_basedir, relative_path)
                    
                    # Ensure directories exist
                    os.makedirs(compressed_dir, exist_ok=True)
                    os.makedirs(wav_dir, exist_ok=True)
                    
                    compressed_file = os.path.join(compressed_dir, f"{file_basename}.{get_extension(codec)}")
                    output_wav_file = os.path.join(wav_dir, f"{file_basename}.wav")
                    
                    # Compress the file
                    compress_command = get_compress_command(input_file, compressed_file, codec, bitrate)
                    subprocess.run(compress_command, shell=True)
                    
                    # Convert compressed file back to WAV
                    # convert_command = get_convert_command(compressed_file, output_wav_file)
                    # subprocess.run(convert_command, shell=True)

def get_extension(codec):
    """Returns the appropriate file extension based on the codec."""
    if codec == 'mp3':
        return 'mp3'
    else:
        return 'm4a'

def get_compress_command(input_file, output_file, codec, bitrate):
    """Returns the appropriate compression command based on the codec."""
    input_file = f'"{input_file}"'  
    output_file = f'"{output_file}"'  

    if codec == 'mp3':
        return f'ffmpeg -i {input_file} -codec:a libmp3lame -b:a {bitrate}k {output_file}'
    elif codec == 'aac_lc':
        return f'ffmpeg -i {input_file} -codec:a aac -b:a {bitrate}k {output_file}'
    elif codec == 'fdk_aac_lc':
        return f'ffmpeg -i {input_file} -codec:a libfdk_aac -b:a {bitrate}k {output_file}'
    elif codec == 'fdk_aac_he1':
        return f'ffmpeg -i {input_file} -codec:a libfdk_aac -profile:a aac_he -ac 1 -b:a {bitrate}k {output_file}'
    else:
        raise ValueError("Unsupported codec")

def get_convert_command(input_file, output_file):
    """Returns the command to convert compressed file back to WAV format."""
    input_file = f'"{input_file}"'  
    output_file = f'"{output_file}"'  

    return f'ffmpeg -i {input_file} -ac 1 {output_file}'

## 이게 찐
# # MUSDB
# org_dir = "/mnt/hdd/Dataset/MUSDB18_HQ_mono_48kHz"
# save_compressed_basedir = "/home/woongjib/Projects/Dataset/MUSDB_CORE_M4a"
# save_wav_basedir = "/home/woongjib/Projects/SBR/aac_analysis/temptemptemp"

org_dir = "/home/woongjib/Projects/Dataset_BESSL/USAC44_mono_48k"
save_compressed_basedir = "/home/woongjib/Projects/SBR/aac_analysis/samples"
save_wav_basedir = "/home/woongjib/Projects/SBR/aac_analysis/temptemptemp"

codec = 'fdk_aac_he1'  # Codec to use for compression
bitrate_list = [16]  # List of bitrates to use
process_files(org_dir, save_compressed_basedir, save_wav_basedir, codec, bitrate_list)

