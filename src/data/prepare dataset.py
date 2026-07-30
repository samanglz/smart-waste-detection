import os
import pathlib
from pathlib import Path
import shutil


ORIGINAL_DIR = Path("Desktop/smart-waste-detection/data/original") 
RAW_IMAGES_DIR = Path("Desktop/smart-waste-detection/data/raw/images")
IMG_EXTENSIONS = { ".jpg",
                  ".png",
                  ".jpeg",
                  ".bmp"
            
    
    }
# RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
# class_dirs = [d for d in ORIGINAL_DIR.iterdir() if d.is_dir()]
# for class_dir in class_dirs:
#     images = list(class_dir.glob("*.jpg"))
    
#     for img_path in images:
#         new_name= f"img_{counter:06d}.jpg"
#         destination = RAW_IMAGES_DIR/new_name
#         shutil.copy2(img_path, destination)
#         counter+=1
        
        
def prepare_output_directory(output_dir):
    
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    
   
def get_class_directories(original_dir):
    class_dirs = [d for d in original_dir.iterdir() if d.is_dir()]   
    return class_dirs
        


def generate_image_name(counter,suffix):
    new_name= f"img_{counter:06d}{suffix}"
    return new_name
    
    
def copy_and_rename_images(class_dirs, raw_images_dir):

    
    counter = 1 


    for class_dir in class_dirs:
        for file_path in class_dir.iterdir():
            
            if( file_path.is_file()
               and 
               file_path.suffix.lower() in IMG_EXTENSIONS
):
                new_name = generate_image_name(counter, file_path.suffix.lower())
                destination = raw_images_dir / new_name
            try:
                shutil.copy2(file_path, destination)
                counter += 1

            except Exception as e:
                print(f"Failed to copy {file_path.name}: {e}")

    return counter - 1
def main():
    prepare_output_directory(RAW_IMAGES_DIR)
    class_dirs = get_class_directories(ORIGINAL_DIR)
    copied_images = copy_and_rename_images(class_dirs, RAW_IMAGES_DIR)
    print(f"Copied {copied_images} images.")
    
if __name__ == "__main__":
    main()    
    
    
    
    
    
    
    
    