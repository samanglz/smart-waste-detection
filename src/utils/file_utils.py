import random
from pathlib import Path 
import shutil
from configs.config import (
    RAW_IMAGES_DIR,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TEST_IMAGES_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    IMAGE_EXTENSIONS,

)



def load_directory(directory: Path,
              format : Path,
                ) -> list[Path]:
    '''
        
    Load all files with supported extensions from a directory.

    Args:
        raw_images_dir: Directory containing images.

    Returns:
        A sorted list of image paths.
    '''
    
    directory_paths = []
    for directory_path in directory.iterdir():
        if (    directory_path.is_directory()  
            and
                directory_path.suffix.lower() in format 
   ):
            directory_paths.append(directory_path)
    directory_paths = sorted(directory_paths)            
    return directory_paths




def prepare_output_directory(output_dir):
    
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    
def load_images(raw_images_dir: Path) -> list[Path]:
    '''
        
    Load all supported image files from a directory.

    Args:
        raw_images_dir: Directory containing images.

    Returns:
        A sorted list of image paths.
    '''
    
    image_paths = []
    for file_path in raw_images_dir.iterdir():
        if (    file_path.is_file()  
            and
                file_path.suffix.lower() in IMAGE_EXTENSIONS 
   ):
            image_paths.append(file_path)
    image_paths = sorted(image_paths)            
    return image_paths


def copy_images(
    image_paths: list[Path],
    destination_dir: Path,
) -> None:
    
    """
Copy images into the destination directory.

Args:
    image_paths: List of image paths.
    destination_dir: Output directory.
    """
    
    
    for image_path in image_paths:
        

        destination = destination_dir / image_path.name
        shutil.copy2(image_path, destination)
        
def shuffle_images(image_paths: list[Path], random_seed: int)-> list[Path]:
    
    
    shuffled_images = image_paths.copy()
    random.seed(random_seed)
    random.shuffle(shuffled_images)

    return shuffled_images
    