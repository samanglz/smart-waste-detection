import shutil
from pathlib import Path 
from src.utils.file_utils import (shuffle_images,copy_images,load_images,)
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



def prepare_directories(    train_images_dir: Path,
    val_images_dir: Path,
    test_images_dir:Path,)-> None:
    
    for directory in [train_images_dir, val_images_dir, test_images_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents= True, exist_ok=True )


        
def split_images(
    image_paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,  
) -> tuple[list[Path], list[Path], list[Path]]:
    
    train_size =  int(len(image_paths)* train_ratio) 
    val_size = int(len(image_paths)* val_ratio)
#    test_size = len(image_paths) - train_size - val_size
     
    train_images = image_paths[:train_size]
    val_images = image_paths[train_size:train_size+val_size]
    test_images = image_paths[train_size+ val_size:]
    
    
    return train_images, val_images, test_images
    

def split_for_auto_annotation(image_Paths, splitratio):
    split_size = int(len(image_Paths) * splitratio)
    splited_dataset = image_Paths[:split_size]
    return splited_dataset

def main():

    prepare_directories(TRAIN_IMAGES_DIR,VAL_IMAGES_DIR,TEST_IMAGES_DIR)

    image_paths = load_images(RAW_IMAGES_DIR)

    shuffled_images = shuffle_images(image_paths,RANDOM_SEED)

    train_images, val_images, test_images = split_images(shuffled_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    copy_images(train_images, TRAIN_IMAGES_DIR)
    copy_images(val_images, VAL_IMAGES_DIR)
    copy_images(test_images, TEST_IMAGES_DIR)



if __name__ == "__main__":
    main()
   