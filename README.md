# Text dataset generator

Installation of packages needed for running:

``pip install PILLOW, numpy, freetype-py, opencv-python, noise, lxml``
or run command
``pip install -r requirements.txt``

How to run the script:  
``text_dataset_generator.py [-h] -c CONFIG``  

Example:  
``python text_dataset_generator.py -c config.ini``

## Image examples
Here are shown examples of generated images. These are just strips, originals with full resolution are located at [Doc/ImageExamples](Doc/ImageExamples).
#### Generated image 
![script output image](Doc/v3_image.png)
#### Generated image with bounding boxes
![script output image with annotations](Doc/v3_image_annotated.png)
#### Semantic data associted with generated image (Grayscale)
![script output image semantic](Doc/v3_image_semantic.png)
#### Semantic data associted with generated image (Colored)
![script output image semantic colored](Doc/v3_image_semantic_colored.png)

### Image metadata
Example pagexml output is located at [Doc/pagexml.xml](Doc/pagexml.xml)

Example annotations output is located at [Doc/annotations.txt](Doc/annotations.txt)
