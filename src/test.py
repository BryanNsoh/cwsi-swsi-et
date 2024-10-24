import requests
import zipfile
import os

# List of image URLs
image_urls = [
    "https://i.ibb.co/PFXvfv2/Field-Layout.png",
    "https://i.ibb.co/GpB939X/datalogger-setup.png",
    "https://i.ibb.co/m9HnmFC/full-system-diagram.png",
    "https://i.ibb.co/8jjD4SZ/irrigation-dashboard-demonstration-section2-7-annotated.png",
    "https://i.ibb.co/pPMbvkf/cloud-function-interactions.png",
    "https://i.ibb.co/dmwskDr/assignment-mechanism.png",
    "https://i.ibb.co/gSRWwgh/battery-voltage-trends-on-outage-event.png",
    "https://i.ibb.co/x53KYGt/sample-irrigation-data-for-demo.png"
]

# Directory to store the downloaded images (current directory)
image_dir = os.getcwd()

# Download the images
image_paths = []
for i, url in enumerate(image_urls):
    image_path = os.path.join(image_dir, f'image_{i+1}.png')
    response = requests.get(url)
    with open(image_path, 'wb') as file:
        file.write(response.content)
    image_paths.append(image_path)

# Create a zip file
zip_path = os.path.join(image_dir, 'images.zip')
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for image_path in image_paths:
        zipf.write(image_path, os.path.basename(image_path))

print(f"All images downloaded and zipped at {zip_path}")
