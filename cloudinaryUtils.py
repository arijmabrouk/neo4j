import cloudinary
from dotenv import load_dotenv
import os
import cloudinary.uploader

load_dotenv()

# Configure Cloudinary with your credentials from environment variables
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)



def upload_tocloudinary(file_path):
    response = cloudinary.uploader.upload(file_path, resource_type="raw")
    print(response)
    url = response['secure_url']

    return url
