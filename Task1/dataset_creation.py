import os
from icrawler.builtin import BingImageCrawler


def download_images(keyword, num_images=200, save_dir="dataset2"):
    save_path = os.path.join(save_dir, keyword.replace(" ", "_"))
    os.makedirs(save_path, exist_ok=True)

    # Using BingImageCrawler to download images
    crawler = BingImageCrawler(storage={"root_dir": save_path})
    crawler.crawl(keyword=keyword, max_num=num_images)

    print(f"Downloaded {num_images} images for '{keyword}' to '{save_path}'")


# Example usage:
download_images("Artic_fox_animal", num_images=200, save_dir="dataset/ClassA")
download_images("Kangaroo", num_images=200, save_dir="dataset/ClassB")
