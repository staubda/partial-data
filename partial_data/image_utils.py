import os
import requests
from multiprocessing.dummy import Pool
from functools import partial

import tqdm


def download_image(image_url, output_dir):
    img_name = os.path.split(image_url)[-1]
    output_filepath = os.path.join(output_dir, img_name)
    try:
        if not os.path.exists(output_filepath):
            img_data = requests.get(image_url).content
            with open(output_filepath, 'wb') as fp:
                fp.write(img_data)
    except:
        return image_url
    else:
        return None


def download_images(image_urls, output_dir, num_parallel=1):
    with Pool(processes=num_parallel) as pool:
        res = list(tqdm.tqdm(
            pool.imap(partial(download_image, output_dir=output_dir), image_urls),
            total=len(image_urls)
        ))
    return res
