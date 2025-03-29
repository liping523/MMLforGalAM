## 1. 逐个下载（速度慢）
#import skimage.io
#import pandas as pd
#import time
#import sys
#import os
#import urllib
#import requests

## stuff to make the status pretty.
#class Printer():
#    """Print things to stdout on one line dynamically"""

#    def __init__(self, data):
#        sys.stdout.write("\r\x1b[K" + data.__str__())
#        sys.stdout.flush()

#def cmdline():
#    ''' Controls the command line argument handling for this little program.
#    '''

#    from optparse import OptionParser

#    # read in the cmd line arguments
#    USAGE = 'usage:\t %prog [options]\n'
#    parser = OptionParser(usage=USAGE)

#    # add options
#    parser.add_option('--output',
#                      dest='output',
#                      default='images_20w',
#                      help='Path to save image data')
#    parser.add_option('--width',
#                      dest='width',
#                      default=128,
#                      help='Default width of images')
#    parser.add_option('--height',
#                      dest='height',
#                      default=128,
#                      help='Default height of images')
#    parser.add_option('--cat',
#                       dest='cat',
#                       default='0data_all_mpa_new.csv',
#                       help='Catalog to get image names from.')
#    parser.add_option('--scale',
#                      dest='scale',
#                      action='store_true',
#                      default=True,
#                      help=('Whether or not to rescale the images to the same '
#                            'physical size. This is done by looking at the '
#                            ' Petrosian radius from the SDSS. If False, the '
#                            'images are of 15" x 15"'))

#    (options, args) = parser.parse_args(args=[])
#    return options, args

#def download_image(url, save_path):
#    """Function to download image from URL and save it."""
#    try:
#        response = requests.get(url)
#        if response.status_code == 200:
#            with open(save_path, "wb") as f:
#                f.write(response.content)
#            return True
#        else:
#            return False
#    except requests.exceptions.RequestException as e:
#        return False

#def main():

#    opt, arg = cmdline()

#    # load the data
#    df = pd.read_csv(opt.cat)

#    width = 128
#    height = 128
#    pixelsize = 0.396  # ''/pixel

#    # remove trailing slash in output path if it's there.
#    opt.output = opt.output.rstrip('\/')

#    # total number of images
#    n_gals = df.shape[0]

#    # Create/open text files to log existing and failed downloads
#    with open('existing_images.txt', 'a') as existing_file, open('failed_downloads.txt', 'a') as failed_file:
#        for row in df.itertuples():
#            scale = 2 * row.petroR90_r / pixelsize / width
#            url = ("http://skyserver.sdss.org/dr18/SkyserverWS/ImgCutout/getjpeg"
#                   "?ra={}"
#                   "&dec={}"
#                   "&scale={}"
#                   "&width={}"
#                   "&height={}".format(row.ra_1, row.dec_1, scale, width, height))

#            image_filename = f'{opt.output}/image-{str(int(row.plateid)).zfill(4)}-{row.mjd}-{str(int(row.fiberid)).zfill(4)}.jpg'

#            if os.path.isfile(image_filename):
#                # Log the existing image to the text file
#                existing_file.write(image_filename + '\n')
#                continue

#            # Try to download the image
#            success = download_image(url, image_filename)
#            if not success:
#                # Log the failed download to the text file
#                failed_file.write(url + '\n')
#                print(f"Download failed for: {url}")
#            else:
#                time.sleep(0.05)

#            # Print progress
#            current = row.Index / n_gals * 100
#            status = "{:.3f}% of {} completed.".format(current, n_gals)
#            Printer(status)

#    print('')

#if __name__ == "__main__":
#    main()


# 2. 并行下载（速度快）
import skimage.io
import pandas as pd
import time
import sys
import os
import urllib
import requests
import concurrent.futures
import psutil

# stuff to make the status pretty.
class Printer():
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()

def cmdline():
    ''' Controls the command line argument handling for this little program.
    '''

    from optparse import OptionParser

    # read in the cmd line arguments
    USAGE = 'usage:\t %prog [options]\n'
    parser = OptionParser(usage=USAGE)

    # add options
    parser.add_option('--output',
                      dest='output',
                      default='images_5w_256x256_scale2',
                      help='Path to save image data')
    parser.add_option('--width',
                      dest='width',
                      default=256,
                      help='Default width of images')
    parser.add_option('--height',
                      dest='height',
                      default=256,
                      help='Default height of images')
    parser.add_option('--cat',
                       dest='cat',
                       default='../5w_data.csv',
                       help='Catalog to get image names from.')
    parser.add_option('--scale',
                      dest='scale',
                      action='store_true',
                      default=True,
                      help=('Whether or not to rescale the images to the same '
                            'physical size. This is done by looking at the '
                            ' Petrosian radius from the SDSS. If False, the '
                            'images are of 15" x 15"'))

    (options, args) = parser.parse_args(args=[])
    return options, args

def download_image(url, save_path):
    """Function to download image from URL and save it."""
    try:
        if os.path.isfile(save_path):
            return True
        with requests.Session() as session:
            response = session.get(url, timeout=60)  # 设置超时时间为10秒
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                return False
    except requests.exceptions.RequestException as e:
        return False

def get_optimal_workers():
    """Dynamically adjust the number of concurrent workers based on system resources (CPU and memory)."""
    cpu_usage = psutil.cpu_percent(interval=1)  # 获取当前 CPU 使用率
    memory_usage = psutil.virtual_memory().percent  # 获取当前内存使用情况

    # 假设 CPU 使用率低于 80% 和内存使用低于 80% 时可以增加线程数
    if cpu_usage < 80 and memory_usage < 80:
        return 80  
    else:
        return 20  

def download_images_concurrently(df, opt):
    """Function to download images concurrently using ThreadPoolExecutor."""
    width = 256
    height = 256
    pixelsize = 0.396  # ''/pixel

    # Create/open text files to log existing and failed downloads
    with open('existing_images_256x256.txt', 'a') as existing_file, open('failed_downloads_256x256.txt', 'a') as failed_file:
        # Get optimal number of workers based on system resources
        max_workers = get_optimal_workers()
        print(f"Using {max_workers} concurrent threads for downloading.")

        # Create a ThreadPoolExecutor for concurrent downloading
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {}

            # Submit the download tasks to the executor
            for row in df.itertuples():
#                scale = 2 * row.petroR90_r / pixelsize / width
#                scale = (2.5 * row.petroR90_r) / width
                scale = 0.4

                url = ("http://skyserver.sdss.org/dr18/SkyserverWS/ImgCutout/getjpeg"
                       "?ra={}"
                       "&dec={}"
                       "&scale={}"
                       "&width={}"
                       "&height={}".format(row.ra, row.dec, scale, width, height))

                image_filename = f'{opt.output}/image-{str(int(row.plateid)).zfill(4)}-{row.mjd}-{str(int(row.fiberid)).zfill(4)}.jpg'

                if os.path.isfile(image_filename):
                    # Log the existing image to the text file
                    existing_file.write(image_filename + '\n')
                    continue

                # Submit the download task to the thread pool
                future = executor.submit(download_image, url, image_filename)
                future_to_url[future] = url

            # Monitor progress and handle completion
            n_gals = df.shape[0]
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_url)):
                url = future_to_url[future]
                if not future.result():
                    # Log the failed download to the text file
                    failed_file.write(url + '\n')
                    print(f"Download failed for: {url}")

                # Print progress
                current = (idx + 1) / n_gals * 100
                status = "{:.3f}% of {} completed.".format(current, n_gals)
                Printer(status)

        print('')

def main():
    opt, arg = cmdline()

    # Load the data
    df = pd.read_csv(opt.cat)

    # Remove trailing slash in output path if it's there
    opt.output = opt.output.rstrip('\/')

    # Start the concurrent download process
    download_images_concurrently(df, opt)

if __name__ == "__main__":
    main()


