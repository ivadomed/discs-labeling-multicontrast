# Vendored from SmaugLab-benchmark to avoid a runtime dependency.
# Upstream source:
#   https://github.com/NathanMolinier/SmaugLab-benchmark/blob/main/smaugbench/utils/reorient_image.py
import argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import warnings
from discslabel.utils.image import Image

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) images and reorient them to a specific orientation.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            reorient_image -i images -o images
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--output-images-dir', '-o', type=Path, required=True,
        help='The folder where output images will be saved (required).'
    )
    parser.add_argument(
        '--orientation', '-ori', type=str, default='RPI',
        help='Output orientation for the images, defaults to "RPI".'
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='Overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    images_path = args.images_dir
    output_images_path = args.output_images_dir
    orientation = args.orientation
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_images_path = "{output_images_path}"
            orientation = "{orientation}"
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    reorient_mp(
        images_path=images_path,
        output_images_path=output_images_path,
        orientation=orientation,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def reorient_mp(
        images_path,
        output_images_path,
        orientation='RPI',
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    output_images_path = Path(output_images_path)

    glob_pattern = '*.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    output_image_path_list = [output_images_path / _.relative_to(images_path).parent / _.name for _ in image_path_list]

    process_map(
        partial(
            _reorient,
            overwrite=overwrite,
            orientation=orientation
        ),
        image_path_list,
        output_image_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _reorient(
        image_path,
        output_image_path,
        overwrite=False,
        orientation='RPI'
    ):
    '''
    Reorient the image to the input orientation.
    '''
    image_path = Path(image_path)
    output_image_path = Path(output_image_path)

    # Check if the output image already exists
    if not overwrite and output_image_path.exists():
        return

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f'Input image file {image_path} does not exist.')

    image = Image(str(image_path))

    # Reorient the image to the specified orientation using SCT's convention
    # https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4419
    image.change_orientation(orientation)

    # Make sure output directory exists and save the image
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_image_path), verbose=False)

if __name__ == '__main__':
    main()