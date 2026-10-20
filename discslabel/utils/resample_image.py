# Vendored from SmaugLab-benchmark to avoid a runtime dependency.
# Upstream source:
#   https://github.com/NathanMolinier/SmaugLab-benchmark/blob/main/smaugbench/utils/resample_image.py
#
# Local addition: a `label` interpolation mode that preserves point-wise
# landmark labels (values 1..N placed at a single voxel each) across
# resampling. See `_resample_label_pointwise` in this file for the algorithm.
import argparse, textwrap
import multiprocessing as mp
from functools import partial

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import warnings
from discslabel.utils.image import Image, resample_nib

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) images and resample them to a specific resolution in mm.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            resample_image -i images -o images
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
        '--resolution', '-res', type=str, default='1x1x1',
        help='Output resolution for the images, defaults to "1x1x1".'
    )
    parser.add_argument(
        '--interpolation', '-int', type=str, default='linear',
        choices=['nn', 'linear', 'spline', 'label'],
        help=(
            'Interpolation method for resampling, defaults to "linear". '
            'Use "label" for point-wise landmark volumes (values 1..N placed '
            'at single voxels): the label is smoothed into a Gaussian sphere, '
            'resampled linearly, and the sphere center is extracted back as a '
            'point in the output grid.'
        ),
    )
    parser.add_argument(
        '--label-sigma', type=float, default=1.5,
        help='Gaussian sigma (in input voxels) used by the "label" interpolation. Defaults to 1.5.'
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
    resolution = args.resolution
    interpolation = args.interpolation
    label_sigma = args.label_sigma
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_images_path = "{output_images_path}"
            resolution = "{resolution}"
            interpolation = "{interpolation}"
            label_sigma = {label_sigma}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    resample_mp(
        images_path=images_path,
        output_images_path=output_images_path,
        resolution=resolution,
        interpolation=interpolation,
        label_sigma=label_sigma,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def resample_mp(
        images_path,
        output_images_path,
        resolution='1x1x1',
        interpolation='linear',
        label_sigma=1.5,
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
            _resample,
            overwrite=overwrite,
            resolution=resolution,
            interpolation=interpolation,
            label_sigma=label_sigma,
        ),
        image_path_list,
        output_image_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _resample(
        image_path,
        output_image_path,
        overwrite=False,
        resolution='1x1x1',
        interpolation='linear',
        label_sigma=1.5,
    ):
    '''
    Resample the image to the specified resolution.
    '''
    image_path = Path(image_path)
    output_image_path = Path(output_image_path)

    # Check if the output image already exists
    if not overwrite and output_image_path.exists():
        return

    image = Image(str(image_path))

    # Resample the image to the specified resolution
    resolution_tuple = tuple(map(float, resolution.split('x')))
    if interpolation == 'label':
        image = _resample_label_pointwise(
            image,
            new_size=resolution_tuple,
            new_size_type='mm',
            sigma=label_sigma,
        )
    else:
        image = resample_nib(
            image,
            new_size=resolution_tuple,
            new_size_type='mm',
            interpolation=interpolation,
            verbose=False,
        )

    # Make sure output directory exists and save the image
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_image_path), verbose=False)


def _resample_label_pointwise(image, new_size, new_size_type, sigma=1.5):
    """Resample a point-wise label volume by Gaussian-sphere smoothing then peak picking.

    Point-wise labels are a few isolated voxels (values 1..N, one voxel per
    landmark); nearest-neighbour resampling to a different grid can shift or
    even delete those voxels because the input and output grids don't line up
    through the single lit voxel. This routine, per label value:

      1. Isolates the binary mask of that label.
      2. Smooths it with a Gaussian (``sigma`` in input voxels) so the lit
         voxel becomes a small sphere with a well-defined peak at its centre.
      3. Resamples the sphere with linear interpolation to the target grid.
      4. Takes the ``argmax`` of the resampled sphere — that voxel is the
         closest-to-centre point on the output grid, which becomes the new
         single-voxel location of the label.

    Nothing is interpolated between labels, so disc identities stay intact.
    """
    data = np.rint(image.data).astype(np.int32)
    affine = image.hdr.get_best_affine()

    labels = sorted(int(v) for v in np.unique(data) if v != 0)

    # Determine the target shape and affine by resampling a dummy volume.
    probe = nib.nifti1.Nifti1Image(
        np.zeros_like(data, dtype=np.float32), affine, image.hdr
    )
    probe_r = resample_nib(
        probe,
        new_size=new_size,
        new_size_type=new_size_type,
        interpolation='linear',
        verbose=False,
    )
    out_shape = probe_r.shape
    out_affine = probe_r.affine
    out_data = np.zeros(out_shape, dtype=np.int32)

    for label in labels:
        mask = (data == label).astype(np.float32)
        if mask.sum() == 0:
            continue
        smoothed = gaussian_filter(mask, sigma=sigma)
        smoothed_img = nib.nifti1.Nifti1Image(smoothed, affine, image.hdr)
        resampled = resample_nib(
            smoothed_img,
            new_size=new_size,
            new_size_type=new_size_type,
            interpolation='linear',
            verbose=False,
        )
        arr = np.asarray(resampled.get_fdata())
        if arr.size == 0 or float(arr.max()) <= 0.0:
            # The sphere fell outside the resampled grid; skip.
            continue
        peak = np.unravel_index(int(np.argmax(arr)), arr.shape)
        out_data[peak] = label

    out_img = nib.nifti1.Nifti1Image(out_data.astype(np.int16), out_affine, probe_r.header)
    return Image(
        np.asanyarray(out_img.dataobj),
        hdr=out_img.header,
        orientation=image.orientation,
        dim=out_img.header.get_data_shape(),
    )

if __name__ == '__main__':
    main()