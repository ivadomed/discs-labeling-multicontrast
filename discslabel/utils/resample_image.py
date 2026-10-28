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
        help=(
            'Gaussian sigma IN MILLIMETRES used by the "label" interpolation. '
            'Converted to per-axis voxel sigmas via the input image zooms, so '
            'the sphere stays isotropic in physical space even on thick-slice '
            'MRI. Defaults to 1.5 mm.'
        ),
    )
    parser.add_argument(
        '--debug-dir', type=Path, default=None,
        help=(
            'Only used with -int label. If set, writes a per-file PNG into '
            'this folder showing (input points, smoothed spheres on the input '
            'grid, resampled spheres on the output grid, peak-picked output '
            'points) on the mid-sagittal slice — use it to confirm '
            '--label-sigma is big enough that each sphere survives the '
            'resample as a well-defined peak.'
        ),
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
    debug_dir = args.debug_dir
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
            label_sigma = {label_sigma} mm
            debug_dir = {debug_dir}
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
        debug_dir=debug_dir,
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
        debug_dir=None,
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

    if debug_dir is not None:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    process_map(
        partial(
            _resample,
            overwrite=overwrite,
            resolution=resolution,
            interpolation=interpolation,
            label_sigma=label_sigma,
            debug_dir=debug_dir,
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
        debug_dir=None,
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
        debug_path = None
        if debug_dir is not None:
            debug_path = Path(debug_dir) / (_strip_nii_ext(image_path.name) + '_debug.png')
        image = _resample_label_pointwise(
            image,
            new_size=resolution_tuple,
            new_size_type='mm',
            sigma_mm=label_sigma,
            debug_path=debug_path,
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


def _strip_nii_ext(name: str) -> str:
    for ext in ('.nii.gz', '.nii'):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _resample_label_pointwise(image, new_size=None, new_size_type=None, sigma_mm=1.5, debug_path=None, image_dest=None):
    """Resample a point-wise label volume by Gaussian-sphere smoothing then peak picking.

    Point-wise labels are a few isolated voxels (values 1..N, one voxel per
    landmark); nearest-neighbour resampling to a different grid can shift or
    even delete those voxels because the input and output grids don't line up
    through the single lit voxel. This routine, per label value:

      1. Isolates the binary mask of that label.
      2. Smooths it with a Gaussian **in physical space** so the lit voxel
         becomes a sphere with a well-defined peak at its centre.
         ``sigma_mm`` is given in millimetres and converted to per-axis voxel
         sigmas via the input image zooms — on thick-slice MRI (anisotropic
         voxels) this keeps the sphere isotropic in mm rather than warping
         it into an ellipsoid along the through-slice direction.
      3. Resamples the sphere with linear interpolation to the target grid.
      4. Takes the ``argmax`` of the resampled sphere — that voxel is the
         closest-to-centre point on the output grid, which becomes the new
         single-voxel location of the label.

    Nothing is interpolated between labels, so disc identities stay intact.

    When ``debug_path`` is set, a 4-panel PNG of the mid-sagittal slice is
    written there: the user can confirm visually that the sphere is big
    enough that its peak survives the resample. The four panels are:
        1. input point-wise labels
        2. union of all smoothed spheres on the INPUT grid
        3. union of all resampled smoothed spheres on the OUTPUT grid
        4. output point-wise labels (post peak-pick)
    """
    data = np.rint(image.data).astype(np.int32)
    affine = image.hdr.get_best_affine()

    # mm → per-axis voxel sigma: keeps the sphere isotropic in physical space.
    zooms = image.hdr.get_zooms()[:3]
    sigma_vox_in = tuple(float(sigma_mm) / float(z) for z in zooms)

    labels = sorted(int(v) for v in np.unique(data) if v != 0)

    # Determine the target shape and affine by resampling a dummy volume.
    probe = nib.nifti1.Nifti1Image(
        np.zeros_like(data, dtype=np.float32), affine, image.hdr
    )
    probe_r = resample_nib(
        probe,
        new_size=new_size,
        new_size_type=new_size_type,
        image_dest=image_dest,
        interpolation='linear',
        verbose=False,
    )
    out_shape = probe_r.shape
    out_affine = probe_r.affine
    out_data = np.zeros(out_shape, dtype=np.int32)

    # Union buffers only built when debug_path is set.
    smoothed_union_in = np.zeros(data.shape, dtype=np.float32) if debug_path else None
    smoothed_union_out = np.zeros(out_shape, dtype=np.float32) if debug_path else None

    for label in labels:
        mask = (data == label).astype(np.float32)
        if mask.sum() == 0:
            continue
        smoothed = gaussian_filter(mask, sigma=sigma_vox_in)
        smoothed_img = nib.nifti1.Nifti1Image(smoothed, affine, image.hdr)
        resampled = resample_nib(
            smoothed_img,
            new_size=new_size,
            new_size_type=new_size_type,
            image_dest=image_dest,
            interpolation='linear',
            verbose=False,
        )
        arr = np.asarray(resampled.get_fdata())
        if arr.size == 0 or float(arr.max()) <= 0.0:
            # The sphere fell outside the resampled grid; skip.
            continue
        peak = np.unravel_index(int(np.argmax(arr)), arr.shape)
        out_data[peak] = label

        if debug_path is not None:
            # Normalise each sphere to its own peak so small labels stay visible
            # when overlaid alongside bigger ones.
            s_max = float(smoothed.max())
            if s_max > 0:
                smoothed_union_in[:] = np.maximum(smoothed_union_in, smoothed / s_max)
            r_max = float(arr.max())
            if r_max > 0:
                smoothed_union_out[:] = np.maximum(smoothed_union_out, arr / r_max)

    if debug_path is not None:
        _save_label_debug_png(
            debug_path,
            input_labels=data,
            smoothed_in=smoothed_union_in,
            smoothed_out=smoothed_union_out,
            output_labels=out_data,
            input_zooms=zooms,
            output_zooms=tuple(probe_r.header.get_zooms()[:3]),
            sigma_mm=float(sigma_mm),
        )

    # Match the header dtype to the output array (int16) BEFORE building the
    # Nifti1Image, so `Image.save` doesn't log a header/dtype mismatch warning.
    # Copy the header so we don't mutate the probe object.
    out_header = probe_r.header.copy()
    out_header.set_data_dtype(np.int16)
    out_data_i16 = out_data.astype(np.int16)
    out_img = nib.nifti1.Nifti1Image(out_data_i16, out_affine, out_header)
    return Image(
        np.asanyarray(out_img.dataobj),
        hdr=out_img.header,
        orientation=image.orientation,
        dim=out_img.header.get_data_shape(),
    )


def _save_label_debug_png(
    debug_path,
    input_labels,
    smoothed_in,
    smoothed_out,
    output_labels,
    input_zooms,
    output_zooms,
    sigma_mm,
):
    """Render a 4-panel mid-sagittal debug PNG for the label resample.

    Mid-sagittal = middle slice along axis 0, under the assumption the input
    was reoriented so axis 0 is R-L (RPI / RAS / RSP / RSA etc.).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[resample label] matplotlib unavailable; skipping debug PNG for {debug_path}")
        return

    debug_path = Path(debug_path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    def _mid(vol: np.ndarray) -> np.ndarray:
        mid = vol.shape[0] // 2
        return vol[mid, :, :].T  # transpose so rows=axis-2, cols=axis-1 for display

    def _points(labels_vol: np.ndarray):
        pts = np.argwhere(labels_vol != 0)
        if pts.size == 0:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))
        xs = pts[:, 1]  # axis-1 coordinate (horizontal in the transposed display)
        ys = pts[:, 2]  # axis-2 coordinate (vertical, but we flip below)
        vals = labels_vol[pts[:, 0], pts[:, 1], pts[:, 2]]
        return xs, ys, vals

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    titles = [
        f"input labels\nzooms {tuple(round(z, 3) for z in input_zooms)} mm",
        f"smoothed spheres (input grid)\nsigma = {sigma_mm:g} mm",
        f"resampled spheres (output grid)\nzooms {tuple(round(z, 3) for z in output_zooms)} mm",
        "output labels (peak-picked)",
    ]
    panels = [
        (np.zeros_like(_mid(input_labels), dtype=np.float32), input_labels, "Blues"),
        (_mid(smoothed_in), None, "magma"),
        (_mid(smoothed_out), None, "magma"),
        (np.zeros_like(_mid(output_labels), dtype=np.float32), output_labels, "Blues"),
    ]
    for ax, title, (bg, label_vol, cmap) in zip(axes, titles, panels):
        ax.imshow(bg, origin="lower", cmap=cmap)
        if label_vol is not None:
            xs, ys, vals = _points(label_vol)
            if xs.size:
                sc = ax.scatter(xs, ys, c=vals, s=24, cmap="tab20", edgecolors="k", linewidths=0.5)
                for x, y, v in zip(xs, ys, vals):
                    ax.text(x + 1, y + 1, str(int(v)), color="white", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"label resample debug — {debug_path.stem}", fontsize=11)
    fig.tight_layout()
    fig.savefig(debug_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    # Quick visual check: run this file directly to resample one label volume
    # and drop the debug PNG next to it, so you can tune --label-sigma without
    # driving the whole prepare pipeline. Flip to `main()` for the normal CLI.
    image_path = Path("/Users/nathan/data/whole-spine/derivatives/labels/sub-unfErssm003/anat/sub-unfErssm003_T1w_labels-disc-manual.nii.gz")
    new_size = (1.0, 1.0, 1.0)
    new_size_type = "mm"
    image = Image(str(image_path)).change_orientation("RSP")
    debug_png = image_path.with_name(_strip_nii_ext(image_path.name) + "_debug.png")
    new_image = _resample_label_pointwise(
        image,
        new_size,
        new_size_type,
        sigma_mm=1.5,
        debug_path=debug_png,
    )
    print(f"debug PNG written to {debug_png}")
    #main()