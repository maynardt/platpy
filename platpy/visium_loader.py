import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy.sparse as sp
import warnings

# Local imports
from .channels import SampleSet, ImageChannel, OmicChannel
from .transform_utils import create_transform_matrix, SpatialMetadata
from .io_utils import build_spatial_block_index


def aggregate_bins_2to4(adata_2um):
    """
    Aggregate 2µm Visium HD bins to 4µm by summing 2x2 bin neighborhoods.

    2µm bin centers fall at 1, 3, 5, 7... µm. This snaps each center to its
    4µm parent (centered at 2, 6, 10... µm), then sums the sparse count matrix
    across each group. Total counts are conserved exactly.

    Returns a new AnnData with 4µm bin centers in obsm['spatial'] and
    uns['bin_size_um'] = 4.0.
    """
    coords = adata_2um.obsm['spatial']  # (N, 2), microns

    # Integer grid index of each 2µm bin's 4µm parent
    ix = np.floor(coords[:, 0] / 4.0).astype(np.int64)
    iy = np.floor(coords[:, 1] / 4.0).astype(np.int64)
    grid_idx = np.stack([ix, iy], axis=1)  # (N, 2)

    unique_grid, inverse = np.unique(grid_idx, axis=0, return_inverse=True)
    n_out = len(unique_grid)

    # Aggregation matrix: row i accumulates all 2µm bins that map to 4µm bin i
    agg = sp.csr_matrix(
        (np.ones(len(inverse), dtype=np.float32), (inverse, np.arange(len(inverse)))),
        shape=(n_out, adata_2um.n_obs)
    )
    X_agg = agg @ adata_2um.X  # (n_out, n_genes)

    # 4µm bin centers: integer grid position * 4 + 2 recovers the bin center
    new_x = unique_grid[:, 0] * 4.0 + 2.0
    new_y = unique_grid[:, 1] * 4.0 + 2.0
    new_coords = np.stack([new_x, new_y], axis=1)

    obs_names = [f"{x:.1f}_{y:.1f}" for x, y in new_coords]
    obs_df = pd.DataFrame(index=obs_names)

    adata_4um = anndata.AnnData(X=X_agg, obs=obs_df, var=adata_2um.var.copy())
    adata_4um.obsm['spatial'] = new_coords
    adata_4um.obs['total_counts'] = np.array(X_agg.sum(axis=1)).flatten()
    adata_4um.uns['bin_size_um'] = 4.0

    return adata_4um


def load_visium_directory(
    base_dir: str,
    name: str = "VisiumSample",
    bin_sizes=None,
    image_types=None,
    omic_name: str = "spots",
    debug=True
):
    if debug: print(f"Loading Visium sample '{name}' from: {base_dir}")
    # Default bins: 8um and 16um. The 2um bin is skipped by default — it is too
    # artifactual for most analyses. Pass bin_sizes=[2, 8, 16] to load it explicitly.
    if bin_sizes is None: bin_sizes = [8, 16]
    if image_types is None: image_types = ['hires', 'lowres', 'detected']

    sample = SampleSet(name=name)                                              #type: ignore
    sample.metadata['scalefactors'] = {} # Initialize universal metadata dict

    image_map = {
        'hires': {'file': 'tissue_hires_image.png', 'scale_key': 'tissue_hires_scalef'},
        'lowres': {'file': 'tissue_lowres_image.png', 'scale_key': 'tissue_lowres_scalef'},
        'detected': {'file': 'detected_tissue_image.jpg', 'scale_key': 'regist_target_img_scalef'}
    }

    # Track these globally so we can load images once
    global_mpp = None
    global_offset_x, global_offset_y = 0.0, 0.0
    global_scales = {}

    # Accumulate all bin levels here; assembled into a single OmicChannel after the loop.
    omic_pyramid = {}      # {mpp_float: adata}
    omic_spatial = None
    omic_geometry = "square"

    # =========================================================================
    # PHASE 1: Load Omics & Determine Global Coordinate System
    # =========================================================================
    for bin_sz in bin_sizes:
        bin_folder = f"square_{bin_sz:03d}um"
        bin_path = os.path.join(base_dir, "binned_outputs", bin_folder)
        if not os.path.exists(bin_path): continue

        scale_json_path = os.path.join(bin_path, "spatial", "scalefactors_json.json")
        if not os.path.exists(scale_json_path): continue

        with open(scale_json_path, 'r') as f:
            scales = json.load(f)

        mpp = scales["microns_per_pixel"]
        actual_bin_um = scales.get("bin_size_um", float(bin_sz))

        # Set the global MPP for the whole sample using the generic metadata dict.
        if global_mpp is None:
            global_mpp = mpp
            global_scales = scales
            sample.metadata['scalefactors']['microns_per_pixel'] = mpp
            if 'tissue_hires_scalef' in scales:
                sample.metadata['scalefactors']['tissue_hires_scalef'] = scales['tissue_hires_scalef']
            if 'tissue_lowres_scalef' in scales:
                sample.metadata['scalefactors']['tissue_lowres_scalef'] = scales['tissue_lowres_scalef']

        h5_file = os.path.join(bin_path, "filtered_feature_bc_matrix.h5")
        parquet_file = os.path.join(bin_path, "spatial", "tissue_positions.parquet")

        if os.path.exists(h5_file) and os.path.exists(parquet_file):
            # --- Load full expression matrix with Scanpy ---
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Variable names are not unique.*")
                adata = sc.read_10x_h5(h5_file)
            adata.var_names_make_unique()
            h5_barcodes = adata.obs_names.to_numpy()

            df = pd.read_parquet(parquet_file).set_index('barcode')
            if df.index.dtype != object: df.index = df.index.astype(str)

            coords_raw = df[["pxl_col_in_fullres", "pxl_row_in_fullres"]].reindex(h5_barcodes).values

            # --- Calculate Global Origin Offset ---
            # We capture this once. Because they are cropped from the same fullres
            # morphology image, the offset in microns is universal across bins.
            min_col = np.nanmin(coords_raw[:, 0])
            min_row = np.nanmin(coords_raw[:, 1])
            offset_x_um = min_col * mpp
            offset_y_um = min_row * mpp

            if global_offset_x == 0.0 and global_offset_y == 0.0:
                global_offset_x, global_offset_y = offset_x_um, offset_y_um

            coords_microns = (coords_raw + 0.5) * mpp

            # Save total counts and coords directly into adata for easy access later.
            # Also tag the bin size in uns so the renderer can read the correct spot size
            # when selecting this level from a multi-bin OmicChannel pyramid.
            adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()        # type: ignore
            adata.obsm['spatial'] = coords_microns
            adata.uns['bin_size_um'] = actual_bin_um

            adata = build_spatial_block_index(adata, mpp=mpp, gene_col=None)

            omic_pyramid[float(actual_bin_um)] = adata
            omic_geometry = "square" if "square" in bin_folder else "spot"

            if omic_spatial is None:
                omic_spatial = SpatialMetadata(
                    transform_matrix=create_transform_matrix(scale_x=1.0, scale_y=1.0, tx=0.0, ty=0.0),
                    unit="microns"
                )

            if debug: print(f"  + Loaded {bin_sz}um bin (Origin: {offset_x_um:.1f}, {offset_y_um:.1f} µm, Genes: {adata.n_vars})")

    # Synthesize 4µm bins from 2x2 aggregation if requested but not on disk
    if 4 in bin_sizes and 2.0 in omic_pyramid and 4.0 not in omic_pyramid:
        if debug: print("  → Synthesizing 4µm bins via 2x2 aggregation of 2µm data...")
        adata_4um = aggregate_bins_2to4(omic_pyramid[2.0])
        adata_4um = build_spatial_block_index(adata_4um, mpp=mpp, gene_col=None)
        omic_pyramid[4.0] = adata_4um
        n2, n4 = omic_pyramid[2.0].n_obs, omic_pyramid[4.0].n_obs
        if debug: print(f"     {n4:,} × 4µm bins (from {n2:,} × 2µm bins)")

    # Assemble all loaded bins into a single OmicChannel with an internal pyramid.
    # The finest bin is the base (self.data); renderers call get_optimal_level() to
    # pick the appropriate bin automatically based on the target canvas resolution.
    if omic_pyramid and omic_spatial is not None:
        sorted_bins = sorted(omic_pyramid.keys())   # finest first
        finest_bin_um = sorted_bins[0]
        omic_chan = OmicChannel(
            name=omic_name,
            data=omic_pyramid[finest_bin_um],
            spatial_meta=omic_spatial,
            metadata={
                "platform": "visium",
                "geometry_type": omic_geometry,
                "bin_size_um": finest_bin_um,
            },
            pyramid=omic_pyramid
        )
        sample.add_omic(omic_chan)
        if debug: print(f"  → OmicChannel '{omic_name}' ready: bins {sorted_bins} µm")

    # =========================================================================
    # PHASE 2: Load Morphology Images (Global Registry)
    # =========================================================================
    # Look for the canonical scalefactors in the root spatial directory first
    base_spatial_json = os.path.join(base_dir, "spatial", "scalefactors_json.json")
    if os.path.exists(base_spatial_json):
        with open(base_spatial_json, 'r') as f:
            master_scales = json.load(f)
    else:
        master_scales = global_scales # Fallback if base is missing
            
    for itype in image_types:
        # Check if itype is valid AND make sure we haven't already loaded it
        if itype not in image_map or itype in sample.images:
            continue
            
        info = image_map[itype]
        scale_val = master_scales.get(info['scale_key'], 1.0)
        
        # Ensure we have a valid MPP even if Phase 1 failed to find omics
        mpp = master_scales.get("microns_per_pixel", global_mpp if global_mpp else 1.0)
        px_to_um = mpp / scale_val 
        
        # 1. Prefer the canonical image in the root /outs/spatial/ folder
        # 2. Fallback to the symlink in the first bin's folder if root is missing
        paths_to_try = [os.path.join(base_dir, "spatial", info['file'])]
        if bin_sizes:
            paths_to_try.append(os.path.join(base_dir, "binned_outputs", f"square_{bin_sizes[0]:03d}um", "spatial", info['file']))

        for p in paths_to_try:
            if os.path.exists(p):
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                
                # 1. Load the array
                img_array = np.array(Image.open(p))
                
                # 2. CAPTURE THE DIMENSIONS (Height, Width)
                # This is the "Local Truth" that SpatialMetadata needs 
                # to project pixels into microns.
                h, w = img_array.shape[:2]
                
                # 3. Pass local_bounds to SpatialMetadata
                img_spatial = SpatialMetadata(
                    transform_matrix=create_transform_matrix(
                        scale_x=px_to_um, scale_y=px_to_um, 
                        tx=0, ty=0
                    ),
                    local_bounds=(0.0, float(w), 0.0, float(h)),  # <--- CRITICAL FIX
                    unit="microns"
                )

                # 4. Add the channel
                sample.add_image(ImageChannel(
                    name=itype,
                    image=img_array,
                    spatial_meta=img_spatial,
                    metadata={"source_file": p}
                ))
                
                if debug: print(f"  + Loaded canonical '{itype}' image ({w}x{h} px) aligned to spot origin.")
                break
        
    return sample


def add_visium_fullres_channel(sample: SampleSet, path: str, name: str, mpp: float = None, debug=True):  #type: ignore
    import os
    import numpy as np
    from PIL import Image
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Full-res image not found at: {path}")

    # Auto-fetch MPP from the sample if the user didn't explicitly provide it
    if mpp is None:
        try:
            mpp = sample.metadata['scalefactors']['microns_per_pixel']
        except KeyError:
            raise ValueError("mpp must be provided explicitly if it is not found in sample.metadata['scalefactors'].")
        
    # Lazy-load TIFFs where possible; fall back to PIL for other formats.
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        import tifffile
        import dask.array as da
        from dask import delayed
        try:
            # Fast path: uncompressed TIFFs support direct memory mapping
            memmap = tifffile.memmap(path)
            img_array = da.from_array(memmap, chunks=(4096, 4096))
            h, w = memmap.shape[:2]
        except ValueError:
            # Compressed TIFF — use delayed read so the file isn't loaded into RAM
            # until the viewport actually needs a crop of it
            with tifffile.TiffFile(path) as tif:
                shape = tif.series[0].shape
                dtype = tif.series[0].dtype
            h, w = shape[:2]
            img_array = da.from_delayed(delayed(tifffile.imread)(path), shape=shape, dtype=dtype)
    else:
        Image.MAX_IMAGE_PIXELS = None
        img_array = np.array(Image.open(path))
        h, w = img_array.shape[:2]
    
    # Pre-compute percentile table for large images (> 20MP) so adjust_display
    # can skip re-reading the full array every time it's called.
    _STATS_THRESHOLD = 20_000_000
    _PCT_KEYS = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.8, 99.9, 99.99, 100]
    display_stats = None
    if h * w > _STATS_THRESHOLD:
        import dask.array as _da
        step = max(1, int(((h * w) / 500_000) ** 0.5))  # target ~500K samples
        subsample = img_array[::step, ::step]
        if isinstance(subsample, _da.Array):
            subsample = subsample.compute()
        flat = subsample[subsample > 0].flatten() if np.any(subsample > 0) else subsample.flatten()
        display_stats = {
            'percentile_keys':   _PCT_KEYS,
            'percentile_values': np.percentile(flat, _PCT_KEYS).tolist(),
            'n_samples': int(len(flat)),
            'dtype': str(img_array.dtype),
            'source_mpp': float(mpp * step),
        }
        if debug: print(f"  → Computed display stats ({len(flat):,} samples, step={step})")

    # Visium full-res maps 1:1 with coords, starting at 0,0
    img_spatial = SpatialMetadata(
        transform_matrix=create_transform_matrix(scale_x=mpp, scale_y=mpp, tx=0.0, ty=0.0),
        local_bounds=(0.0, float(w), 0.0, float(h)),
        unit="microns"
    )

    chan = ImageChannel(
        name=name,
        image=img_array,
        spatial_meta=img_spatial,
        metadata={"source_file": path, "display_stats": display_stats}
    )
    chan.generate_pyramid()
    sample.add_image(chan)

    if debug:
        n_levels = len(chan.pyramid)
        mpps = sorted(chan.pyramid.keys())
        print(f"Added '{name}' ({w}x{h} px, {mpp:.4f} mpp) — {n_levels} pyramid level(s): {[f'{m:.2f}' for m in mpps]} mpp")