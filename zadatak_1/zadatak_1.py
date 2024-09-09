import rasterio
import numpy as np
import logging
from rasterio.errors import RasterioIOError
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inspect_band(src, band_num, band_name):
    band_data = src.read(band_num)
    logging.info(f"{band_name} band - Shape: {band_data.shape}, Type: {band_data.dtype}")
    logging.info(f"Range: {np.min(band_data)} to {np.max(band_data)}")
    logging.info(f"Sample data (5x5 corner): \n{band_data[:5, :5]}")

    # Save individual band
    with rasterio.open(f"{band_name.lower()}_band.tif", 'w', driver='GTiff',
                       width=src.width, height=src.height,
                       count=1, dtype=band_data.dtype, crs=src.crs,
                       transform=src.transform) as dst:
        dst.write(band_data, 1)
    logging.info(f"Saved {band_name} band as {band_name.lower()}_band.tif")


def calculate_index(band1, band2, name):
    np.seterr(divide='ignore', invalid='ignore')
    index = (band1.astype(np.float32) - band2.astype(np.float32)) / (
                band1.astype(np.float32) + band2.astype(np.float32))
    index_valid = np.where(np.isfinite(index), index, np.nan)

    logging.info(f"{name} calculation - Shape: {index_valid.shape}, Type: {index_valid.dtype}")
    logging.info(f"Range: {np.nanmin(index_valid)} to {np.nanmax(index_valid)}")
    logging.info(f"Sample data (5x5 corner): \n{index_valid[:5, :5]}")

    return index_valid


def save_geotiff(data, output_path, src):
    profile = src.profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    logging.info(f"Saved result as {output_path}")


def run_gdalinfo(file_path):
    try:
        result = subprocess.run(['gdalinfo', file_path], capture_output=True, text=True)
        logging.info(f"GDALINFO for {file_path}:\n{result.stdout}")
    except Exception as e:
        logging.error(f"Error running gdalinfo: {e}")


def main():
    image_path = "response_bands.tiff"

    try:
        with rasterio.open(image_path) as src:
            logging.info(f"Opened image: {image_path}")
            logging.info(f"Image shape: {src.shape}, CRS: {src.crs}, Bands: {src.count}")

            run_gdalinfo(image_path)

            red = src.read(4)
            nir = src.read(8)
            swir = src.read(11)

            inspect_band(src, 4, "Red")
            inspect_band(src, 8, "NIR")
            inspect_band(src, 11, "SWIR")

            ndvi = calculate_index(nir, red, "NDVI")
            ndmi = calculate_index(nir, swir, "NDMI")

            save_geotiff(ndvi, "ndvi_debug.tif", src)
            save_geotiff(ndmi, "ndmi_debug.tif", src)

            run_gdalinfo("ndvi_debug.tif")
            run_gdalinfo("ndmi_debug.tif")

    except RasterioIOError as e:
        logging.error(f"Error opening the image: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
