from osgeo import gdal
import numpy as np

# Suppress GDAL warnings
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


def read_band(dataset, band_number):
    band = dataset.GetRasterBand(band_number)
    return band.ReadAsArray().astype(np.float32)


def calculate_index(band1, band2):
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    return np.where(
        (band1 + band2) != 0,
        (band1 - band2) / (band1 + band2 + epsilon),
        -9999  # Use -9999 as NoData value
    )


def save_tiff(output_path, data, input_dataset):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32,
                            options=['COMPRESS=DEFLATE', 'INTERLEAVE=BAND'])
    dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    dataset.SetProjection(input_dataset.GetProjection())
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()
    return dataset


def process_image(input_file):
    try:
        dataset = gdal.Open(input_file)
        if dataset is None:
            raise IOError(f"Unable to open {input_file}")

        print(f"Satelitska snimka sadrži {dataset.RasterCount} kanala.")

        band_4 = read_band(dataset, 4)
        band_8 = read_band(dataset, 8)
        band_11 = read_band(dataset, 11)

        ndvi = calculate_index(band_8, band_4)
        ndmi = calculate_index(band_8, band_11)

        save_tiff('ndvi_output.tiff', ndvi, dataset)
        save_tiff('ndmi_output.tiff', ndmi, dataset)

        print(f"Prosječna vrijednost NDVI: {np.mean(ndvi[ndvi != -9999]):.4f}")
        print(f"Prosječna vrijednost NDMI: {np.mean(ndmi[ndmi != -9999]):.4f}")
        print("TIFF files created successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    input_file = "response_bands.tiff"
    process_image(input_file)
