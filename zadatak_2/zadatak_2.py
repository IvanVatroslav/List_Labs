import sys
import requests
import geopandas as gpd


def fetch_data(url):
    """Fetch data from the given API URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        sys.exit(1)


def main():
    # Fetch the data from the API
    url = 'https://plovput.li-st.net/getObjekti/'
    data = fetch_data(url)

    gdf = gpd.GeoDataFrame.from_features(data['features'])

    gdf.set_crs(epsg=4326, inplace=True)

    print(f"Total number of objects: {len(gdf)}")

    filtered_gdf = gdf[gdf['tip_objekta'] == 16]

    filtered_count = len(filtered_gdf)
    print(f"Number of objects with 'tip_objekta' == 16: {filtered_count}")

    output_filename = 'filtered_objects.geojson'
    filtered_gdf.to_file(output_filename, driver='GeoJSON')

    print(f"Filtered data saved to {output_filename}")


if __name__ == "__main__":
    main()
