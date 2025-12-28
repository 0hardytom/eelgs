import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.regions import PolygonSkyRegion, CircleSkyRegion
from astropy.table import Table
from astropy.wcs import WCS

def parse_s_region(s_region_string: str) -> list[PolygonSkyRegion]:
    """
    Parses an s_region string from an astronomical catalog query into a list
    of astropy PolygonSkyRegion objects.

    Handles strings containing one or multiple POLYGON definitions.
    """
    # The string can contain multiple polygons, so we split by the keyword "POLYGON"
    # This will result in a list where the first element is empty.
    polygon_strs = s_region_string.strip().upper().split('POLYGON')[1:]
    
    regions = []
    for poly_str in polygon_strs:
        # The first word might be a coordinate system (e.g., ICRS).
        # We need to determine if the first element is a string or a coordinate.
        parts = poly_str.strip().split()
        if not parts:
            continue

        # Check if the first part is the coordinate system or the first coordinate
        try:
            # If this succeeds, the first part is a coordinate, and there is no system string
            float(parts[0])
            coord_parts = parts
        except (ValueError, IndexError):
            # Otherwise, the first part is the system string (e.g., 'ICRS'), so we skip it
            coord_parts = parts[1:]

        # Extract coordinate values (as floats)
        coords_flat = [float(p) for p in coord_parts]

        # A valid polygon must have an even number of coordinates. If not, skip it.
        if len(coords_flat) % 2 != 0:
            # Consider logging a warning here in a real application
            continue

        # Group the flat list into pairs of (ra, dec)
        vertices_coords = np.reshape(coords_flat, (-1, 2))
        
        # Create an astropy SkyCoord object for the vertices
        vertices = SkyCoord(vertices_coords, unit='deg', frame='icrs')
        
        # Create the PolygonSkyRegion and add it to our list
        regions.append(PolygonSkyRegion(vertices=vertices))
        
    return regions

def is_circle_in_footprint(
    s_region_string: str,
    circle_center: SkyCoord,
    circle_radius: u.Quantity
) -> bool:
    """
    Checks if a circular aperture is fully contained within an observation footprint.

    The footprint can consist of one or more polygons (e.g., for multi-chip detectors).
    The circle is considered "in" if it is fully contained by ANY of the polygons.

    Args:
        s_region_string: The string value from the 's_region' column.
        circle_center: The center of the circular aperture as an astropy SkyCoord.
        circle_radius: The radius of the aperture as an astropy Quantity (e.g., 1 * u.arcsec).

    Returns:
        True if the circle is contained in any of the footprint's polygons, False otherwise.
    """
    if not s_region_string or not isinstance(s_region_string, str):
        return False

    # Create the circular region for your object of interest
    circle_to_check = CircleSkyRegion(center=circle_center, radius=circle_radius)
    
    # Parse the footprint string into one or more PolygonSkyRegion objects
    footprint_polygons = parse_s_region(s_region_string)

    if not footprint_polygons:
        return False

    # To check if a SkyRegion contains another, we need a WCS object to project
    # the regions onto a common 2D plane. We can create a simple tangential
    # projection centered on the region of interest.
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0, 0]
    # Use a pixel scale appropriate for the circle size, e.g., 1/10th of the radius
    pixel_scale = (circle_radius / 10).to(u.deg).value
    wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale])
    # Center the projection on the circle's center for accuracy
    wcs.wcs.crval = [circle_center.ra.deg, circle_center.dec.deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    # Check if the circle is contained in *any* of the polygons
    for polygon in footprint_polygons:
        if polygon.contains(circle_to_check, wcs=wcs):
            return True
            
    return False

def check_footprints_in_table(
    table: Table,
    circle_center: SkyCoord,
    circle_radius: u.Quantity,
    s_region_col: str = 's_region'
) -> np.ndarray:
    """
    Efficiently checks which rows in an Astropy Table contain a circular aperture.

    This function iterates over the table's s_region column and returns a
    boolean numpy array that can be used to mask the table.

    Args:
        table: The Astropy Table containing the observation footprints.
        circle_center: The center of the circular aperture (SkyCoord).
        circle_radius: The radius of the aperture (astropy Quantity).
        s_region_col: The name of the column containing the s_region strings.

    Returns:
        A numpy array of booleans with the same length as the table.
        True where the circle is contained, False otherwise.
    """
    is_contained_list = [
        is_circle_in_footprint(row[s_region_col], circle_center, circle_radius)
        for row in table
    ]
    return np.array(is_contained_list)
