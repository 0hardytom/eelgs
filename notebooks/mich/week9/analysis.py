import numpy as np
from astropy.table import Table

def calculate_log_n_log_s(table: Table) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the log-N-log-S data from an astropy table.

    This function takes an astropy table, filters for positive fluxes,
    sorts by flux, and then calculates the cumulative number counts (N)
    as a function of flux (S). It returns the base-10 logarithm of
    both quantities.

    Args:
        table (Table): An astropy Table containing the source data.
                       Must include an 'oiii5007_flux' column.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       - log_S: The logarithm of the flux values.
                                       - log_N: The logarithm of the cumulative counts.
    """
    # Ensure the table has the required column
    if 'oiii5007_flux' not in table.colnames:
        raise ValueError("Table must contain an 'oiii5007_flux' column.")

    # Filter out sources with non-positive flux
    positive_flux_table = table[table['oiii5007_flux'] > 0]

    if len(positive_flux_table) == 0:
        print("No sources with positive flux found. Returning empty arrays.")
        return np.array([]), np.array([])

    # Sort the table by flux in descending order
    sorted_table = positive_flux_table.copy()
    sorted_table.sort('oiii5007_flux', reverse=True)

    # Get the flux (S) and calculate the cumulative number (N)
    S = sorted_table['oiii5007_flux']
    N = np.arange(1, len(sorted_table) + 1)

    # Calculate log10(S) and log10(N)
    log_S = np.log10(S)
    log_N = np.log10(N)

    return log_S, log_N
