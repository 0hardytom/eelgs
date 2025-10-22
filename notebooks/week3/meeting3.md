# Meeting Notes: Spectral Analysis Strategy & Next Steps

**Date:** 19th October
**Supervisor:** Prof. Alastair Edge

---

## Primary Analysis Goals

The main focus is to systematize the analysis of emission line galaxies in the MUSE datacubes. The core strategy revolves around using the strong [OIII] emission line as a primary anchor for identifying and characterizing these objects.

1.  **Anchor on [OIII] Emission:**
    -   The immediate priority is to robustly identify and measure the flux of the [OIII] 5007Å line.
    -   Once [OIII] is secured, use its redshift and spatial position to search for the corresponding Hβ line.
    -   **Verification:** The theoretical flux ratio of the [OIII] doublet (5007Å to 4959Å) is ~3:1. This fixed ratio should be used to confirm the identification of [OIII] emitters, especially in low signal-to-noise (SNR) spectra.

2.  **Derive Key Physical Ratios:**
    -   The primary diagnostic will be the **[OIII]/Hβ ratio**. This is a critical indicator of the ionization state of the gas.
    -   For objects with detectable Hα, the Hα/Hβ ratio can also be used to estimate dust extinction.
    -   The goal is to derive these ratios (or their lower limits) for a couple of key, representative objects first.

3.  **Statistical Analysis of Galaxy Populations:**
    -   For the full sample of ~200, the plan is to analyze the different velocity components within the cluster.
    -   By measuring the centroid of the emission lines for each source, we can trace the kinematic properties of the galaxy populations.
    -   Ultimately, we can create an average spectrum for the entire population to statistically enhance faint features and study the aggregate properties of galaxies with detected [OIII] and Hβ.

---

## Action Items & Methodology

1.  **Cross-correlation with External Catalogues:**
    -   Utilize existing catalogues (e.g., from the WFI) to extract 1D spectra for known objects.
    -   **Key Question:** How do the properties of our lensed galaxies compare to the general population of galaxies in the foreground and background of the cluster?

2.  **Incorporate Archival Imaging:**
    -   **HST WFPC2:** Use archival Hubble imaging to analyze the continuum level underneath the emission lines. This is crucial for calculating accurate equivalent widths.
        -   Even if an object is purely an emission line source with no intrinsic continuum, the broad-band HST filter will still detect flux from the line itself, which can be used to estimate the line's contribution to the overall brightness.
    -   **HST/JWST Photometry:** The strong emission lines in these galaxies can significantly affect their broad-band colours, potentially causing "spikes" or excesses in photometric redshift calculations. This effect should be quantified.

3.  **Address Selection Bias:**
    -   We need to consider how our selection methods (e.g., prioritizing strong [OIII] emitters) might bias our sample. For this stage of the analysis, we can likely ignore photo-spectroscopic biases and focus on the spectroscopic sample.

4.  **Ensure Repeatability:**
    -   Verify the analysis by checking for consistency in the combined redshift measurements from multiple lines.
    -   Investigate how faint of an [OIII] line we can reliably detect and still trust the 3:1 doublet ratio for confirmation.

---

## Specific Objects & Scientific Questions

-   **"Aurora" Object:** There is a particular object of interest that shows a strong rise in its continuum towards the [OIII] line. This feature warrants a more detailed investigation to understand its nature.
-   **Comparison to Literature:** The results should be compared to established values from the literature, such as the typical properties of Hα emitters at a redshift of z ≈ 0.84.

---

## Tools & Resources

-   **LensTool:** For gravitational lensing analysis and modeling, Dougal is the recommended contact.
