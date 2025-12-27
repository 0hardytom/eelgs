# Meeting Notes: Data Exploration & Initial Analysis

**Date:** 9th October
**Supervisor:** Prof. Alastair Edge

---

## Data & Example Objects

The primary dataset for this initial analysis is a MUSE datacube for the galaxy cluster **MACS J0159.8-3413**. Alastair will provide the background-subtracted data cube via Microsoft Teams.

We have identified several interesting objects within this cube to act as test cases:

1.  **Foreground Galaxy (z ≈ 0.25):**
    -   **Coordinates:** `01h 59m 04.03s`, `-34d 13m 31.8s`
    -   **Properties:** This is a star-forming galaxy located in the foreground of the main cluster. It exhibits a strong Hβ emission line relative to its [OIII] emission, which is a typical characteristic of such galaxies.

2.  **Background Galaxy (z ≈ 0.6):**
    -   **Coordinates:** `01h 59m 06.05s`, `-34d 13m 03.8s`
    -   **Properties:** This is a background galaxy whose light is gravitationally lensed by the foreground cluster, causing it to appear as multiple, mirrored, and fragmented images. It is an [OII] emitter. The fragmented appearance is due to the fact that it is a line-dominated object with distinct emission components, each being lensed differently. This object serves as an excellent test case for verifying that the intrinsic line ratios are consistent across its multiple images.

3.  **Faint [OIII] Emitter:**
    -   **Properties:** A very faint object with a companion at approximately one-third of its flux. Hβ is fainter, and the [OII] line is very weak. This object is a candidate for a "Green Pea" galaxy—a type of compact, young, star-forming galaxy with extremely strong [OIII] emission.

---

## Analysis Plan & Action Items

1.  **Source Extraction & Photometry:**
    -   **Manual Extraction:** For now, perform manual source identification and analysis. A full source extraction pipeline is in development by another student (David).
    -   **Aperture Photometry:** For each identified object, extract a spectrum by summing the flux within a small aperture (e.g., a 3-pixel radius) around the source. This can be done using standard astronomical libraries.
    -   **Cubelet Extraction:** Cut out smaller "cubelets" centered on the objects of interest to facilitate focused analysis.

2.  **Spectral Analysis:**
    -   **Line Identification:** Identify and measure the flux of all major emission lines from Hα (if visible) down to [OII] in the rest frame.
    -   **Line Ratio Diagnostics:**
        -   For objects where multiple lines are detected, calculate diagnostic line ratios to infer physical properties. For example, the ratio of [OIII] to Hβ is sensitive to the ionization state and metallicity of the gas.
        -   Where lines are *not* detected (e.g., weak [OII] in an [OIII]-strong galaxy), calculate the upper limit on the flux based on the noise floor. This provides a lower limit on line ratios, which can still constrain the physical conditions (e.g., ionization level).
    -   **Gaussian Fitting:** Use Gaussian profiles to measure the flux under emission features, especially when trying to derive limits from non-detections.

3.  **Multi-wavelength Counterpart Identification:**
    -   Use the coordinates of the MUSE sources to find their stellar counterparts in imaging data from other telescopes. This is crucial for understanding the host galaxy properties (e.g., morphology, stellar mass).
    -   **Available Archives:**
        -   **Hubble Space Telescope (HST):** Check the HST Legacy Archive for high-resolution optical imaging.
        -   **James Webb Space Telescope (JWST):** Some objects may have 1.5µm and 3µm imaging.
        -   **Spitzer Space Telescope:** Check for 3µm and 4µm imaging for bright sources to get limits on IR fluxes.
        -   **Very Large Telescope (VLT):** K-band (near-infrared) imaging is also available.

4.  **Comparison with Wider Surveys:**
    -   Compare the properties of the identified galaxies with larger samples from surveys like **MUSE-WIDE**. This will help place our findings in the context of the broader galaxy population at various redshifts.

---

## Key Concepts & Resources

-   **[OIII] Doublet:** The [OIII] emission feature is a doublet with two lines at 4959Å and 5007Å. Due to the underlying atomic physics, the ratio of their fluxes is fixed at approximately **1:3**. This is a key signature to look for when identifying [OIII] emitters.
-   **Recommended Reading:**
    -   **"Astrophysics of Gaseous Nebulae and Active Galactic Nuclei" by Donald E. Osterbrock.** This book is an essential reference for interpreting astronomical spectra and understanding the physics behind diagnostic line ratios.
        -   **[OIII] ratio:** Temperature-dependent.
        -   **[OII] and [SII] ratios:** Density-dependent. These forbidden lines are sensitive to changes in density, which affects the balance between collisional and spontaneous emission.
-   **Green Pea Galaxies:** These are a class of compact, luminous, young galaxies characterized by strong emission lines, particularly [OIII], indicating a high degree of ionization and intense star formation. The faint [OIII] emitter we found is a potential candidate.
