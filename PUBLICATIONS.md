# Publications and preprints using `Herculens`

## Original publications

- "**_Using wavelets to capture deviations from smoothness in galaxy-scale strong lenses_**" ([**Galan et al. 2022**](https://scixplorer.org/abs/2022A%26A...668A.155G/abstract)): presentation of one of the first auto-differantiable lensing modelling codes, implementing efficient wavelet-based regularisation for both surface brightness and lensing potential pixelated corrections. <ins>Please cite this work if you make use of `Herculens` in any way</ins> ([bibtex entries](CITATION.md)).

- In addition, you can also refer to this [Astrophysics Source Code Library entry](https://scixplorer.org/abs/2022ascl.soft09002G/abstract).

## Galaxy properties & evolution

- "**_El Gordo needs El Anzuelo: Probing the structure of cluster members with multi-band extended arcs in JWST data_**" ([**Galan et al. 2024a**](https://scixplorer.org/abs/2024A%26A...689A.304G/abstract)): high-resolution and multi-wavelength reconstruction of a star-forming galaxy by three members of the supermassive cluster _El Gordo_, using 3D correlated fields. <ins>Please cite this work if you use the `CorrelatedField` model of `Herculens`; please also cite the [`NIFTy.re` paper](https://scixplorer.org/abs/2024JOSS....9.6593E/abstract), which implements the underlying forward model</ins> ([bibtex entries](CITATION.md)).

- "**_Unveiling a 36 billion solar mass black hole at the centre of the Cosmic Horseshoe gravitational lens_**" ([**Melo-Carneiro et al. 2025**](https://scixplorer.org/abs/2025MNRAS.541.2853M/abstract)): test for systematic errors using a different lens modeling code in the study of the super-massive black hole at the center of the "Cosmic Horseshoe" gravitational lens.

- "**_A Salpeter IMF and an NFW halo: disentangling the dark and stellar mass of an elliptical galaxy through precise lens modelling of a double-source-plane system_**" ([**Li et al. 2026a**](https://scixplorer.org/abs/2026MNRAS.548ag683L/abstract)): strong lensing analysis of the "Jackpot" to measure its inner dark matter density profile, stellar-to-halo mass ratio, and stellar initial mass function normalisation.

- "**_Disentangling the dark and stellar mass through precise lens modelling of the JWST observation of lensed quasar WFI2033-4723_**" ([**Li et al. 2026b**](https://scixplorer.org/abs/2026MNRAS.553g1986L/abstract)): multiplane lens modelling of JWST imaging of the lensed quasar WFI2033-4723 with a stars-plus-generalised-NFW mass model, separating the dark and stellar mass of the main deflector.

- "**_Microlensing of microlensing: effects of random stars on the double-source-plane gravitational lens_**" ([**Salama et al. 2026**](https://scixplorer.org/abs/2026MNRAS.551g1382S/abstract)): multi-plane lens model of the "Jackpot", used to study how microlensing by stars in the foreground deflector affects the double-source-plane lens.

## Dark matter properties

- "**_The overconcentrated dark halo in the strong lens SDSS J0946+1006 is a subhalo: evidence for self interacting dark matter?_**" ([**Enzi et al. 2025**](https://scixplorer.org/abs/2025MNRAS.540..247E/abstract)): multi-plane field modeling of the unique "Jackpot" lensing system to measure robustly the mass density profile of a dark matter substructure, further confirming its CDM-challenging nature.

## Cosmology

- "**_Can we constrain cosmology with compound gravitational lenses?_**" ([**Bowden et al. 2026**](https://scixplorer.org/abs/2026arXiv261115714B/abstract)): simulation and multi-plane forward modelling of mock _Euclid_ and HST double-source-plane lenses, to assess how well compound lenses can constrain cosmological parameters.

- "**_Harnessing stellar kinematics to constrain dark energy with the double-source-plane gravitational lens SDSS J0946+1006_**" ([**Ballard et al. 2026**](https://scixplorer.org/abs/2026arXiv260908573B/abstract)): joint modelling of HST imaging and MUSE stellar kinematics of the "Jackpot" on GPU, with an analytic lens mass and a pixelated source that avoids any lensing matrix inversion, to constrain dark energy.

## Confirmation of lens candidates

- "**_Euclid Quick Data Release (Q1): XXIX. The strong-lensing discovery engine D: Double-source-plane lens candidates_**" ([**Li et al. 2025**](https://scixplorer.org/abs/2026A%26A...711A..29E/abstract)): detailed lens models of five double-source-plane lens candidates found in _Euclid_ data, including four confirmed to have two sources at different redshifts by the models.

- "**_Euclid Quick Data Release (Q1). The Strong Lensing Discovery Engine F: Bright and low-redshift strong lenses_**" ([**Ecker et al. 2026**](https://scixplorer.org/abs/2026arXiv260328580E/abstract)): multi-plane lens modelling used to check that one of the bright, low-redshift lens candidates found in _Euclid_ data is a plausible lens.

- "**_Euclid Data Release 1 (DR1). Double-source-plane lens candidates_**" ([**Li et al. 2026c**](https://scixplorer.org/abs/2026arXiv261120166E/abstract)): double-source-plane lens models of 84 galaxy-scale candidates found in _Euclid_ DR1 data (80 of them new), to assess whether they are plausible double-source-plane lenses.

## Mitigation of systematic biases

- "**_Exploiting the diversity of modeling methods to probe systematic biases in strong lensing analyses_**" ([**Galan et al. 2024b**](https://scixplorer.org/abs/2024A%26A...692A..87G/abstract)): Consistent comparison of 6 different modelling methods, including two methods implemented in `Herculens`, to characterize and reduce sources of systematic errors in strong lens modelling analyses.

- "**_ALPACA I: Controlling source and PSF systematics in JWST time-delay cosmography with differentiable lens modeling_**" ([**Krizic et al. 2026**](https://scixplorer.org/abs/2026arXiv260904312K/abstract)): presentation of `ALPACA`, a semi-automatic, GPU-accelerated modelling pipeline built on `Herculens` and `STARRED`, which jointly reconstructs a pixelated correlated-field source, the lens mass and the PSF for JWST time-delay cosmography.

## Methodology

- "**_Modeling lens potentials with continuous neural fields in galaxy-scale strong lenses_**" ([**Biggio et al. 2023**](https://scixplorer.org/abs/2023A%26A...675A.125B/abstract)): proof-of-concept of the use of continuous neural fields coupled to usual lens models to reconstruct the lensing potential maps at high resolution.

- "**_Gaussian processes on ray-guided transformed uniform grids for fast, flexible, and auto-differentiable adaptive source reconstruction in lens modelling_**" ([**Enzi et al. 2026**](https://scixplorer.org/abs/2026MNRAS.tmp.1859E/abstract)): a new adaptive pixelated source reconstruction built on Gaussian processes over ray-guided grids, implemented in `Herculens` ([code](https://github.com/CKrawczyk/RTU-grid-herculens)).

- "**_Improving Lens Modelling from Ground-Based Imaging with Deconvolution_**" ([**Paic et al. 2026**](https://scixplorer.org/abs/2026Galax..14...88P/abstract)): `STARRED` deconvolution combined with `Herculens` multi-band modelling of HSC lenses from SuGOHI, giving more precise and accurate lens parameters from ground-based imaging.

## Lensing standard integration

- "**_COOLEST: COde-independent Organized LEns STandard_**" ([**Galan et al. 2023**](https://scixplorer.org/abs/2023JOSS....8.5567G/abstract)): `Herculens` supports the COOLEST lensing standard to export, read and compare lens models.
