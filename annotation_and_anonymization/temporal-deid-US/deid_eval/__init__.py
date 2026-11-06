"""Phase 2 - evaluation of de-identification tools on temporal (2D+t / 3D+t) ultrasound DICOM.

Research question: does de-identification remove identifiable information without altering the
scientific information required to correctly reuse a temporal ultrasound DICOM?

Modules
-------
config       paths and run limits (environment-variable driven, no machine-specific defaults)
dicom_utils  frame decoding, ultrasound regions, per-frame functional groups
datasets     public data acquisition with a pinned, checksummed manifest
synthetic    small synthetic 2D+t / 3D+t DICOMs (quick mode and tests)
canaries     known-positive metadata PHI injection and byte-level leak scanning
burned_in    burned-in text challenge, OCR and text-residual metrics
phase1       reader for the subgroup's Phase 1 synthetic-PHI outputs (phi_overlays.csv)
adapters     de-identification tools and negative controls behind one batch interface
validators   privacy, scientific-preservation and reuse validators
plots        shared plotting style and generic charts
report       data-driven answer to the research question
"""

__version__ = "0.1.0"
