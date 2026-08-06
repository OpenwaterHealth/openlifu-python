# Technical Reports

Verification studies, characterization measurements, and accuracy analyses for
the Open-LIFU platform.

Each report in this folder documents a measurement made against the code in this
repository. Reports are versioned alongside the software they validate, so a
reader can trace a stated accuracy figure to the specific implementation it was
measured on.

Reports are archived on Zenodo with a DOI and are citable in publications, IRB
submissions, and grant applications.

---

## Index

| Report                                                                                       | Date    | Authors                              | Validates                                                                                   | DOI                                                                |
| -------------------------------------------------------------------------------------------- | ------- | ------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [Open-LIFU Transducer Localization Accuracy Study](./2026-transducer-localization-accuracy/) | 2026-08 | Tahir, Hollender, Paribello, Konecky | Photogrammetry → MRI → transducer co-registration path in the Open-LIFU desktop application | [`10.5281/zenodo.XXXXXXX`](https://doi.org/10.5281/zenodo.XXXXXXX) |

---

## What belongs here

- Verification and characterization studies with reported measurements
- Accuracy, precision, and repeatability analyses
- Benchmark comparisons between methods, algorithms, or configurations
- Device characterization useful to external research teams (duty cycle, thermal
  behavior, calibration)

## What does not belong here

- API and usage documentation — see
  [docs.openwater.health](https://docs.openwater.health)
- Controlled quality records — these live in the QMS and are not published from
  this repository
- Clinical study results — these are published through peer review, not this
  folder
- Marketing or announcement copy

A report published here may be **derived from** a controlled record, but the
controlled record itself is never the file in this folder. Confirm status with
the Quality lead before adding a report that traces to a design-control
document.

---

## Structure for a new report

```
technical-reports/
└── YYYY-short-descriptive-slug/
    ├── index.md          # Narrative version; source for the docs site
    ├── report.pdf        # Formatted PDF of record; carries the DOI
    └── figures/          # Figure source files
```

Both `index.md` and `report.pdf` must report identical numbers. The markdown is
the authoring source; the PDF is generated from it.

## Adding a report

1. Draft `index.md` and have it reviewed by the technical owner of the code
   being validated.
2. Reserve a DOI on Zenodo (Resource type: **Publication / Report**) and place
   it in the PDF footer and at the top of `index.md`.
3. Open a pull request adding the report folder and a row in the index table
   above.
4. After merge, publish the Zenodo record with the final PDF and add this
   repository folder as a related identifier.
5. Add the page to `mkdocs.yml` navigation, so it renders on the documentation
   site.

Full sequence, including how reports are surfaced on openwater.health and the
community hub, is documented in the contribution guide.

## Corrections

Open an issue against this repository if a reported figure appears inconsistent
or a result does not reproduce. Substantive corrections are published as a new
Zenodo version rather than a silent edit, so that anything already cited remains
resolvable.

---

## Licensing

Documents in this folder — including markdown, PDFs, and figures — are licensed
under [**CC BY 4.0**](./LICENSE).

This is **separate from the software license governing the rest of this
repository.** Source code in `openlifu-python` is licensed under AGPL-3.0. The
document license applies only to the contents of `technical-reports/` and does
not extend to any code referenced or reproduced within a report.

You may share and adapt these documents for any purpose, including commercially,
with appropriate credit.

---

## Regulatory notice

_The platform discussed in these reports is exclusively intended for research
purposes and is not cleared or approved by the FDA for clinical use. It is
solely available to researchers and research institutions who are able to
customize it to address the specific disease state they wish to study. The
safety and effectiveness of the platform have not been established through the
FDA's formal review process._

Measurements reported here are engineering characterizations. They are not
claims of clinical performance, diagnostic accuracy, or fitness for any clinical
application.

---

**Maintained by:** Openwater ·
[community@openwater.health](mailto:community@openwater.health)
