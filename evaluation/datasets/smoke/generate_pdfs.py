"""Generate the PDFs of the synthetic smoke-test dataset.

The three "articles" are fictional and exist only to exercise the evaluation
pipeline end to end (ingestion, retrieval, gate calibration, generation,
citations, English and Finnish questions). Their numbers are invented.

    cd backend
    python ../evaluation/datasets/smoke/generate_pdfs.py

Writes pdfs/*.pdf next to this script. Output is deterministic for a given
PyMuPDF version.
"""

from pathlib import Path

import pymupdf

ARTICLES: dict[str, dict[str, object]] = {
    "hsi": {
        "title": "Hyperspectral Classification of Mineral Samples: LDA versus KNN",
        "author": "Synthetic Author A",
        "pages": [
            "Hyperspectral Classification of Mineral Samples: LDA versus KNN\n\n"
            "Abstract. This synthetic study compares linear discriminant analysis (LDA) and the "
            "k-nearest neighbours (KNN) classifier for identifying minerals from reflectance "
            "spectra. LDA reached an overall accuracy of 91.3 percent, while KNN reached 87.6 "
            "percent on the same held-out test set.\n\n"
            "1. Introduction. Hyperspectral imaging records reflectance in hundreds of narrow, "
            "contiguous wavelength bands. Each pixel therefore carries a spectrum that can be "
            "used to recognise materials. Mineral identification is a classic application "
            "because many minerals have characteristic absorption features in the short-wave "
            "infrared region. Classical classifiers remain attractive in this setting because "
            "they need little training data and their behaviour is easy to inspect. In this "
            "work we compare a parametric method, linear discriminant analysis, with a "
            "non-parametric method, the k-nearest neighbours classifier.",
            "2. Data. The data set contains 1,240 spectra measured from six mineral classes: "
            "calcite, dolomite, gypsum, kaolinite, muscovite and quartz. Spectra were recorded "
            "between 400 and 2500 nm with a spectral resolution of 10 nm, giving 211 bands per "
            "spectrum. The samples were split into a training set of 868 spectra and a test set "
            "of 372 spectra, stratified by class.\n\n"
            "3. Methods. LDA assumes that each class follows a Gaussian distribution with a "
            "shared covariance matrix, which gives linear decision boundaries between classes. "
            "KNN makes no distributional assumptions and assigns each spectrum to the majority "
            "class among its nearest training spectra. Before KNN, the spectra were reduced to "
            "20 principal components to limit the effect of noisy bands. The number of "
            "neighbours for KNN was chosen with 10-fold cross-validation on the training set, "
            "which selected k = 5. Euclidean distance was used for KNN.",
            "4. Results. LDA achieved an overall accuracy of 91.3 percent and KNN 87.6 percent. "
            "Most KNN errors occurred between dolomite and calcite, whose spectra differ mainly "
            "in the position of the carbonate absorption near 2340 nm. Training LDA took 0.4 "
            "seconds, and classifying the whole test set with KNN took 2.1 seconds.\n\n"
            "5. Discussion. KNN was more sensitive to noisy bands than LDA: when the principal "
            "component step was skipped, KNN accuracy dropped to 79.8 percent. LDA benefited "
            "from the shared covariance assumption because the classes had similar spread. "
            "A limitation of the study is that all samples were measured in the laboratory, so "
            "atmospheric effects present in airborne data were not considered.",
        ],
    },
    "soil": {
        "title": "Soil Moisture Retrieval from Sentinel-1 Backscatter in Boreal Forest",
        "author": "Synthetic Author B",
        "pages": [
            "Soil Moisture Retrieval from Sentinel-1 Backscatter in Boreal Forest\n\n"
            "Abstract. This synthetic study estimates surface soil moisture under boreal forest "
            "canopy from Sentinel-1 C-band radar backscatter. A random forest model trained on "
            "in situ measurements reached a root-mean-square error of 0.048 cubic metres per "
            "cubic metre.\n\n"
            "1. Introduction. Soil moisture controls evaporation, runoff and forest growth, but "
            "it is difficult to measure over large areas. Synthetic aperture radar is sensitive "
            "to the dielectric constant of the soil and works regardless of cloud cover. Under "
            "a forest canopy, however, the soil signal is attenuated by the trees, which makes "
            "the retrieval problem harder than over open fields.",
            "2. Study area and data. The study used 42 field sites near Sodankyla in northern "
            "Finland, monitored during the snow-free seasons of 2019 to 2021. Soil moisture was "
            "measured with in situ probes at a depth of 5 cm every hour. Sentinel-1 images in "
            "VV polarisation were acquired every six days on average. Images from November to "
            "April were excluded because of snow cover.\n\n"
            "3. Methods. A random forest regression model with 500 trees was trained to "
            "predict soil moisture from VV backscatter, local incidence angle, and canopy "
            "height from airborne laser scanning. Sites were split by location so that no site "
            "appeared in both the training and the test set.",
            "4. Results. On the independent test sites the model reached a root-mean-square "
            "error of 0.048 cubic metres per cubic metre and a correlation coefficient of 0.71. "
            "Errors increased with canopy height: sites with trees taller than 15 metres had a "
            "root-mean-square error of 0.067 cubic metres per cubic metre.\n\n"
            "5. Discussion. Canopy attenuation was the main source of error, because the radar "
            "signal from the soil weakens as it passes through dense vegetation. Adding canopy "
            "height as an input reduced the error by 18 percent compared with a model using "
            "backscatter and incidence angle only. Future work should test L-band radar, which "
            "penetrates vegetation better than C-band.",
        ],
    },
    "battery": {
        "title": "Capacity Fade of Lithium Iron Phosphate Cells under Cold Cycling",
        "author": "Synthetic Author C",
        "pages": [
            "Capacity Fade of Lithium Iron Phosphate Cells under Cold Cycling\n\n"
            "Abstract. This synthetic study measures how low temperature affects the ageing of "
            "lithium iron phosphate (LFP) cells. After 800 cycles, cells cycled at -10 degrees "
            "Celsius retained 82 percent of their initial capacity, compared with 95 percent "
            "for cells cycled at 25 degrees Celsius.\n\n"
            "1. Introduction. LFP batteries are widely used in electric buses and stationary "
            "storage because of their long cycle life and thermal stability. In cold climates, "
            "however, charging at low temperature can cause lithium plating on the graphite "
            "anode, which permanently removes active lithium from the cell.",
            "2. Experimental. Twenty-four commercial 2.5 Ah cylindrical LFP cells were divided "
            "into three groups of eight. The groups were cycled at -10, 10 and 25 degrees "
            "Celsius in climate chambers. All cells were charged and discharged at a rate of 1C "
            "between 2.5 and 3.65 V for 800 cycles. Every 100 cycles the capacity was measured "
            "at 25 degrees Celsius with a C/10 rate, and electrochemical impedance spectroscopy "
            "was recorded at 50 percent state of charge.",
            "3. Results. Capacity retention after 800 cycles was 82 percent at -10 degrees "
            "Celsius, 91 percent at 10 degrees Celsius and 95 percent at 25 degrees Celsius. "
            "Impedance spectra showed that the charge-transfer resistance of the cold-cycled "
            "cells increased by 140 percent, while the ohmic resistance changed little.\n\n"
            "4. Discussion. The faster fade at -10 degrees Celsius is attributed to lithium "
            "plating during charging. In an additional test, inserting a 30-minute rest period "
            "after each charge reduced capacity fade at -10 degrees Celsius by 30 percent, "
            "because plated lithium had time to re-intercalate into the graphite. Preheating "
            "the cells before charging is therefore recommended for cold-climate applications.",
        ],
    },
}

MARGIN = 56  # points


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, article in ARTICLES.items():
        doc = pymupdf.open()
        for text in article["pages"]:  # type: ignore[attr-defined]
            page = doc.new_page(width=595, height=842)  # A4
            rect = pymupdf.Rect(MARGIN, MARGIN, 595 - MARGIN, 842 - MARGIN)
            overflow = page.insert_textbox(rect, text, fontsize=10.5, fontname="helv")
            if overflow < 0:
                raise RuntimeError(f"{name}: page text does not fit ({overflow})")
        doc.set_metadata({"title": str(article["title"]), "author": str(article["author"])})
        path = out_dir / f"{name}.pdf"
        doc.save(path, garbage=4, deflate=True)
        doc.close()
        print(f"wrote {path}")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "pdfs")
