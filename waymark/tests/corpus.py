from __future__ import annotations

import random
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

PAPERS: list[tuple[str, list[str]]] = [
    (
        "shifting_sands_survey.pdf",
        [
            "Shifting Sands: A Dune Migration Survey of the Empty Quarter was written by Dr. Lin Chen and published in 2019. "
            "The survey tracked barchan dunes along the eastern margin of the Rub al Khali for six seasons.",
            "Chen measured migration rates with repeat drone photogrammetry. Chen reported that the fastest dunes moved 18 metres a year, "
            "far faster than the coastal sabkha formations.",
            "The fieldwork relied on soil moisture probes calibrated against readings from the Liwa Oasis Research Station.",
        ],
    ),
    (
        "chen_biography.pdf",
        [
            "Lin Chen is a hydrologist who studies groundwater beneath desert dunes. Lin Chen trained at the Qasr Al Sarab Field Laboratory "
            "before joining the Liwa Oasis Research Station in 2015.",
            "During her years at the field laboratory, Chen built the first low-cost aquifer sensor network in the region. "
            "She was born in Kunming and later moved to Abu Dhabi.",
        ],
    ),
    (
        "field_lab_history.pdf",
        [
            "The Qasr Al Sarab Field Laboratory was established in 2004 near Hamim. The Qasr Al Sarab Field Laboratory is funded by the "
            "Al Dhafra Heritage Institute, which also pays for the annual date palm census.",
            "The laboratory director, Omar Al Mansoori, expanded the sensor programme in 2011.",
        ],
    ),
    (
        "heritage_institute_profile.pdf",
        [
            "The Al Dhafra Heritage Institute was founded in 1996 in Madinat Zayed. The institute is chaired by Fatima Al Hameli.",
            "Its archive holds the oral histories of falconers and camel breeders from the western region.",
        ],
    ),
    (
        "investment_note.pdf",
        [
            "Wei Chen is an economist at the Abu Dhabi Investment Office. Wei Chen wrote the 2021 note on logistics corridors.",
            "In the note, Chen argued that rail freight would cut transport costs for the Ruwais refinery by a third.",
            "Chen previously worked at the Gulf Monetary Bureau in Manama.",
        ],
    ),
    (
        "jordan_water_brief.pdf",
        [
            "Jordan faces one of the most severe water shortages in the world. Water ministries in Jordan rely on the Disi aquifer.",
            "The brief compares Jordan with the United Arab Emirates, where desalination supplies most drinking water.",
        ],
    ),
    (
        "hale_profile.pdf",
        [
            "Jordan Hale is a geologist at Mercury Labs in Dubai. Hale mapped the gypsum crusts of the Liwa crescent in 2018.",
            "Hale collaborated with Lin Chen on a paper about crust strength, which Hale presented in Muscat.",
        ],
    ),
    (
        "mercury_contamination.pdf",
        [
            "Mercury contamination in sabkha sediments was traced to flaring at the Ruwais refinery.",
            "Samples were analysed at Mercury Labs, a private company founded in 2009 by Rashid Al Suwaidi.",
        ],
    ),
    (
        "station_history.pdf",
        [
            "The Liwa Oasis Research Station was founded in 1998. The station is operated by the Ministry of Climate Change and Environment.",
            "Its first director was Salem Al Mazrouei, who had trained as a soil scientist in Aberdeen.",
        ],
    ),
]

FILLER = [
    "Wind regimes over the inland dunes change little from year to year, though summer shamal events can reverse slip faces.",
    "Field teams log temperature, humidity and grain size at dawn, when the surface is most stable.",
    "Calibration runs are repeated every quarter to control for drift in the capacitance probes.",
    "Satellite imagery provides context, but ground truth still depends on transects walked by hand.",
    "The archive stores raw logger files alongside cleaned tables so that later teams can repeat the analysis.",
    "Vegetation cover remains sparse, dominated by Calligonum shrubs and scattered Haloxylon.",
    "Sediment cores were sectioned at two centimetre intervals and stored at four degrees.",
    "Most of the instruments are solar powered and transmit summaries over a cellular link.",
]


def _write_pdf(path: Path, paragraphs: list[str], pad_pages: int = 0, seed: int = 0) -> None:
    rng = random.Random(seed)
    doc = fitz.open()
    blocks = list(paragraphs)
    if pad_pages:
        padded: list[str] = []
        spread = max(1, pad_pages // max(len(paragraphs), 1))
        for para in paragraphs:
            padded.append(para)
            for _ in range(spread):
                padded.append(" ".join(rng.choice(FILLER) for _ in range(9)))
        blocks = padded
    for block in blocks:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(56, 64, 539, 790), block, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


def _write_image(path: Path, lines: list[str]) -> None:
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 30)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    image = Image.new("RGB", (1300, 120 + 60 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 1290, image.height - 10], outline=(40, 70, 90), width=3)
    for i, line in enumerate(lines):
        draw.text((40, 50 + 60 * i), line, fill=(20, 20, 20), font=font)
    image.save(path)


def build_corpus(target: Path, long_pages: int = 0) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (name, paragraphs) in enumerate(PAPERS):
        path = target / name
        pad = long_pages if name == "shifting_sands_survey.pdf" else 0
        _write_pdf(path, paragraphs, pad_pages=pad, seed=i)
        paths.append(path)
    image = target / "station_poster.png"
    _write_image(image, [
        "Liwa Oasis Research Station - visitor notice",
        "The station's weather mast was installed by Mercury Labs.",
        "Night tours are guided by Salem Al Mazrouei.",
    ])
    paths.append(image)
    return paths
