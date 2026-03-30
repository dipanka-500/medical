import zipfile

from PIL import Image, ImageDraw

from medicscan_ocr.document import DocumentPreparer


def _make_image(path, text):
    image = Image.new("RGB", (320, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((15, 40), text, fill="black")
    image.save(path)


def test_zip_bundle_extracts_and_orders_pages(tmp_path):
    first = tmp_path / "page_10.png"
    second = tmp_path / "page_2.png"
    _make_image(first, "ten")
    _make_image(second, "two")

    archive_path = tmp_path / "pages.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(first, arcname="page_10.png")
        archive.write(second, arcname="page_2.png")

    preparer = DocumentPreparer(tmp_path / ".artifacts", enable_preprocessing=False)
    prepared = preparer.prepare(str(archive_path))
    assert prepared.source_type == "zip"
    assert len(prepared.pages) == 2
    assert prepared.pages[0].source_name == "page_2.png"
    assert prepared.pages[1].source_name == "page_10.png"


def test_unsafe_zip_is_blocked(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    preparer = DocumentPreparer(tmp_path / ".artifacts", enable_preprocessing=False)
    try:
        preparer.prepare(str(archive_path))
    except ValueError as exc:
        assert "unsafe path" in str(exc).lower()
    else:
        raise AssertionError("Expected unsafe zip extraction to be blocked.")

