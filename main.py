import csv

import easyocr
from pdf2image import convert_from_path  # For scanned PDFs
from PIL import Image
from pdfminer import high_level # Convert text PDF to text
from pdfminer import pdfpage
import cv2
import pytesseract
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

from unittest.mock import MagicMock
from flair.data import Sentence, Span, Token
from flair.visual import *
from flair.visual.ner_html import render_ner_html, HTML_PAGE, TAGGED_ENTITY, PARAGRAPH
import tempfile
import webbrowser



def pdf2img(PDF):
    # print(len(PDF))
    pages = convert_from_path(PDF, 500, poppler_path=r"C:\poppler-21.09.0\Library\bin", first_page=150,
                              last_page=160, timeout=10000)
    #
    # # Counter to store images of each page of PDF to image
    image_counter = 1
    #
    # Iterate through all the pages stored above
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> page_1.jpg
        # PDF page 2 -> page_2.jpg
        # PDF page 3 -> page_3.jpg
        # ....
        # PDF page n -> page_n.jpg
        filename = "page" + str(image_counter) + ".png"

        # Save the image of the page in system
        page.save(r'C:\Users\33669\PycharmProjects\OCR\pdf2img\ ' + filename)
        print('Saved page number ' + str(image_counter))
        # Increment the counter to update filename
        image_counter = image_counter + 1


def searchable_ocr(img):  # From image to searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(img, config='--oem 1', extension='pdf')
    print(pytesseract.image_to_string(img))
    with open(r'C:\Users\33669\PycharmProjects\OCR\outputs\Searchable.pdf', 'w+b') as f:
        f.write(pdf)


def img_ocr(loc):  # For Image/Scanned PDF to text
    image = cv2.imread(loc)
    reader = easyocr.Reader(['en'],
                            recog_network='custom_example')  # , recog_network='custom_example' this needs to run only once to load the model into memory
    result = reader.readtext(img_loc,
                             paragraph=True)  # , rotation_info=[90, 180, 270], y_ths=1, x_ths=0.09, height_ths=0.5, ycenter_ths=0.5, width_ths=0.5
    cv2.startWindowThread()
    for (bbox, text) in result:  # , prob
        # display the OCR'd text and associated probability
        # print("[INFO] {:.4f}: {}".format(prob, text))
        print(text)
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        # cleanup the text and draw the box surrounding the text along
        # with the OCR'd text itself
        cv2.rectangle(image, tl, br, (0, 0, 255), 4)
        cv2.putText(image, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 90, 200), 8)

    file = open(r"C:\Users\33669\PycharmProjects\OCR\outputs\OCR_OUT.txt", 'w')
    for (bbox, text) in result:  # , prob
        file.write(str(text))
        file.write('\n')
    file.close()
    # show the output image
    cv2.namedWindow('PDF Output', cv2.WINDOW_NORMAL)
    cv2.imshow("PDF Output", image)
    cv2.waitKey(0)


def ner(pdf):
    sentences = []
    tagger = SequenceTagger.load(
        r'C:\Users\33669\PycharmProjects\OCR\trainer\resources\taggers\flair-embd\final-model.pt')
    print(tagger)
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams(char_margin=30, line_margin=2, boxes_flow=1)
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    rawdata = []
    fp = open(pdf, 'rb')

    for pagenumber, page in enumerate(pdfpage.PDFPage.get_pages(fp, check_extractable=True)):
        # print(pagenumber)
        if pagenumber:
            interpreter.process_page(page)
            data = retstr.getvalue()
            # cleansent = clean(sent)
            encoded_string = data.encode("ascii", "ignore")
            cleanpage = encoded_string.decode()
            splitter = SegtokSentenceSplitter()
            sentences = splitter.split(cleanpage)
            tagger.predict(sentences)
            for sentence in sentences:
                print(sentence.to_tagged_string())
    colors = {
        "default": "#F7FF53",
        "O": "#ddd",
    }
    actual = render_ner_html(sentences, colors=colors)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(actual)
    webbrowser.open(url)

def mock_ner_span(text, tag, start, end):
    span = Span([]).set_label('class', tag)
    span.start_pos = start
    span.end_pos = end
    span.tokens = [Token(text[start:end])]
    return span


def test_html_rendering():
    text = (
        "Boris Johnson has been elected new Conservative leader in a ballot of party members and will become the "
        "next UK prime minister. &"
    )
    sent = Sentence()
    sent.get_spans = MagicMock()
    sent.get_spans.return_value = [
        mock_ner_span(text, "PER", 0, 13),
        mock_ner_span(text, "MISC", 35, 47),
        mock_ner_span(text, "LOC", 109, 111),
    ]
    sent.to_original_text = MagicMock()
    sent.to_original_text.return_value = text
    colors = {
        "PER": "#F7FF53",
        "ORG": "#E8902E",
        "LOC": "yellow",
        "MISC": "#4647EB",
        "O": "#ddd",
    }
    actual = render_ner_html([sent], colors=colors)

    expected_res = HTML_PAGE.format(
        text=PARAGRAPH.format(
            sentence=TAGGED_ENTITY.format(
                color="#F7FF53", entity="Boris Johnson", label="PER"
            )
                     + " has been elected new "
                     + TAGGED_ENTITY.format(color="#4647EB", entity="Conservative", label="MISC")
                     + " leader in a ballot of party members and will become the next "
                     + TAGGED_ENTITY.format(color="yellow", entity="UK", label="LOC")
                     + " prime minister. &amp;"
        ),
        title="Flair",
    )

    html = actual

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html)
    webbrowser.open(url)
    assert expected_res == actual


if __name__ == '__main__':
    pdfname = 'EIL Specs'
    PDF_file = r'H:\Code\Data\EIL Specs.pdf'
    img_loc = r'C:\Users\33669\PycharmProjects\OCR\pdf2img\K2.jpg'
    filename = 'PGCIL'
    # img_ocr(img_loc)
    # pdf2img(PDF_file)
    # searchable_ocr(img_loc) # For converting image to text embedded PDF
    ner(PDF_file)
    #test_html_rendering()
