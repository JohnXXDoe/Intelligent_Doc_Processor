import csv
import argparse

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1
from tqdm import tqdm
import easyocr
from pdf2image import convert_from_path  # For scanned PDFs
import time
from pdfminer import high_level  # Convert text PDF to text
from pdfminer import pdfpage
import cv2
import pytesseract
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

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


def ner(pdf, titles, limit):
    i = 1
    sentences = []
    tagger = SequenceTagger.load(
        r'E:\PycharmProjects\DL\Doc_IMG-OCR\trainer\resources\taggers\full-fixed-roberta-base\best-model.pt') #all-fixed-roberta-base-resume
    print(tagger)
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams(char_margin=30, line_margin=2, boxes_flow=1)
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    fp = open(pdf, 'rb')
    parser = PDFParser(fp)  # For getting total pages
    document = PDFDocument(parser)  # For getting total pages
    total_pages = resolve1(document.catalog['Pages'])['Count']  # For making progress bar
    pbar = tqdm(total=total_pages, desc='Reading PDF')
    ##############
    # Prediction
    ##############

    for pagenum, page in enumerate(pdfpage.PDFPage.get_pages(fp, check_extractable=True)):
        if pagenum:
            interpreter.process_page(page)
        pbar.update(1)
    pbar.close()
    data = retstr.getvalue()
    encoded_string = data.encode("ascii", "ignore")
    clean = encoded_string.decode()
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(clean)
    for num, sentence in enumerate(tqdm(sentences, desc=f'Predicting labels . . .')):
        tagger.predict(sentence)

    ###################
    # LOG
    ##################
    logfile = f'C:\Data\Output\{titles}_summary.txt'
    with open(logfile, 'w', newline='', encoding="utf-8") as f:
        print('Writing values to file. . . ')
        print(f'////////////////////////////////////////////////////////////////////////////////')
        print(f'//////////////////  E X T R A C T I O N    R E S U L T  ///////////////////////')
        print(f'-------------------------------------------------------------------------------')
        print(f'//  Text ,   Entity Tag ,  Confidence percentage   //')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////  E X T R A C T I O N    R E S U L T  /////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//  Text ,   Entity Tag ,  Confidence percentage   //')
        f.writelines(f'------------------------------------------------------------------------------- \n')
        for sentence in sentences:
            for entity in sentence.get_spans('ner', min_score=threshold):
                f.writelines(f'> {entity.text}, {entity.tag}[{(round(entity.score, 4) * 100)}% ] \n')
                f.writelines(f'>> {sentence.to_original_text()}, {entity.tag} \n')
                print(f'// =={entity.text}  ====  {entity.tag} :::: {(round(entity.score, 4) * 100)}% :::://')
                print(f'// =={sentence.to_original_text()}  ====  {entity.tag} :::: :::://')
        print(f'|______________________________________________________________________________|')

    colors = {

        "default": "#FF40A3",
        "O": "#ddd",
    }
    actual = render_ner_html(sentences, title=titles, colors=colors)

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(actual)
    webbrowser.open(url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='threshold', help='Default is to show all tags  , Limit: [0,1]', type=float)
    parser.add_argument('-f', dest='pdfname', help='Name of PDF to be processed', type=str)
    args = parser.parse_args()

    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'///////////////////     D O C U M E N T - N E R - T E S T    ///////////////////')
    print(f'///////////////////             HAVELLS NEW TECH             ///////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'-------------------------------------------------------------------------------')
    print(f'|::::::::::::::::   Threshold Confidence : {args.threshold}   :::::::::::::::::|')
    print(f'|::::::::::::::::     PDF to be evaluated : {args.pdfname}    :::::::::::::::::|')
    print(f'|______________________________________________________________________________|')

    threshold = args.threshold
    pdfname = args.pdfname

    # pdfname = 'PGCIL'#PGCIL BSES TENDER EIL Specs BHEL
    PDF_file = r'C:\Data\test/' + pdfname + '.pdf'
    img_loc = r'C:\Users\33669\PycharmProjects\OCR\pdf2img\K2.jpg'

    # img_ocr(img_loc)
    # pdf2img(PDF_file)
    # searchable_ocr(img_loc) # For converting image to text embedded PDF
    ner(PDF_file, pdfname, threshold)

'''
CLI command :
# // E:\PycharmProjects\DL\venv\scripts\python.exe E:\PycharmProjects\DL\Doc_IMG-OCR\main.py -c 0.7 -f EIL //
'''