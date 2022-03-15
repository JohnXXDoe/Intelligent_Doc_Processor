import csv
import camelot
import argparse

import huggingface_hub
TRANSFORMERS_OFFLINE = 1

from flair.data import Sentence
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1
from segtok.segmenter import split_single
from tqdm import tqdm
import easyocr
from pdf2image import convert_from_path  # For scanned PDFs
from collections import OrderedDict
from pdfminer import pdfpage
import cv2
import pytesseract
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO, BytesIO
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

from flair.visual.ner_html import render_ner_html, HTML_PAGE, TAGGED_ENTITY, PARAGRAPH
import tempfile
import webbrowser


def pdf2img(pdf, name, pagenums=None):
    '''

    :param pdf: Pdf location
    :param name: Pdf name for image file
    :param pagenums: page number to be converted
    :return: save pages as jpeg images
    '''
    # print(len(PDF))
    global total_pages
    pages = convert_from_path(pdf, 500, poppler_path=r"C:\poppler-0.68.0\bin", timeout=10000, first_page=pagenums,
                              last_page=pagenums)

    #
    # # Counter to store images of each page of PDF to image
    image_counter = 1
    #
    # Iterate through all the pages stored above
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> pdfname_1.jpg
        # PDF page 2 -> pdfname_2.jpg
        # PDF page 3 -> pdfname_3.jpg
        # . . . .
        # PDF page n -> pdfname_n.jpg
        filename = f'{name}_' + str(image_counter) + ".png"

        # Save the image of the page in system
        page.save(r'C:/Data/Output/OCR/images/' + filename)
        # print('Saved page number ' + str(image_counter))
        # Increment the counter to update filename
        image_counter = image_counter + 1
    total_pages = image_counter


def searchable_ocr(img):  # From image to searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(img, config='--oem 1', extension='pdf')
    print(pytesseract.image_to_string(img))
    with open(r'C:\Data\test\Searchable.pdf', 'w+b') as f:
        f.write(pdf)


def table_extraction(pdf, name, page):
    tables, text = camelot.read_pdf(pdf, strip_text='\n', pages=str(page), backend="poppler", split_text=True,
                                    process_background=False, copy_text=['h', 'v'], line_scale=60,
                                    layout_kwargs={'char_margin': 1, 'line_margin': 0.2, 'boxes_flow': 1})
    tables.export(f'C:/Data/Output/tables/{name}table.html', f='html',
                  compress=False)  # json, excel, html, markdown, sqlite
    # print(tables.export(r'C:\Data\Output\tables\table.txt', f='txt'))
    tablesfin, line, dic, header, tables_list = [], '', {}, 0, []
    for table in text:
        for row_index, row in enumerate(table):
            para = []
            for col_index, col in enumerate(row):
                dic.setdefault(f'Col{col_index}', [])
                if row_index <= header:
                    if table[row_index][col_index] not in dic.get(f'Col{col_index}', ''):
                        dic[f'Col{col_index}'].append(table[row_index][col_index])
                if table[row_index][col_index] in dic.get(f'Col{col_index}', ''):
                    header = row_index
                else:
                    head = ' '.join(dic.get(f'Col{col_index}', ''))
                    line = f'{head} - {table[row_index][col_index]},'
                para.append(line)
            lines = ' '.join(para)
            tables_list.append(lines)
        tablesfin.append(tables_list)

    # for table in tablesfin:
    # print(f'\n ------ TABLES ------\n {table}\n')

    if tablesfin:
        return tablesfin
    else:
        return None


def img_ocr(location, filename):  # For Image/Scanned PDF to text
    total_text = ''
    for page in range(1, total_pages):  # tqdm(range(1, total_pages), desc='Converting images to text. . .'):
        loc = f'{location}/{filename}_{page}.png'
        image = cv2.imread(loc)
        reader = easyocr.Reader(['en'],
                                recog_network='custom_example')  # , recog_network='custom_example' this needs to run only once to load the model into memory
        result = reader.readtext(loc, height_ths=0.2,
                                 ycenter_ths=0.3, width_ths=0.5, paragraph=True, decoder='wordbeamsearch', y_ths=0.2,
                                 x_ths=50)

        # paragraph=True)  # , rotation_info=[90, 180, 270], y_ths=1, x_ths=0.09, height_ths=0.5, ycenter_ths=0.5, width_ths=0.5
        cv2.startWindowThread()
        for (bbox, text) in result:  # , prob
            # display the OCR'd text and associated probability
            # print("[INFO] {:.4f}: {}".format(prob, text))
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

        file = open(f"C:/Data/Output/OCR/{filename}_OCR.txt", 'a')
        for (bbox, text) in result:  # , prob
            total_text += str(text) + '\n'
            file.write(str(text))
            file.write('\n')
        file.close()
        # show the output image
        cv2.namedWindow('PDF Output', cv2.WINDOW_NORMAL)
        cv2.imshow("PDF Output", image)
        cv2.waitKey(20)
    # print(f'FINAL PAGE TEXT : {total_text}')
    return str(total_text)


def ner(pdf, titles, im_loc):
    i = 1
    table_sent = []
    data = ''
    tagger = SequenceTagger.load(
        r'E:\PycharmProjects\DL\Doc_IMG-OCR\trainer\resources\taggers\full-fixed-roberta-base\best-model.pt')  # all-fixed-roberta-base-resume
    # print(tagger)
    tables = []
    rsrcmgr = PDFResourceManager()
    retstr = BytesIO()
    codec = 'utf-8'
    laparams = LAParams(char_margin=15, line_margin=2, boxes_flow=1)
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
        if pagenum is not None:
            interpreter.process_page(page)
            if len(retstr.getvalue()) < 10:
                # print(f'>> OCR PAGE >>{retstr.getvalue()} <<<<<<< Page number: {pagenum + 1}<<<<< ! ! ! ')
                # Page is OCR only
                pdf2img(pdf, titles, pagenums=pagenum)  # Convert page to image
                data += img_ocr(im_loc, titles)  # Get OCR from converted image
            else:
                try:
                    page_tables = table_extraction(pdf, titles, pagenum)  # Returns list of tables in the specified page
                except IndexError:
                    print(f'Page  {pagenum} Table not readable. Skipping it.')
                if page_tables:
                    for table in page_tables:
                        tables.append(table)  # Save tables in universal 'tables' list
                data += retstr.getvalue().decode('ascii', 'ignore')
                data = data.replace('\x0c', ' ')
                # print(f'::PAGE IS NORMAL AND EXTRACTABLE::')
            retstr.truncate(0)
            retstr.seek(0)
        pagenum += 1
        pbar.update(1)
    pbar.close()
    # data = retstr.getvalue()
    # encoded_string = data.encode("ascii", "ignore")
    # clean = encoded_string.decode()
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(data)

    for pages in tqdm(tables, desc=f'Predicting Tables . . .'):
        for table_no, multi_table in enumerate(pages):
            table_sent.append(Sentence(multi_table, use_tokenizer=True))

    for table_lines in table_sent:
        tagger.predict(table_lines)
    for num, sentence in enumerate(tqdm(sentences, desc=f'Predicting labels . . .')):
        tagger.predict(sentence)

    ###################
    # LOG
    ##################
    logfile = f'C:/Data/Output/{titles}_summary.txt'
    dic = {}  # Declare dictionary for removing duplicate sentences
    with open(logfile, 'w', newline='', encoding="utf-8") as f:
        print('Writing values to file. . . ')
        print(f'////////////////////////////////////////////////////////////////////////////////')
        print(f'//////////////////  E X T R A C T I O N    R E S U L T  ///////////////////////')
        print(f'-------------------------------------------------------------------------------')
        print(f'//  Text ,   Entity Tag ,  Confidence percentage   //')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////  E X T R A C T I O N    R E S U L T  //////////////////////// \n')
        f.writelines(f'//////////////////      Text, Entity - [Confidence]    ///////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'------------------------------------------------------------------------------- \n\n\n')
        for sentence in sentences:
            dic.setdefault(sentence.to_plain_string(), [])
            for entity in sentence.get_spans('ner', min_score=threshold):
                if str(entity.tag) != 'tenderid':
                    dic[sentence.to_plain_string()].append(
                        f'> {entity.text}, {entity.tag} - [{(round(entity.score, 4) * 100)}%]\n')
                    # f.writelines(f'> {entity.text}, {entity.tag}-[{(round(entity.score, 4) * 100)}%] \n')
                    # f.writelines(f'>> {sentence.to_original_text()}, {entity.tag} \n\n')
                    print(f'// =={entity.text}  ====  {entity.tag} :::: {(round(entity.score, 4) * 100)}% :::://')
        print(f'|______________________________________________________________________________|')
        for k, v in dic.items():
            if len(v) > 0:
                res = list(OrderedDict.fromkeys(v))
                for tags in res:
                    f.writelines(f'Tags: {tags}')
                f.writelines(f'\nSentence : {k} \n\n')
                f.writelines(f'X----------------------------------X-------------------------------X \n\n')
        tablefile = f'C:/Data/Output/{titles}_table.txt'
        dic2 = {}  # Declare dictionary for removing duplicate sentences
        with open(tablefile, 'w', newline='', encoding="utf-8") as f:
            f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
            f.writelines(f'/////////////////////  T A B L E S     R E S U L T  /////////////////////////// \n')
            f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
            f.writelines(f'------------------------------------------------------------------------------- \n\n\n')

            for sent in table_sent:
                dic2.setdefault(sent.to_plain_string(), [])
                for entity in sent.get_spans('ner', min_score=threshold):
                    if str(entity.tag) != 'tenderid':
                        dic2[sent.to_plain_string()].append(
                            f'> {entity.text}, {entity.tag} - [{(round(entity.score, 4) * 100)}%]\n')
                        # f.writelines(f'> {entity.text}, {entity.tag}-[{(round(entity.score, 4) * 100)}%] \n')
                        # f.writelines(f'>> {sentence.to_original_text()}, {entity.tag} \n\n')
                        print(f'// =={entity.text}  ====  {entity.tag} :::: {(round(entity.score, 4) * 100)}% :::://')
            print(f'|______________________________________________________________________________|')
            for k, v in dic2.items():
                if len(v) > 0:
                    res = list(OrderedDict.fromkeys(v))
                    for tags in res:
                        f.writelines(f'Tags: {tags}')
                    f.writelines(f'\nSentence : {k} \n\n')
                    f.writelines(f'X----------------------------------X-------------------------------X \n\n')

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
    parser.add_argument('-c', dest='threshold', help='Default is to show all tags  , Limit: [0,1]', type=float,
                        default=-1)
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
    PDF_file = f'C:/Data/test/{pdfname}.pdf'
    img_loc = r'C:/Data/Output/OCR/images'

    # pdf2img(PDF_file, pdfname)
    # img_ocr(img_loc, pdfname)
    # searchable_ocr(img_loc)  # For converting image to text embedded PDF
    ner(PDF_file, pdfname, img_loc)

'''
CLI command :
# // E:\PycharmProjects\DL\venv\scripts\python.exe E:\PycharmProjects\DL\Doc_IMG-OCR\main.py -c 0.7 -f EIL //
pdftopng
'''
