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
    """
    Takes PDF page and converts it to png image for running OCR
    :param pdf: Pdf location
    :param name: Pdf name for image file
    :param pagenums: page number to be converted
    :return: save pages as jpeg images
    """

    # print(len(PDF))
    global total_pages
    pages = convert_from_path(pdf, 500, poppler_path=r"C:\poppler-0.68.0\bin", timeout=10000, first_page=pagenums,
                              last_page=pagenums)

    # Counter to store images of each page of PDF to image
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


def table_extraction(pdf, name, page, type):
    """
    Find any lattice tables in the page and convert its rows to syntactically sane text sentences for NER run
    :param pdf: Location of PDF file
    :param name: Name of PDF file
    :param page: Page numbers to find tables in
    :param type: If OCR or regular PDF page
    :return: Text string that contains row vise sentences
    """
    if type == 'lattice':
        try:
            tables, text = camelot.read_pdf(pdf, flavor='lattice', strip_text='\n', pages=str(page), backend="poppler",
                                            split_text=True,
                                            process_background=False, copy_text=['h', 'v'], line_scale=50, )
            # layout_kwargs={'char_margin': 1, 'line_margin': 0.2,
            #                'boxes_flow': 1})  # Text based page
        except ZeroDivisionError:  # if (bbox_intersection_area(ba, bb) / bbox_area(ba)) > 0.8: ZeroDivisionError:
            # float division by zero
            print('Zero Division Error')
            return None
    else:
        tables, text = camelot.read_pdf(pdf, flavor='lattice_ocr', pages=str(page))  # OCR based page

    tables.export(f'C:/Data/Output/{name} tables.csv', f='txt',
                  compress=True)  # json, excel, html, markdown, sqlite, txt
    # print(tables.export(r'C:\Data\Output\tables\table.txt', f='txt'))
    tablesfin, item, dic, header, table_line = [], '', {}, 0, []
    for table in text:
        for row_index, row in enumerate(table):
            items = []
            for col_index, col in enumerate(row):
                dic.setdefault(f'Col{col_index}', [])
                if row_index <= header:
                    if table[row_index][col_index] not in dic.get(f'Col{col_index}', ''):
                        dic[f'Col{col_index}'].append(table[row_index][col_index])
                if table[row_index][col_index] in dic.get(f'Col{col_index}', ''):
                    header = row_index
                else:
                    head = ' '.join(dic.get(f'Col{col_index}', ''))
                    item = f'{head} - {table[row_index][col_index]},'
                items.append(item)
            row_line = ' '.join(items)
            table_line.append(row_line)
        tablesfin.append(table_line)

    # for table in tablesfin:
    # print(f'\n ------ TABLES ------\n {table}\n')

    if tablesfin:
        return tablesfin
    else:
        return None


def img_ocr(location, filename):  # For Image/Scanned PDF to text
    """
    Opens PNG image (single page) and runs OCR model to extract text
    :param location: Location of PNG image
    :param filename: Name of PNG image
    :return: Text extracted from scanned image (string)
    """
    total_text = ''
    for page in range(1, total_pages):  # tqdm(range(1, total_pages), desc='Converting images to text. . .'):
        loc = f'{location}/{filename}_{page}.png'
        image = cv2.imread(loc)
        reader = easyocr.Reader(['en'],
                                recog_network='custom_example')  # recog_network='custom_example' this needs to run only once to load the model into memory
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
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 90, 200), 5)

        file = open(f"C:/Data/Output/OCR/{filename}_OCR.txt", 'a')
        for (bbox, text) in result:  # , prob
            total_text += str(text) + '\n'
            file.write(str(text))
            file.write('\n')
        file.close()
        # show the output image
        cv2.namedWindow('PDF Output', cv2.WINDOW_NORMAL)
        cv2.imshow("PDF Output", image)
        cv2.waitKey(2)
    # print(f'FINAL PAGE TEXT : {total_text}')
    return str(total_text)


def read_dic():
    """
    Read dictionary mappings for converting entity names to Normal text
    :return: None
    """
    global key_mappings
    key_mappings = {}
    with open('./Outputs/EntityDic.text', 'r') as f:
        for line in f:
            key, value = line.split(',')
            key_mappings.fromkeys(key)
            key_mappings[key] = value


def ner(pdf, titles, im_loc):
    """
    Takes PDF runs it page by page to extract its text using OCR or text extraction to run NER model and save its output as text summary and temp HTML render
    :param pdf: PDF file location
    :param titles: Name of PDF file
    :param im_loc: Location for saving PNG image in case of scanned page
    :return: None
    """
    i = 1
    table_sent = []
    data = ''
    tagger = SequenceTagger.load(
        r'E:\PycharmProjects\DL\Doc_IMG-OCR\trainer\resources\taggers\roberta-manul-strd/final-model.pt')  # all-fixed-roberta-base-resume
    # print(tagger)
    tables = []
    open(f"C:/Data/Output/{titles} tables.csv", "w").close()  # Clear/Wipe if there is older version of table.csv
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
        if pagenum is not None:  # is not None:
            interpreter.process_page(page)
            if len(retstr.getvalue()) < 10:
                # print(f'>> OCR PAGE >>{retstr.getvalue()} <<<<<<< Page number: {pagenum + 1}<<<<< ! ! ! ')
                # Page is OCR only
                pdf2img(pdf, titles, pagenums=pagenum)  # Convert page to image
                data += img_ocr(im_loc, titles)  # Get OCR from converted image
                try:
                    page_tables = table_extraction(pdf, titles, pagenum,
                                                   'lattice_ocr')  # Run OCR based table extraction
                except:  # Outer line missing in table
                    print(f'Page  {pagenum} Table not readable. Skipping it.')
                    page_tables = None
                    continue
            else:
                try:
                    page_tables = table_extraction(pdf, titles, pagenum,
                                                   'lattice')  # Returns list of tables in the specified page
                except:
                    page_tables = None
                    print(f'Page  {pagenum} Table not readable. Skipping it.')

                data += retstr.getvalue().decode('ascii', 'ignore')  # add extracted text from bytesIO to data variable
                data = data.replace('\x0c', ' ')  # Remove useless character
            if page_tables:
                tok_table_lines, tok_line, extraction = [], None, []
                # NEW LOGIC
                for table in page_tables:
                    for line in table:
                        tok_table_lines.append(Sentence(line, use_tokenizer=True))
                for tok_line in tok_table_lines:
                    tagger.predict(tok_line)
                with open(f"C:/Data/Output/{titles} tables.csv", 'a', newline='', encoding="utf-8") as f:
                    for tok_line in tok_table_lines:
                        for entity in tok_line.get_spans('ner', min_score=threshold):
                            if str(entity.tag) != 'tenderid' and entity.tag != 'marking':
                                print(f'-- Adding table extraction to CSV file --')
                                extraction.append(f'"{entity.text} , {entity.tag}"')
                    if extraction:
                        f.write("-------------------------- , ---------------------- \n")
                        f.write("Attribute , Type")
                        f.write("\n")
                        res = list(OrderedDict.fromkeys(extraction))
                        for tags in res:
                            f.write(tags)
                            f.write("\n")
                        f.write("\n -------------------------- , ---------------------- \n")
                '''
                #OLD LOGIC
                for table in page_tables:
                    tables.append(table)  # Save tables in universal 'tables' list
                '''
                # print(f'::PAGE IS NORMAL AND EXTRACTABLE::')
            retstr.truncate(0)
            retstr.seek(0)
        pagenum += 1
        pbar.update(1)
    pbar.close()
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(data)
    '''
    for pages in tqdm(tables, desc=f'Converting Tables . . .'):
        for table_no, multi_table in enumerate(pages):
            table_sent.append(Sentence(multi_table, use_tokenizer=True))

    for table_lines in table_sent:
        tagger.predict(table_lines)
    '''
    for num, sentence in enumerate(tqdm(sentences, desc=f'Predicting labels . . .')):
        tagger.predict(sentence)

    ###################
    # LOG OUTPUT
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
        f.writelines(f'//////////////////     Text, Entity - [Confidence]     ///////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'------------------------------------------------------------------------------- \n\n\n')
        cable_list = ['pvc insulated', 'xlpe insulated',
                      'cable',
                      'cables']  # 'cable', 'lt', 'lt cable', 'cables']
        forbidden = ['accessory', 'accessories', 'standard', 'standards', 'cable and accessory', 'cable and accessories']
        cable_flag = 0
        for sentence in sentences:
            dic.setdefault(sentence.to_plain_string(), [])  # Create list initialised dictionary where Key = sentence
            for entity in sentence.get_spans('ner', min_score=threshold):

                #########################################################
                #        CUSTOMISATION FOR TENDER SPECIFICATIONS        #
                #########################################################

                if str(entity.tag) != 'tenderid' and str(entity.tag) != 'standard':
                    '''
                    # Add entity name normalization logic using dictionaries
                    if entity.tag in key_mappings:
                        OG_ent = key_mappings[entity.tag]
                    dic[sentence.to_plain_string()].append(
                        f'> {entity.text}, {OG_ent} - [{(round(entity.score, 4) * 100)}%]\n')
                    '''
                    if str(entity.tag) == 'cableItype':  # Cable type subset detection logic
                        print(f'Cable Type enitity detected  - {entity.text}')
                        for x in forbidden:  # Filtering results of Cable Type
                            if entity.text.lower().find(x) == -1:
                                for y in cable_list:
                                    if entity.text.lower().find(y) != -1:
                                        print(f'======= Cable Type set {x} =======')
                                        cable_flag = 1
                                        cable_name = entity.text
                                        break
                            else:  # Else set flag = 0
                                cable_flag = 0

                    if cable_flag == 1 and entity.tag != 'cableItype':
                        if entity.tag in ['marking', 'packing'] and len(
                                entity) > 2:  # Removing entity output and less than 2 word entities from final text for markings
                            dic[sentence.to_plain_string()].append(
                                f'Tag: >> {entity.tag} |> {cable_name}')  # Adding specific formatted line to final text file
                            print(
                                f'// =={entity.text}  ====  {entity.tag} :::: {(round(entity.score, 4) * 100)}% :::://')  # Debugging/CLI output
                            continue
                        dic[sentence.to_plain_string()].append(  # Adding specific formatted line to final text file
                            f'Tag: >> {entity.text}, {entity.tag} |> {cable_name} - [{(round(entity.score, 4) * 100)}%]\n')
                        # f.writelines(f'> {entity.text}, {entity.tag}-[{(round(entity.score, 4) * 100)}%] \n')
                        # f.writelines(f'>> {sentence.to_original_text()}, {entity.tag} \n\n')
                        print(
                            f'// =={entity.text}  ====  {entity.tag} :::: {(round(entity.score, 4) * 100)}% :::://')  # Debugging/CLI output

        print(f'|______________________________________________________________________________|')

        for k, v in dic.items():
            if len(v) > 0:
                res = list(OrderedDict.fromkeys(v))  # To remove multiple same Keys from different similar sentences
                for tags in res:
                    f.writelines(f'{tags}')
                f.writelines(f'\nSentence : {k} \n\n')
                f.writelines(f'X----------------------------------X-------------------------------X \n\n')
        tablefile = f'C:/Data/Output/{titles}_table.txt'
        dic2 = {}  # Declare dictionary for removing duplicate sentences in tables
        with open(tablefile, 'w', newline='', encoding="utf-8") as f:
            f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
            f.writelines(f'/////////////////////  T A B L E S     R E S U L T  /////////////////////////// \n')
            f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
            f.writelines(f'------------------------------------------------------------------------------- \n\n\n')

            for sent in table_sent:
                dic2.setdefault(sent.to_plain_string(), [])
                for entity in sent.get_spans('ner', min_score=threshold):
                    if str(entity.tag) != 'tenderid' and entity.tag != 'marking' and entity.tag != 'cableItype':
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
    parser.add_argument('-c', dest='threshold', help='Default = 0.8  , Limit: [0,1]', type=float,
                        default=0.8)
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

"""
CLI command :
                E:\PycharmProjects\DL\venv\scripts\python.exe E:\PycharmProjects\DL\Doc_IMG-OCR\main.py -c 0.7 -f EIL //
"""
