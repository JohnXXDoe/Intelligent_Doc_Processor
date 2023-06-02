# GCP Integration
import logging

import pdfminer

log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
# Set logging level for root logger (To capture package Errors)
logging.basicConfig(filename='IDP_run.log', filemode='a', format=log_format,
                    level=logging.INFO)

# Set logging level for IDP logger (To capture main.py Errors)
log_idp = logging.getLogger(__name__)
# hdlr = logging.StreamHandler()
# hdlr.setLevel(logging.ERROR)  # Only show Error level in console
fhdlr = logging.FileHandler("IDP_run.log")
formatter = logging.Formatter(log_format)
fhdlr.setFormatter(formatter)

# log_idp.addHandler(hdlr)
# log_idp.addHandler(fhdlr)
log_idp.setLevel(logging.INFO)

from google.cloud import storage
import time
import camelot
import csv
import os
import pyodbc
import io
import json
import re

TRANSFORMERS_OFFLINE = 1
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1
from segtok.segmenter import split_single
import easyocr
from pdf2image import convert_from_path  # For scanned PDFs
from collections import OrderedDict
from pdfminer import pdfpage
import cv2
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO, BytesIO
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from flair.visual.ner_html import render_ner_html, HTML_PAGE, TAGGED_ENTITY, PARAGRAPH

#####################
# ONLY FOR DEMO USE #
#####################
import warnings

warnings.filterwarnings("ignore")

#####################
# ONLY FOR DEMO USE #
#####################


logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('flair').setLevel(logging.ERROR)
logging.getLogger('camelot').setLevel(logging.ERROR)
logging.getLogger('pyodbc').setLevel(logging.INFO)

# Read file locations
file_locs = []
with open(r'./inputs/paths.csv', mode='r') as csv_file:
    csvFile = csv.reader(csv_file, delimiter=',')
    for col in csvFile:
        file_locs.append(col[0].strip(' '))
# Read SQL credentials
with io.open(r'./inputs/intelligent-document-processor-sql-info.json', "r", encoding="utf-8") as json_fi:
    credentials_info = json.load(json_fi)

pop_path = file_locs[0]
img_loc = file_locs[1]
pdffile_path = file_locs[2]
out_path = file_locs[3]
path_to_private_key = file_locs[4]  # GCP bucket details
ner_model_loc = file_locs[5]
ocr_model = file_locs[6]
ocr_network = file_locs[7]
bucket_name = file_locs[8]

# Create output path if it doesnot exist
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(pdffile_path):
    os.makedirs(pdffile_path)
if not os.path.exists(img_loc):
    os.makedirs(img_loc)

# Create a client for interacting with GCS
client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)
# Use the client to get the bucket object
bucket = storage.Bucket(client, bucket_name)


######################
# MSSQL Integrations #
######################

def getFiles():
    id = []
    name = []
    connection_string = (
        f"Driver={credentials_info.get('Driver')};"
        f"Server={credentials_info.get('Server')};"
        f"Database={credentials_info.get('Database')};"
        f"UID={credentials_info.get('UID')};"
        f"PWD={credentials_info.get('PWD')};"
    )
    connection = pyodbc.connect(connection_string)

    cursor = connection.cursor()

    cursor.execute('exec getfilename')

    for row in cursor:
        name.append(row[0])
        id.append(row[1])

    cursor.commit()
    cursor.close()

    return id, name


def inProcess(flag, filename):
    connection_string = (
        f"Driver={credentials_info.get('Driver')};"
        f"Server={credentials_info.get('Server')};"
        f"Database={credentials_info.get('Database')};"
        f"UID={credentials_info.get('UID')};"
        f"PWD={credentials_info.get('PWD')};"
    )
    connection = pyodbc.connect(connection_string)

    cursor = connection.cursor()

    if flag == True:
        storedProc = "Exec inProc @filename =?"
        params = (f'{filename}.pdf'.upper())
        cursor.execute(storedProc, params)
        cursor.commit()
    cursor.close()


def isGenerated(flag, filename, page_count, time_taken):
    connection_string = (
        f"Driver={credentials_info.get('Driver')};"
        f"Server={credentials_info.get('Server')};"
        f"Database={credentials_info.get('Database')};"
        f"UID={credentials_info.get('UID')};"
        f"PWD={credentials_info.get('PWD')};"
    )
    connection = pyodbc.connect(connection_string)

    cursor = connection.cursor()
    if time_taken <= 1:
        time_taken = "<1"
    if flag == True:
        storedProc = "Exec isGenerated @filenames =?, @totalpages=?, @timetaken=?"
        params = [(f'{filename}.pdf'.upper(), str(page_count), str(time_taken))]
        cursor.executemany(storedProc, params)
        cursor.commit()
    cursor.close()


def store_output(id, outfilename):
    """
    Function to store output in table SQL
    :param id: Document ID for the PDF
    :param outfilename: Output file name for SQL record
    :return: none
    """
    try:
        connection_string = (
            f"Driver={credentials_info.get('Driver')};"
            f"Server={credentials_info.get('Server')};"
            f"Database={credentials_info.get('Database')};"
            f"UID={credentials_info.get('UID')};"
            f"PWD={credentials_info.get('PWD')};"
        )
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()
        storedProc = "Exec storeoutput @id=?, @outfilename =?"
        params = [(id, outfilename)]
        cursor.executemany(storedProc, params)
        cursor.commit()
        log_idp.info("Record inserted successfully in SQL")
    except pyodbc.Error as e:
        log_idp.exception(f"Failed to insert into MySQL table {e}")


def download_from_gcs(filename):
    folder_name = 'Input'  # GCP storage folder name
    # Use the bucket object to get the file object
    file = bucket.get_blob(f'{folder_name}/{filename}')

    # Use the file object to download the file to a local directory
    file.download_to_filename(f'{pdffile_path}/{filename}')
    log_idp.info(f'Downloading {filename} from GCP {bucket_name}. GCP Path: {folder_name}/{filename}')


def upload_to_gcs(filename):
    folder_name = 'Output'
    # Upload the file to the bucket
    blob = bucket.blob(f'{folder_name}/{filename}')
    blob.upload_from_filename(f'{out_path}/{filename}')
    log_idp.info(f"File {filename} was uploaded to the bucket {bucket_name}.")


def delete_local_pdf(location, filename):
    # Get the path of the file
    file_path = f'{location}/{filename}'

    # Check if the file exists
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)
        log_idp.info(f'PDF file deleted from local storage')
    else:
        log_idp.critical(f'PDF could not be deleted! Please check code/directory')


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
    pages = convert_from_path(pdf, 300, poppler_path=pop_path, timeout=10000, first_page=pagenums,
                              last_page=pagenums)
    # Counter to store images of each page of PDF to image
    image_counter = 1
    #
    # Iterate through all the pages stored above
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> pdfname_1.png
        # PDF page 2 -> pdfname_2.png
        # PDF page 3 -> pdfname_3.png
        # . . . .
        # PDF page n -> pdfname_n.png
        filename = f'{name}_' + str(image_counter) + ".png"
        # Save the image of the page in system
        page.save(img_loc + filename)
        # print('Saved page number ' + str(image_counter))
        # Increment the counter to update filename
        image_counter = image_counter + 1
    total_pages = image_counter


def table_extraction(pdf, name, page, page_type):
    """
    Find any lattice tables in the page and convert its rows to syntactically sane text sentences for NER run
    :param pdf: Location of PDF file
    :param name: Name of PDF file
    :param page: Page numbers to find tables in
    :param page_type: If OCR or regular PDF page
    :return: Text string that contains row vise sentences
    """
    if page_type == 'lattice':
        try:
            tables, text_outside_tables = camelot.read_pdf(pdf, flavor='lattice', strip_text='\n', pages=str(page),
                                                           backend="poppler", split_text=True, process_background=False,
                                                           copy_text=['h', 'v'], line_scale=60)
            # layout_kwargs={'char_margin': 1, 'line_margin': 0.2,
            #                'boxes_flow': 1})  # Text based page
        except ZeroDivisionError:  # if (bbox_intersection_area(ba, bb) / bbox_area(ba)) > 0.8: ZeroDivisionError:
            # float division by zero
            log_idp.exception('Zero Division Error while extracting text Table')
            return None
        except NotImplementedError:
            log_idp.critical(
                f'[TABLE_EXTRACTION] Not Implemented error while decoding - only algorithm code 1 and 2 are supported ')
            try:
                tables = camelot.read_pdf(pdf, flavor='lattice_ocr', pages=str(page))  # bypass index error
                log_idp.info(f'\nSwitching to OCR extraction of Table at page {page}')
            except Exception as e:
                log_idp.exception('Index Error - ' + str(e))
                return None
        except IndexError:  # list index out of range:
            # index out of range
            try:
                tables = camelot.read_pdf(pdf, flavor='lattice_ocr', pages=str(page))  # bypass index error
                log_idp.info(f'\nSwitching to OCR extraction of Table at page {page}')
            except Exception as e:
                log_idp.exception('Index Error - ' + str(e))
                return None
    else:
        tables = camelot.read_pdf(pdf, flavor='lattice_ocr', pages=str(page))  # OCR based page

    tables.export(f'{out_path}{name}_tables.csv', f='txt',
                  compress=False)  # json, excel, html, markdown, sqlite, txt
    log_idp.info(f'Table Export complete')

    if page_type == 'lattice' and text_outside_tables is not None:
        return text_outside_tables
    else:
        return None
    # Convert table to paragprah logic >> configure text from readpdf function call. (table, text)
    """
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
    """


def headings_para_split(text):
    """
    Convert text block into headings and paragraphs for better NER rejection
    :param text: string containing PDF text
    :return:
    """
    headings = []
    paragraphs = []
    lines = text.split('\n')
    # print('==============lines are==========',lines)
    current_heading = ""
    current_paragraph = ""
    is_heading = False
    is_paragraph = False
    heading_appended_flag = False
    pattern_special = r'^[^a-zA-Z0-9]'  # str starts with special character
    pattern_digit = r'^\d[\s\S]*$'  # str starts with digit
    headings_count = 1
    para_count = 1
    for line in lines:
        # print('==============lines are==========',line)
        if re.search(pattern_digit, line) and len(line.split()) < 6:
            if is_paragraph:
                paragraphs.append(current_paragraph)
                # print(f'\t Paragraph {para_count} -', current_paragraph, '\n')
                current_paragraph = ""
                is_paragraph = False
                para_count += 1
            current_heading = line
            is_heading = True

        elif line.isupper() or line.istitle():
            if is_paragraph:
                paragraphs.append(current_paragraph)
                # print(f'\t Paragraph {para_count} -', current_paragraph, '\n')
                current_paragraph = ""
                is_paragraph = False
                para_count += 1
            current_heading = line
            is_heading = True

        elif line.strip():
            if is_heading:
                headings.append(current_heading.strip(' '))
                # print(f'Heading {headings_count} -  ', current_heading, '\n')

                current_heading = ""
                is_heading = False
                headings_count += 1

            current_paragraph += line + "\n\n"
            is_paragraph = True
    log_idp.info(f'Current Heading count {len(headings)} :: {current_heading}')
    log_idp.info(f'Current Paragraph count {len(paragraphs)} {current_paragraph}\n Loop end')

    '''if current_heading and heading_appended_flag is False:
        headings.append(current_heading.strip(' '))
        heading_appended_flag = True


    if current_paragraph and heading_appended_flag is True:
        heading_appended_flag = False
        paragraphs.append(current_paragraph)
        '''
    # print(f'\t Paragraph {para_count}:\n', current_paragraph)
    #
    # print('===========Total number of headings===========',headings_count)
    # print('===========Total number of paragraphs===========',para_count-1)

    '''for count, head in enumerate(headings):
        log_idp.info(f'Heading{count} :: {head[count]}')
    for count, para in enumerate(paragraphs):
        log_idp.info(f'Para {count}:: {para[count]}')'''

    log_idp.info(f'\nCount: {len(headings)}\nHeads:: {headings}\nCount: {len(paragraphs)}\nPara:: {paragraphs}')
    return headings, paragraphs


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
        reader = easyocr.Reader(['en'], model_storage_directory=ocr_model, user_network_directory=ocr_network,
                                recog_network='custom_example')  # recog_network='custom_example' this needs to run only once to load the model into memory
        try:
            result = reader.readtext(loc, height_ths=0.2,
                                     ycenter_ths=0.3, width_ths=0.5, paragraph=True, decoder='wordbeamsearch',
                                     y_ths=0.2,
                                     x_ths=50)
        except Exception as e:
            result = ''  # Exception occured.
            log_idp.exception(f'OCR encountered error: {e}')
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

        file = open(f'{out_path}{filename}_OCR.txt', 'a')
        for (bbox, text) in result:  # , prob
            total_text += str(text) + '\n'
            file.write(str(text))
            file.write('\n')
        file.close()
        # Remove image file after processing to save space
        for filename in os.listdir(location):
            if filename.endswith('.png'):
                os.remove(f'{location}/{filename}')
                log_idp.info(f'Deleting OCR image file {filename}')

        '''
        cv2.namedWindow('PDF Output', cv2.WINDOW_NORMAL)
        cv2.imshow("PDF Output", image)
        cv2.waitKey(2)
        '''
    return str(total_text)


#
# def read_dic():
#     """
#     Read dictionary mappings for converting entity names to Normal text
#     :return: None
#     """
#     global key_mappings
#     key_mappings = {}
#     with open('./Outputs/EntityDic.text', 'r') as f:
#         for line in f:
#             key, value = line.split(',')
#             key_mappings.fromkeys(key)
#             key_mappings[key] = value


def ner(file_did, pdf, titles, im_loc, page_limits=(1, 0), threshold=0.75):
    """
    Takes PDF runs it page by page to extract its text using OCR or text extraction to run NER model and save its output as text summary and temp HTML render
    :param threshold: Threshold for NER class to be added
    :param file_did: Document ID from SQL
    :param page_limits: start and end page number for extraction
    :param pdf: PDF file location
    :param titles: Name of PDF file
    :param im_loc: Location for saving PNG image in case of scanned page
    :return: None
    """
    # with Pdf.open(pdf, allow_overwriting_input=True) as pdf_f:  # To decrypt Permission pdfs
    #     pdf_f.save()

    inProcess(True, titles)  # Set File in_Process flag to 1 in MSSQL
    log_idp.info(f'{titles} In Process flag = 1')

    data = ''  # Data string variable to save all text data of PDF
    tagger = SequenceTagger.load(
        ner_model_loc)  # 2048layers_std
    log_idp.info(f'NER Model : \n{tagger}')
    open(f"{out_path}{titles}_OCR.txt", "w").close()  # Clear/Wipe OCR txt file

    rsrcmgr = PDFResourceManager()
    retstr = BytesIO()
    codec = 'utf-8'
    laparams = LAParams(char_margin=15, line_margin=2, boxes_flow=1)
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pdf_read_binary = open(pdf, 'rb')
    parser = PDFParser(pdf_read_binary)  # For getting total pages
    document = PDFDocument(parser)  # For getting total pages
    pdf_total_pages = resolve1(document.catalog['Pages'])['Count']  # For making progress bar
    paras = []
    headings = {}
    total_sentences = []
    forbidden = [
        'applicable', 'standard', 'standards', 'accessory', 'accessories', 'pipe',
        'applicable standard', 'applicable standards', 'system', 'switch',
        'station', 'circuit', 'isolator', 'hdpe', 'mccb', 'breaker', 'pole', 'duct', 'fence', 'meter',
        'switchgear', 'bus', 'loose', 'lugs', 'pole',
        'transformer', 'surge', 'insulator', 'ring', 'smoke', 'lug',
        'ABBREVIATION', 'page', 'IS:'
    ]
    splitter = SegtokSentenceSplitter()
    ##############
    # Prediction #
    ##############
    start = page_limits[0]
    end = page_limits[1]

    if start == 0:  # if no page range defined set last page as last page of PDF
        end = pdf_total_pages
    if end == 0:
        end = pdf_total_pages
    # Timer to measure total time taken to process a PDF
    start_time = time.perf_counter()
    for pagenum, page in enumerate(pdfpage.PDFPage.get_pages(pdf_read_binary, check_extractable=True)):
        pagenum += 1  # To start page count from 1
        if int(start) <= pagenum <= int(end):
            interpreter.process_page(page)
            if len(retstr.getvalue()) < 50:  # OCR case
                log_idp.warning(f'Page {pagenum + 1} is OCR only')

                pdf2img(pdf, titles, pagenums=pagenum)  # Convert page to image
                log_idp.info(f'Page {pagenum + 1} converted for OCR-NER data')

                data += img_ocr(im_loc, titles)  # Get OCR from converted image
                log_idp.info(f'Page {pagenum + 1} OCR data stored')

                try:
                    page_tables = table_extraction(pdf, titles, pagenum,
                                                   'lattice_ocr')  # Run OCR based table extraction
                except Exception as e:
                    log_idp.exception(f'Error while extracting OCR Table: {e}')
                    page_tables = None
                    continue
            else:
                try:

                    '''text_outside_table = table_extraction(pdf, titles, pagenum,
                                                          'lattice')  # Returns list of tables in the specified page

                    if text_outside_table is not None:
                        text_outside = "\n".join(text_outside_table)
                        log_idp.info(f'Outside table data:: {str(text_outside)}')
                        t_headings, t_para = headings_para_split(str(text_outside))
                        headings.append(t_headings)
                        paras.append(t_para)
                    else:
                    '''
                    t_headings, t_para = headings_para_split(retstr.getvalue().decode('ascii', 'ignore'))
                    # for heading in t_headings:
                    #     headings.append(heading)
                    # for para in t_para:
                    #     paras.append(para)

                    for index_count, heading in enumerate(t_headings, start=0):
                        skip_flag = False
                        heading_splits = str(heading).strip(' ').split(' ')  # Split cable name into words
                        log_idp.info(f'|| Heading {index_count} :: {heading_splits} ||\n')
                        for heading_word in heading_splits:
                            if heading_word.lower() in forbidden:  # Filtering results of Cable Type
                                skip_flag = True
                                log_idp.info(f'Word found in forbidden list {heading_word} || Skip flag: {skip_flag}')
                            else:
                                headings[str(index_count)] = heading  # Add extraction positive heading to unique list

                        if skip_flag is not True:
                            try:
                                para = str(t_para[index_count])
                                sentences = splitter.split(para)
                                for num, sentence in enumerate(sentences):
                                    tagger.predict(sentence)  # Predict paragraph contents

                                    log_idp.info(
                                        f'Prediction (Heading:{index_count + 1}) (Sub sent: {num}): {sentence.to_tagged_string()}\n\n')

                                total_sentences += (
                                    index_count, sentences)  # Maintain total sentences prediction in one list
                                log_idp.info(f'Total sentences: {len(total_sentences)}')

                            except RuntimeError:
                                log_idp.critical(f'Too Large page CUDA OFM error')
                                continue
                    log_idp.info(
                        f'\nPage {pagenum} Total valid headings: {len(headings)} Sentences Count: {len(total_sentences)}')

                    # data += retstr.getvalue().decode('ascii', 'ignore')  # add extracted text from bytesIO to data variable
                    # data = data.replace('\x0c', ' ')  # Remove useless character

                except IndexError:
                    log_idp.exception(f'Error while extracting using Lattice method: Index Error')
                    # log_idp.info(f'Running with Lattice-OCR again')
                    log_idp.info(f'Moving to next page')
                    # table_extraction(pdf, titles, pagenum, 'lattice_ocr')
                    continue
                except Exception as e:
                    log_idp.exception(f'Error while extracting text Table: {e}')
                    table_extraction(pdf, titles, pagenum, 'lattice_ocr')
                    log_idp.info(f'Attempting OCR Table extraction')

            ##################################################################
            # NEW LOGIC                                                      #
            # Append table NER extractions to table csv file after each page #
            ##################################################################
            '''
            TABLE APPEND NER
            try:    # If page has a large diagram, catch Out of memory exception of CUDA
                if page_tables:
                    tok_table_lines, tok_line, extraction = [], None, []
                    for table in page_tables:
                        for line in table:
                            tok_table_lines.append(Sentence(line, use_tokenizer=True))
                    for tok_line in tok_table_lines:
                        tagger.predict(tok_line)
                    with open(f"C:/Data/Output/{titles} tables.csv", 'a', newline='', encoding="utf-8") as f:
                        for tok_line in tok_table_lines:
                            for entity in tok_line.get_spans('ner', min_score=threshold):
                                if str(entity.tag) != 'tenderid' and entity.tag != 'marking' and entity.tag != '<unk>':
                                    # print(f'-- Adding table extraction to CSV file --')
                                    extraction.append(f'"{entity.text} , {entity.tag}"')
                        if extraction:
                            f.write("-------------------------- , ---------------------- \n")
                            f.write('"Attribute , Type"')
                            f.write("\n")
                            res = list(OrderedDict.fromkeys(extraction))
                            for tags in res:
                                f.write(tags)
                                f.write("\n")
                            f.write("\n -------------------------- , ---------------------- \n")
                retstr.truncate(0)
                retstr.seek(0)
            except RuntimeError:
                print(f'Too Large page CUDA OFM error')
                retstr.truncate(0)
                retstr.seek(0)
            '''
        retstr.truncate(0)  # Clear byte stream for new page extraction
        retstr.seek(0)
        log_idp.info(f'Page {pagenum} Processed')

    # log_idp.info(f'\nCount: {len(heading)}\n<Final> HEADINGS:: {headings}\nXxXxX\nCount: {len(paras)}\nPARAS {paras}\n\n')
    log_idp.info('File read complete ')
    splitter = SegtokSentenceSplitter()
    # sentences = splitter.split(data)

    #############
    # NEW LOGIC #
    #############
    log_idp.info(f'Start Predictions')

    summary_file_path = f'{out_path}{titles}_summary_experimental.rtf'
    forbidden = [
        'applicable', 'standard', 'standards', 'accessory', 'accessories', 'pipe',
        'applicable standard', 'applicable standards', 'system', 'switch',
        'station', 'circuit', 'isolator', 'hdpe', 'mccb', 'breaker', 'pole', 'duct', 'fence', 'meter',
        'switchgear', 'bus', 'loose', 'lugs', 'pole',
        'transformer', 'surge', 'insulator', 'ring', 'smoke', 'lug',
        'ABBREVIATION', 'page'
    ]

    #################
    # NEW LOGIC END #
    #################
    log_idp.info(f'Predictions complete')

    ##################
    #   LOG OUTPUT   #
    ##################

    summary_file_path = f'{out_path}{titles}_summary.rtf'
    html_file_path = f'{out_path}{titles}_web.html'

    sen = {}  # Dictionary for removing duplicate sentences
    dic = {}  # Dictionary for saving output under Cable Name
    misc = {}  # Dictionary for packing and marking tags
    with open(summary_file_path, 'w', newline='', encoding="utf-8") as f:
        f.writelines(r'{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\fs40{\f0 Rooney Sans;}}}')

        log_idp.info('Writing values to summary file')
        log_idp.info(f'////////////////////////////////////////////////////////////////////////////////')
        log_idp.info(f'///////////////// E X T R A C T I O N    R E S U L T ///////////////////////')
        log_idp.info(f'////////////////////////////////////////////////////////////////////////////////\n')

        f.writelines(r'/////////////////////////////////////////////////////////////////////////////////////////////'
                     r'///////////////////////////////// \par')
        f.writelines(
            r'/////////////////////////// \b E X T R A C T I O N    R E S U L T \b0 /////////////////////'
            r'////// \par')
        f.writelines(r'/////////////////////////////////////////////////////////////////////////////////////'
                     r'///////////////////////////////////////\par')
        cable_name = None
        cable_list = ['cable', 'line', 'covered conductors', 'control cables']
        forbidden = [
            'applicable', 'standard', 'standards', 'accessory', 'accessories', 'pipe',
            'applicable standard', 'applicable standards', 'system', 'switch',
            'station', 'circuit', 'isolator', 'hdpe', 'mccb', 'breaker', 'pole', 'duct', 'fence', 'meter',
            'switchgear', 'bus', 'loose', 'lugs', 'pole',
            'transformer', 'surge', 'insulator', 'ring', 'smoke', 'lug',
            'ABBREVIATION'
        ]
        #########################################################
        #        CUSTOMISATION FOR TENDER SPECIFICATIONS        #
        ########################################################
        cable_flag = 1
        for count, list_item, in enumerate(total_sentences):
            if isinstance(list_item, int):
                # Heading refrence
                heading_ref = list_item

            if isinstance(list_item, list):
                for sentence in list_item:
                    sen.setdefault(sentence.to_plain_string(),
                                   [])  # Create list initialised dictionary where Key = sentence
                    log_idp.info(f'Sentence {count} in processing: {sentence.to_plain_string()}\n\n')
                    #########################
                    # CABLE TYPE IDENTIFIER #
                    #########################
                    '''
                    # Check if cable is in Accepted list for the sentence, set cable flag = 0 if true
                    for entity in sentence.get_spans('ner', min_score=0.7):
                        if str(entity.tag) == 'cableItype':
                            for y in cable_list:
                                if entity.text.lower().find(y) != -1 and len(entity) > 1:
                                    log_idp.info(
                                        f'Cable found :{entity.text.upper()} - Conf: {(round(entity.score, 4) * 100)}%')
                                    cable_flag = 1
                                    cable_name = entity.text.upper()
                                    break
                            continue
                        else:
                            continue
                    # Check if cable is in Forbidden list for the sentence, set cable flag = 0 if true
                    for entity in sentence.get_spans('ner', min_score=0.8):
                        if str(entity.tag) == 'cableItype':  # Check all entities if in Forbidden list
                            cable_splits = (entity.text.lower().split(' '))  # Split cable name into words
                            for cable_split in cable_splits:
                                for x in forbidden:  # Filtering results of Cable Type
                                    if cable_split.find(x) != -1 and len(cable_split) == len(x):
                                        log_idp.info(f'X Cable Type Rejected {cable_split.upper()} > {x.upper()} X')
                                        cable_flag = 0
                                        break
                                if cable_flag == 0:  # if cable word is detected in forbidden list, flag = 0 and break
                                    break
                            break
                        else:
                            continue
                    if cable_flag == 1 and cable_name is not None:  # If cable is present in sentence'''

                    for entity in sentence.get_spans('ner', min_score=threshold):
                        if entity.tag != 'cableItype' and str(entity.tag) != 'tenderid' and str(
                                entity.tag) != 'standard' and entity.tag != '<unk>':
                            if entity.tag in ['marking', 'packing'] and len(
                                    entity) > 2:  # Removing entity output and less than 2 word entities from final text for markings
                                misc.setdefault(entity.tag, [])
                                misc[entity.tag].append(
                                    f'{sentence.to_plain_string()}')  # Adding specific formatted line to final text file
                                log_idp.info(
                                    f'=== {entity.text} === {entity.tag} ::CONF:: {(round(entity.score, 4) * 100)}% :::')  # Debugging/CLI output
                                continue
                            elif (len(sentence.to_plain_string()) > len(
                                    entity.text) + 4) and entity.tag != 'marking' and entity.tag != 'packing':

                                # log_idp.info(f'\n= = = = = Cable Type  {cable_name.upper()} = = = = =\n')
                                ent = entity.tag.replace('I',
                                                         ' ')  # Replace insulation'I'sheath with a blank = insulation sheath

                                sen[sentence.to_plain_string()].append(
                                    # Adding to sentence dictionary to avoid multiple same sentences
                                    fr'\ul {ent.title()} : : {entity.text} \ul0 \par')
                                # f.writelines(f'> {entity.text}, {entity.tag}-[{(round(entity.score, 4) * 100)}%] \n')
                                # f.writelines(f'>> {sentence.to_original_text()}, {entity.tag} \n\n')
                                log_idp.info(
                                    f'=== {entity.text} === {ent} ::CONF:: {(round(entity.score, 4) * 100)}% :::')  # Debugging/CLI output
                    if len(sen[
                               sentence.to_plain_string()]) > 0:  # Only add sentence to Cable dictionary if it has Entities
                        heading_for_extraction = headings[str(heading_ref)].strip(' ').lower()
                        log_idp.info(f'HEADING under extraction :: |{heading_for_extraction}|')

                        dic.setdefault(heading_for_extraction,
                                       [])  # Initialise blank value list in cable type dictionary
                        print(dic)
                        dic[heading_for_extraction].append(sentence.to_plain_string())
                        log_idp.info(
                            f'<Sentence append>: {sentence.to_plain_string()}\n Heading reference: {heading_ref}')

        log_idp.info(f'SENTENCES APPENDED: {len(dic.values())}')
        ##############################
        # CABLE LEVEL FILTRATION    #
        ##############################
        newline = r" \par "
        index_cables = newline.join(dic.keys())
        f.writelines(r'\par ================== \par')
        f.writelines(r' \par INDEX OF CABLES: \par\par')
        f.writelines(fr'\b {index_cables} \b0  \par')
        f.writelines(r'\par ================== \par \par')
        for k, values in dic.items():
            f.writelines(
                r'-------------------------------------------------------------------------------------- \par \par')
            f.writelines(fr'CABLE TYPE: \b {k} \b0')
            f.writelines(r'\par____________________________________________________________ \par \par')
            for line in values:
                tags = sen[line]
                unq_tags = list(
                    OrderedDict.fromkeys(tags))  # To remove multiple same Keys from different similar sentences
                f.writelines(r'\par')
                for tag in unq_tags:
                    f.writelines(f'{tag}')
                f.writelines(fr'\par \b Sentence \b0:  {line} \par')
                f.writelines(r'\par \par')
        f.writelines(r'X----------------------------------X-----------------------------------X \par')
        log_idp.info(f'File processing complete')

        for k, v in misc.items():
            if len(v) > 0:
                res = list(OrderedDict.fromkeys(v))  # Remove multiple same Keys from different similar sentences
                f.writelines(r'\par____________________________________________________________\par')
                f.writelines(fr'\b{str(k).upper()} \b0 \par')
                f.writelines(r'____________________________________________________________\par')
                for count, tags in enumerate(res):
                    f.writelines(fr' \par \b {str(k).title()} {count + 1} \b0 : {tags} \par\par')
        f.writelines(r'\par X----------------------------------X-----------------------------------X \par')
        f.writelines(r'}\n\x00')

    colors = {
        "default": "#FF40A3",
        "O": "#ddd",
    }

    # Create HTML file for summary
    html_content = render_ner_html(total_sentences, title=titles, colors=colors)
    with open(html_file_path, "w") as final:
        url = 'file://' + final.name
        final.write(html_content)

    # Calculate total time taken to process, in seconds/60 to get minutes
    end_time = time.perf_counter()
    time_taken = int((end_time - start_time) / 60)

    #################
    # Upload to GCS #
    #################
    '''log_idp.info(f'GCP + File operations starting =!=')
    try:
        ###########################
        # SUMMARY FILE OPERATIONS #
        ###########################

        summary_name = f'{titles}_summary.rtf'
        table_name = f'{titles}_tables.csv'
        ocr_name = f'{titles}_OCR.txt'
        html_name = f'{titles}_web.html'
        log_idp.info(f'Adding files to GCP Output folder:  {bucket_name}//{summary_name}, {ocr_name}, {table_name}')

        upload_to_gcs(summary_name)
        store_output(file_did, summary_name)  # NER file
        delete_local_pdf(out_path, summary_name)

        log_idp.info(f'Process done for: {summary_name}')

        try:
            #######################
            # OCR FILE OPERATIONS #
            #######################

            upload_to_gcs(ocr_name)
            store_output(file_did, ocr_name)  # OCR file
            delete_local_pdf(out_path, ocr_name)

            log_idp.info(f'Process done for: {ocr_name}')

            try:
                ########################
                # HTML FILE OPERATIONS #
                ########################
                store_output(file_did, html_name)  # HTML file
                upload_to_gcs(html_name)
                delete_local_pdf(out_path, html_name)

                log_idp.info(f'Process done for: {html_name}')

                try:
                    #########################
                    # TABLE FILE OPERATIONS #
                    #########################
                    upload_to_gcs(table_name)
                    store_output(file_did, table_name)  # Table file
                    delete_local_pdf(out_path, table_name)
                    log_idp.info(f'Process done for: {table_name}')

                    isGenerated(True, titles, pdf_total_pages, time_taken)  # Change pdf status in SQL to generated
                    log_idp.info(
                        f'File did_ID: {file_did} Generated = 1 in SQL\n  Time Taken: {time_taken} minutes\n  Total Pages: {pdf_total_pages}\n\n=!=!=!=!==!=!=!=!\n')
                except FileNotFoundError:
                    log_idp.critical(f'No tables file created. Skipping upload for {table_name}')
                    isGenerated(True, titles, pdf_total_pages, time_taken)  # Change pdf status in SQL to generated
                    log_idp.info(
                        f'File did_ID: {file_did} Generated = 1 in SQL\n  Time Taken: {time_taken} minutes\n  Total Pages: {pdf_total_pages}\n\n=!=!=!=!==!=!=!=!\n')
                except Exception as e:
                    log_idp.critical(f'! Error while finishing up {table_name} file --  {str(e)}')
            except Exception as e:
                log_idp.critical(f'! Error while finishing up {html_name} file --  {str(e)}')
        except Exception as e:
            log_idp.exception(f'! Error while finishing up {ocr_name} file --  {str(e)}')
    except FileNotFoundError:
        log_idp.exception(f'No summary file created. Skipping upload for {summary_name}')
        pass
    except Exception as e:
        log_idp.exception(f'! Error while finishing up {summary_name} file --  {str(e)}')'''
    # Close pdf binary read.
    pdf_read_binary.close()


def log_file_params(start, end, filename, conf):
    """
    Parameters
    ----------
    start: start page
    end: last page
    filename: filename
    conf: minimum confidence to for NER engine
    Returns
    -------
    None
    """

    log_idp.info(f'-------------------------------------------------------------------------------')
    log_idp.info(f'Threshold Confidence : {conf}')
    log_idp.info(f'PDF to be evaluated  : {str(filename.upper())}')
    if start: log_idp.info(f'Page limits      : {start} - {end}')
    log_idp.info(f'|______________________________________________________________________________|')


def run_single_file(file_did, pdf_name, pdf_name_full, s_page=0, e_page=0, thresh=0.69):
    pdf_loc = f'{pdffile_path}{pdf_name}.pdf'

    pages = (s_page, e_page)
    log_file_params(s_page, e_page, pdf_name, thresh)  # display main menu with variables
    try:
        ner(file_did, pdf_loc, pdf_name, img_loc, pages, threshold=thresh)
        delete_local_pdf(pdffile_path, pdf_name_full)  # Delete PDF file from local storage
    except pdfminer.pdfparser.PDFSyntaxError:
        log_idp.exception(f'No /Root object! File not being detected as PDF. Skipping file . . .')
        isGenerated(True, pdf_name, 404, 404)  # Change pdf status in SQL to generated
        log_idp.info(f'Setting Generated flag to 1.')
        log_idp.info(f'Setting Page number and Time to 404 to show Error.')


def file_check_schedule():
    did_id, files = getFiles()
    log_idp.info(f'Files in pipeline {" ".join(files)}\n')
    if len(files) > 0:
        for file_num, single_file in enumerate(files):
            file_did = did_id[file_num]
            log_idp.info(f'File to be run {single_file}')
            try:
                download_from_gcs(single_file)  # Download PDF file form GCP storage and save in local storage
                log_idp.info(f'| | Download complete from GCP: {single_file}')
            except Exception as e:
                log_idp.exception(f'Error while downloading file from GCP: {single_file} || {e}')
            pdf_name = single_file[:-4]  # remove file extension name
            run_single_file(file_did, pdf_name, pdf_name_full=single_file)


if __name__ == '__main__':
    interval = 10
    print(f'Prototype Program start')
    # Run the function within the specified interval
    pdf_loc = 'C:\Data\logic_test.pdf'
    file_did = '9999999'
    pdf_name = 'logic_test'

    ner(file_did, pdf_loc, pdf_name, img_loc, threshold=0.70)

    # while True:
    #     print(f'Checking for new uploads <DEV ENVIRONMENT>')
    #     file_check_schedule()
    #     # Wait for the specified interval before running the function again
    #     time.sleep(interval)

"""
CLI command :
                E:\PycharmProjects\DL\venv\scripts\python.exe E:\PycharmProjects\DL\Doc_IMG-OCR\main.py//
"""
