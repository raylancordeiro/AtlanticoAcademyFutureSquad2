import PyPDF2
import os
import re

def read_pdf_file(file: str) -> str:
    """
        Read a file in pdf format

    Args:
        file (object): a string with the full path of the file

    Returns:
        A string with the contents of all pages of the read file
    """
    pdf_file = open(file, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    page_content_all= ''
    for i in range(number_of_pages):
        page_content = read_pdf.getPage(i).extractText()
        page_content_all += page_content
    return page_content_all

def load_pfds(patch_files: str) -> str:
    """
        Reads all PDF files that are in the specific directory

    Args:
        patch_files: a string with the full path of the directory

    Returns:
        A string with the contents of all pages of all read files
    """
    os.chdir(patch_files)
    file_base_all = ''
    for file in os.listdir():
        if file.endswith(".pdf"):
            file_base_all += read_pdf_file(os.path.join(patch_files, file))
    return file_base_all

def join_and_remove_breaks(base: str) -> list[str]:
    """
        Removes all line breaks and converts content to a string list

    Args:
        base: A string with the content of the read files

    Returns:
        A list of strings with the content of the read files
     """
    parsed = re.sub('\n', '', base)
    parsed = list(parsed.split(" "))
    return parsed