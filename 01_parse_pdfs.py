# Author: Sergej Levich
# Journal article: Sergej Levich and Lucas Knust, International Journal of Accounting Information Systems, https://doi.org/10.1016/j.accinf.2025.100750

from doc2data.pdf import PDFCollection

# create pdf collection & parse files
pdf_collection = PDFCollection(path_to_files = 'data/annual_reports/')
print(f"Number of PDFs to parse: {pdf_collection.count_source_files()}")
pdf_collection.parse_files()
pdf_collection.save('data/pdf_collection.pickle')
