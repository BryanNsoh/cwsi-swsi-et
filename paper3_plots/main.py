import os
from matplotlib.backends.backend_pdf import PdfPages
from utilities import connect_db, DATABASE_PATH, PDF_OUTPUT_PATH
from figures.figure3 import generate_figure3
from figures.figure4 import generate_figure4
from figures.figure5 import generate_figure5
from figures.figure6 import generate_figure6
from figures.figure7 import generate_figure7
from figures.figure8 import generate_figure8
from figures.figure9 import generate_figure9
from figures.figure10 import generate_figure10
from figures.figure11 import generate_figure11
from figures.figure12 import generate_figure12
from datetime import datetime

def main():
    """Main function to generate all figures and compile them into a single PDF."""
    conn = connect_db(DATABASE_PATH)
    if conn is None:
        print("Failed to connect to the database. Exiting.")
        return

    with PdfPages(PDF_OUTPUT_PATH) as pdf:
        # Generate Figures
        generate_figure3(conn, pdf)
        generate_figure4(conn, pdf)
        generate_figure5(conn, pdf)
        generate_figure6(conn, pdf)
        generate_figure7(conn, pdf)
        generate_figure8(conn, pdf)
        generate_figure9(conn, pdf)
        generate_figure10(conn, pdf)
        generate_figure11(conn, pdf)
        generate_figure12(conn, pdf)
        # Skip figure 13 for now
        # generate_figure13(conn, pdf)  # Commented out until neutron probe data is available

        # Optional: Add PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Irrigation and Stress Index Analysis Figures'
        d['Author'] = 'Your Name'
        d['Subject'] = 'Generated by main.py'
        d['Keywords'] = 'Irrigation, CWSI, SWSI, Water Use Efficiency, Agriculture'
        d['CreationDate'] = datetime.now()
        d['ModDate'] = datetime.now()

    print(f"All figures have been compiled into {PDF_OUTPUT_PATH}")
    conn.close()
    print("Database connection closed.")

if __name__ == "__main__":
    main()
