#!/usr/bin/env python
"""patient-data.py: Match patient survival data to TCGA melanoma samples.

The patient data is available from the melanoma paper data portal:
https://tcga-data.nci.nih.gov/docs/publications/skcm_2015/

It is in two files. First is the patient data matching patient IDs to
clinical and survival data. It is in an Excel spreadsheet:
https://tcga-data.nci.nih.gov/docs/publications/skcm_2015/Table_S1D.xlsx

Second is the table matching sample barcode to patient ID. It is in
tab-delimited format:
https://tcga-data.nci.nih.gov/docs/publications/skcm_2015/SKCM_Final_Data_Freeze.txt

We want a single file matching the column IDs from the counts table to
survival data. This script achieves that.
"""

import pandas as pd


def remove_01(string):
    return string[:-3]


def patient_xlsx2df(filename):
    """Read the patient-centric spreadsheet into a pandas dataframe.

    Parameters
    ----------
    filename : string
        The path to the input Excel file.

    Returns
    -------
    patients : pandas DataFrame
        The patient data, including only interesting columns.
    """
    patients = pd.read_excel(filename,
                             skiprows=1,  # first row is obnoxious title
                             index_col=0,
                             na_values=['NA', '[Not Available]', '-'])
    # Rename the columns with the most egregious names.
    patients.rename(columns={'CURATED_MELANOMA_SPECIFIC_VITAL_STATUS [0 = '
                             '"ALIVE OR CENSORED"; 1 = "DEAD OF MELANOMA"]':
                                                    'melanoma-dead',
                             'CURATED_TCGA_DAYS_TO_DEATH_OR_LAST_FU':
                                                    'melanoma-survival-time',
                             'RNASEQ-CLUSTER_CONSENHIER':
                                                    'original-clusters'},
                    inplace=True)
    # Keep only "interesting" columns.
    patients = patients[['UV-signature', 'original-clusters',
                         'melanoma-survival-time', 'melanoma-dead']]
    # patient IDs have a '-01' suffix in this spreadsheet not present in
    # the barcode table. We remove it here for later matching.
    patients.index = [x[:-3] for x in patients.index]
    patients.index.name = 'PatientID'
    return patients


def barcodes_tsv2df(filename):
    """Read the sample barcode info spreadsheet into a pandas dataframe.

    Parameters
    ----------
    filename : string
        The path to the input tab-delimited text file.

    Returns
    -------
    barcodes2patients : dict
        A dictionary mapping sample barcodes to patients.
    """
    barcodes = pd.read_csv(filename, sep='\t')
    patients = barcodes['#PARTICIPANT']
    barcodes = barcodes['UUID_ALIQUOT']
    return dict(zip(barcodes, patients))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('usage: {} <patients> <barcodes> <counts>'.format(sys.argv[0]))
        sys.exit()
    patients_fn, barcodes_fn, counts_fn = sys.argv[1:4]
    patients = patient_xlsx2df(patients_fn)
    barcodes2patients = barcodes_tsv2df(barcodes_fn)
    counts = pd.read_csv(counts_fn, index_col=0)
    patients.to_csv('patients.csv')
    counts.columns = [barcodes2patients.get(b, b) for b in counts.columns]
    counts.to_csv(counts_fn[:-3] + 'remapped.csv')
