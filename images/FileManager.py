import csv


# DMSO csv FILE
file_path="/Volumes/My Passport/PFM/DMSO.csv"
raw_files_path="/Volumes/My Passport/PFM/raw_files"
delimiter = ";"

def process_csv(file, delim):
    with open(file, encoding='UTF-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = -1
        for row in csv_reader:
            if line_count == -1:
                line_count += 1
            else:
                print(f'\tCOMPOUND: {row[11]}, PathName DAPI: {row[3]}, Image DAPI: {row[2]} ')
                # ﻿TableNumber, 0
                # ImageNumber, 1
                # Image_FileName_DAPI, 2
                # Image_PathName_DAPI, 3
                # Image_FileName_Tubulin, 4
                # Image_PathName_Tubulin, 5
                # Image_FileName_Actin, 6
                # Image_PathName_Actin, 7
                # Image_Metadata_Plate_DAPI, 8
                # Image_Metadata_Well_DAPI, 9
                # Replicate, 10
                # Image_Metadata_Compound, 11
                # Image_Metadata_Concentration, 12
                line_count += 1
        print(f'Processed {line_count} images.')

process_csv(file_path, delimiter)