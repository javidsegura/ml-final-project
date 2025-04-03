import os
import zipfile
import pandas as pd

zip_filename = "AndMal2020-Dynamic-BeforeAndAfterReboot.zip"

extract_folder = "Extracted_Files"

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

csv_files = [os.path.join(root, file)
             for root, _, files in os.walk(extract_folder)
             for file in files if file.endswith(".csv")]

df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
final_df = pd.concat(df_list, ignore_index=True)


final_df.to_csv("FinalDynamicDataset.csv", index=False)
print(len(final_df))
