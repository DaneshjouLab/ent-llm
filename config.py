# Constants (e.g., PROJECT_ID, CPT codes)

# GCP Configuration
PROJECT_ID = "som-nero-phi-roxanad-entllm"

# List of datasets to be combined
DATASET_IDS = [
    'starr_phi_confidential_stride_79984_ChronicSinusitis_2016',
    'starr_phi_confidential_stride_79984_ChronicSinusitis_2017',
    'starr_phi_confidential_stride_79984_ChronicSinusitis_2018',
    'starr_phi_confidential_stride_79984_ChronicSinusitis_2019',
    'starr_phi_confidential_stride_79984_ChronicSinusitis_2020_21',
    'starr_phi_confidential_stride_79984_ChronicSinusitis_2022_23',
    'starr_phi_confidential_stride_79984_ChronicSinusitis_2024_25',
]

# Name of tables to load from each dataset
DATA_TABLES = [
    'demographics',
    'clinical_note',
    'procedures',
    'labs',
    'med_orders',
    'radiology_report'
]

# Clinical note types and titles
CLINICAL_NOTE_TYPES = {
    'Other Note', 'Other', 'Progress Note, Outpatient',
    'Operative/Procedure Report', 'Consultation Note',
    'History and Physical', 'Progress Note, Inpatient',
    'Progress/Discharge/Transfer Summary', 'Discharge/Transfer Summary'
}

CLINICAL_NOTE_TITLES = {
    'Procedure Note', 'Health Plan Operations CM Note', 'Airway',
    'OUTSIDE RECORDS', 'PROC CT SCAN INFORMATION', 'Operative Note',
    'Consult Follow-Up', 'Interval H&P Note', 'Unmapped External Note',
    'OUTSD CLINIC VISITS/CONSULTS/NOTES', 'Surgical Procedure',
    'Tertiary Survey', 'Admission H&P', 'H&P Interval', 'LARYNGOSCOPY ENT',
    'Pre-Op H&P', 'SHC CT RESULT', 'H&P Preop', 'Procedures',
    'IN CLINIC VNA - CT - MAXILLOFACIAL AREA', 'ORD CONSULT REQUEST',
    'IMAGE ONLY - ENT ENDOSCOPY', 'Care Plan Note', 'Clinic Support Note',
    'Clinic Visit', 'Consults', 'H&P', 'NASAL ENDOSCOPY ENT',
    'Operative Report', 'Progress Notes', 'Sign Out Note'
}

# Radiology report types and titles
RADIOLOGY_REPORT_TYPE = {'CT'}
RADIOLOGY_REPORT_TITLE = {
    'CT ENT SINUS WO IV CONTRAST', 'CT SINUS LIMITED', 'CT FACIAL BONES OR SINUS', 
    'CT FACIAL SINUSES O CONTRAST'
}

# CPT codes and keywords
SURGERY_CPT_CODES = {
    '31253', '31254', '31255', '31256', '31257', '31259', '31267',
    '31276', '31287', '31288', '31240'
}
LAB_KEYWORDS = {'cbc', 'eosinophil count', 'eosinophil %', 'bmp', 'cmp', 'ige', 'igg', 'iga', 'igm', 'sinus culture', 'nasal culture'}
MED_KEYWORDS = {
    'amoxicillin', 'amoxicillin-clavulanate', 'doxycycline', 'trimethoprim-sulfamethoxazole', 'clindamycin',
    'levofloxacin', 'cefdinir', 'moxifloxacin', 'cefuroxime', 'azithromycin', 'mupirocin', 'gentamicin',
    'tobramycin', 'vancomycin', 'prednisone', 'methylprednisone', 'dexamethasone', 'budesonide',
    'mometasone', 'fluticasone', 'azelastine', 'saline rinse'
}
DIAGNOSTIC_ENDOSCOPY_CPT_CODES = {'31231', '31237'}