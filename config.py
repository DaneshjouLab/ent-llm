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

# Surgery keywords
STRONG_SURGICAL_PHRASES = [
    r'surgical\s+intervention(?:\s+for\s+\w+\s+sinusitis)?',
    r'surgical\s+treatment(?:\s+of\s+\w+\s+sinusitis)?',
    r'proceed\s+with\s+surgical\s+intervention(?:\s+for\s+\w+\s+sinusitis)?',
    r'surgical\s+management(?:\s+of\s+(?:\w+\s+)*(?:sinusitis|crs))?',
    r'plan\s+for(?:\s+\w+)*\s+sinus\s+surgery',
    r'scheduled?\s+for(?:\s+\w+)*\s+sinus\s+surgery',
    r'candidate\s+for(?:\s+\w+)*\s+(?:nasal|sinus)\s+surgery',
    r'patient\s+(?:was\s+)?(?:agreed?|elected|opted)(?:\s+to\s+proceed)?(?:\s+with)?(?:\s+\w+)*\s+sinus\s+surgery',
    r'(?:plan|proceed|scheduled?|recommended?|candidate).{0,50}\b(?:FESS|ESS)\b',
    r'\b(?:FESS|ESS)\b\s+(?:is\s+)?(?:recommended|planned|scheduled|indicated)',
    r'(?:considering?|planning\s+for)\s+(?:FESS|ESS)',
]

WEAK_SURGICAL_PHRASES = [
    r'surgical\s+planning',
    r'surgical.{0,20}(?:planning|plan|discussion)',
    r'^(?:assessment\s+and\s+plan|plan):',  # anchored to start of chunk

    # Discussion/decision-making - FIXED
    r'surgery.{0,20}(?:discussed?|discussion)',
    r'(?:sinus\s+)?surgery\s+was\s+discussed',
    r'consider\s+(?:endoscopic\s+)?surgery',
    r'patient\s+agrees?\s+(?:with\s+(?:the\s+)?plan)',
    r'we\s+(?:have\s+)?discussed\s+(?:sinus\s+)?surgery',
    r'consented?\s+(?:to|for)\s+(?:sinus\s+)?surgery',
    r'referred\s+to\s+ENT\s+for\s+(?:evaluation\s+and\s+)?surgery',

    # Abbreviations and procedure types
    r'\bESS\b',
    r'\bFESS\b',
    r'\bSEPT\b',
    r'\bESS/FESS\b',
    r'endoscopic\s+sinus\s+surgery',
    r'functional\s+endoscopic\s+sinus\s+surgery',
    r'\bseptoplasty\b',
    r'\bturbinate\s+reduction\b',
    r'\bturbinectomy\b',
    r'\bballoon\s*sinuplasty\b',
    r'\bpolypectomy\b',
]

