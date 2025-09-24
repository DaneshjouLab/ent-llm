# Define "strong" and "weak" surgical planning phrases
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

# Compile regex patterns
strong_patterns = [re.compile(p, re.IGNORECASE) for p in STRONG_SURGICAL_PHRASES]
weak_patterns = [re.compile(p, re.IGNORECASE) for p in WEAK_SURGICAL_PHRASES]
all_patterns = strong_patterns + weak_patterns