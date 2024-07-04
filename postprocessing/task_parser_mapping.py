from parsers import *

"""
Here we list all the task parsers. We can expand to multiple different
parsers for different models on a task by replacing the parser with a
dictionary of model name to parser. Further, we can expand that to
multiple different parsers for each instruction (in an individual task
and model) turning it into a dictionary from each annotator's name to a
parser. Also, the key "generally" will be used to give parser for any
model or annotator who does not otherwise have an entry in the dictionary.
In cases where we do not have a parser, the parser will be None.
"""


def get_parser(task, model=None, annotator=None):
    parser = task_parsers[task]
    if isinstance(parser, dict):
        assert model is not None
        if model not in parser.keys():
            parser = parser["generally"]
        else:
            parser = parser[model]
        if isinstance(parser, dict):
            assert annotator is not None
            if annotator not in parser.keys():
                parser = parser["generally"]
            else:
                parser = parser[annotator]
    else:
        parser = task_parsers[task]()
    return parser


def get_yes_no_parser(label0=0, label1=1):
    return ClassificationBinaryParser(
        class_info=[
            ("(yes|Yes|YES|does meet|do meet)([\s,.]+.*)?", label1),
            (None, label0),
        ],
        class_priority=[label1, label0],
    )


def get_multi_class_parser(label0=0, label1=1, label2=2, label3=3):
    return ClassificationMultiParser(
        class_info=[
            ("(unmentioned|Unmentioned|UNMENTIONED)([\s,.]+.*)?", label0),
            ("(questionable|Questionable|QUESTIONABLE)([\s,.]+.*)?", label1),
            ("(absent|Absent|ABSENT)([\s,.]+.*)?", label2),
            ("(present|Present|PRESENT)([\s,.]+.*)?", label3),
        ],
        class_priority=[label3, label2, label1, label0],
    )


task_parsers = {
    "Cohort Abdominal Classification": {
        "generally": get_yes_no_parser(),
        "flan-ul2": {
            "generally": get_yes_no_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Cohort Alcohol Abuse Classification": {
        "generally": get_yes_no_parser(),
        "flan-ul2": {
            "generally": get_yes_no_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Cohort Drug Abuse Classification": {
        "generally": get_yes_no_parser(),
        "flan-ul2": {
            "generally": get_yes_no_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Cohort English Classification": {
        "generally": get_yes_no_parser(),
        "flan-ul2": {
            "generally": get_yes_no_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Cohort Make Decisions Classification": {
        "generally": get_yes_no_parser(),
        "flan-ul2": {
            "generally": get_yes_no_parser(),
            "Jennifer_Fury": None,
        },
    },
    "MIMIC Mortality Prediction": {
        "generally": get_yes_no_parser(),
        "flan-ul2": {
            "generally": get_yes_no_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Obesity Classification": {
        "generally": get_multi_class_parser(),
        "flan-ul2": {
            "generally": get_multi_class_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Obesity Co-Morbidity Classification (Asthma)": {
        "name": "asthma_classification",
        "generally": get_multi_class_parser(),
        "flan-ul2": {
            "generally": get_multi_class_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Obesity Co-Morbidity Classification (CAD)": {
        "name": "cad_classification",
        "generally": get_multi_class_parser(),
        "flan-ul2": {
            "generally": get_multi_class_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Obesity Co-Morbidity Classification (Diabetes Mellitus)": {
        "name": "diabetes_mellitus_classification",
        "generally": get_multi_class_parser(),
        "flan-ul2": {
            "generally": get_multi_class_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Readmission Classification": {
        "generally": get_yes_no_parser(),
        "flan-ul2": {
            "generally": get_yes_no_parser(),
            "Jennifer_Fury": None,
        },
    },
    "Medication Extraction": ExtractionParser,
    "Drug Extraction": ExtractionParser,
    "Concept Treatment Extraction": ExtractionParser,
    "Concept Problem Extraction": ExtractionParser,
    "Concept Test Extraction": ExtractionParser,
    "Risk Factor CAD Extraction": ExtractionParser,
    # "Risk Factor Diabetes Extraction",
    # "Risk Factor Family History Extraction",
    # "Risk Factor Hyperlipidemia Extraction",
    # "Risk Factor Hypertension Extraction",
    # "Risk Factor Medication Extraction",
    # "Risk Factor Obese Extraction",
    # "Risk Factor Smoker Extraction",
    # "Concept ADE Extraction",
    # "Concept Dosage Extraction",
    # "Concept Drug Extraction",
    # "Concept Duration Extraction",
    # "Concept Form Extraction",
    # "Concept Frequency Extraction",
    # "Concept Reason Extraction",
    # "Concept Strength Extraction",
    # "Assertion Classification",
    # "Smoker Classification",
    # "Relation Drug Classification",
    # "Temporal Relations Classification",
    # "Coreference Resolution GT",
    # "Medication Information Extraction",
    # "Coreference Resolution Extraction",
    # "Medication Dosage Extraction",
    # "Medication Duration Extraction",
    # "Medication Frequency Extraction",
    # "Medication Mode Extraction",
    # "Relation Classification",
    # "Medication Reason Extraction",
    # "Event Extraction",
    # "Expression Extraction",
}
