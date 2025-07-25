import logging
from typing import List, Dict, Optional
import re
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JambQuestionExtractor:
    def __init__(self):
        self.question_patterns = [
            re.compile(r'(?:^|\n)\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\n\s*SECTION\s+[IVXLCDM]+|\n\s*Questions\s+\d+-\d+|\n\s*Question\s+\d+|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'(?:^|\n)\s*(?:QUESTION|Q)\s*(\d+)\:?\s*(.+?)(?=\n\s*(?:QUESTION|Q)\s*\d+\:?|\n\s*SECTION\s+[IVXLCDM]+|\n\s*Questions\s+\d+-\d+|\n\s*Question\s+\d+|$)', re.DOTALL | re.IGNORECASE),
            re.compile(r'(?:^|\n)\s*(\d+)\)\s*(.+?)(?=\n\s*\d+\)|\n\s*SECTION\s+[IVXLCDM]+|\n\s*Questions\s+\d+-\d+|\n\s*Question\s+\d+|$)', re.DOTALL | re.IGNORECASE),
        ]

        self.option_patterns = [
            re.compile(r'^\s*([A-Ea-e])\.\s*(.+?)(?=\n\s*[A-Ea-e]\.|\n\s*\d+\.|\Z)', re.MULTILINE),
            re.compile(r'^\s*([A-Ea-e])\)\s*(.+?)(?=\n\s*[A-Ea-e]\)|\n\s*\d+\.|\Z)', re.MULTILINE),
        ]
        
        self.subject_patterns = {
            'mathematics': r'(?i)(?:math|mathematics|maths|further\s*maths)',
            'english': r'(?i)(?:english|literature\s*in\s*english|use\s*of\s*english)',
            'physics': r'(?i)physics',
            'chemistry': r'(?i)chemistry',
            'biology': r'(?i)biology',
            'economics': r'(?i)economics',
            'geography': r'(?i)geography',
            'history': r'(?i)history',
            'government': r'(?i)government',
            'commerce': r'(?i)commerce',
            'accounting': r'(?i)(?:accounting|accounts|book\s*keeping)',
            'agricultural_science': r'(?i)(?:agricultural\s*science|agric)',
            'technical_drawing': r'(?i)(?:technical\s*drawing|tech\s*drawing)',
            'food_and_nutrition': r'(?i)(?:food\s*and\s*nutrition|nutrition)',
            'christian_religious_knowledge': r'(?i)(?:christian\s*religious\s*knowledge|crk|christianity)',
            'islamic_religious_studies': r'(?i)(?:islamic\s*religious\s*studies|irs|islam)',
            'civic_education': r'(?i)civic\s*education',
            'data_processing': r'(?i)data\s*processing',
            'computer_studies': r'(?i)computer\s*studies',
            'general_knowledge': r'(?i)(?:general\s*knowledge|gk|aptitude)',
        }
    
    def _clean_text(self, text: str) -> str:
        lines = text.splitlines()

        instructional_line_patterns = [
            re.compile(r'.*JAMB\s*(?:Past\s*Questions)?\s*(?:for\s+.*?)?\s*-\s*Uploaded\s*on\s*(?:https?://)?(?:www\.)?myschool\.ng.*', re.IGNORECASE),
            re.compile(r'.*(?:www\.)?myschool\.ng.*', re.IGNORECASE),
            re.compile(r'.*myschool\.ng.*', re.IGNORECASE),
            re.compile(r'^\s*UTME\s*\d{4}.*', re.IGNORECASE),
            re.compile(r'.*Joint\s+Admissions\s+and\s+Matriculation\s+Board.*', re.IGNORECASE),
            re.compile(r'.*Unified\s+Tertiary\s+Matriculation\s+Examination.*', re.IGNORECASE),
            re.compile(r'.*Name:.*', re.IGNORECASE),
            re.compile(r'.*Identification Number:.*', re.IGNORECASE),
            re.compile(r'.*\[\d+\s*marks?\].*', re.IGNORECASE),
            re.compile(r'.*Write your Name and Identification Number.*', re.IGNORECASE),
            re.compile(r'.*Answer\s+\w+\s+questions?.*', re.IGNORECASE),
            re.compile(r'.*choos(?:e|ing)\s+at\s+least\s+one\s+question\s+from\s+each\s+section.*', re.IGNORECASE),
            re.compile(r'.*All\s+questions\s+carry\s+equal.*', re.IGNORECASE),
            re.compile(r'^\s*\d{3,4}$'),
            re.compile(r'^\s*SECTION\s+[A-ZIVXLCDM]+\s*$', re.IGNORECASE),
            re.compile(r'^\s*Answer\s+at\s+least\s+one\s+question\s+from\s+this\s+section.*', re.IGNORECASE),
            re.compile(r'.*You\s+are\s+advised\s+to\s+start\s+each\s+section\s+on\s+a\s+new\s+page.*', re.IGNORECASE),
            re.compile(r'.*This\s+question\s+paper\s+consists\s+of.*pages.*', re.IGNORECASE),
            re.compile(r'.*Do\s+not\s+turn\s+over\s+this\s+page\s+until\s+you\s+are\s+told\s+to\s+do\s+so.*', re.IGNORECASE),
            re.compile(r'^\s*Instructions:?\s*.*', re.IGNORECASE),
            re.compile(r'^\s*Time:?\s*.*', re.IGNORECASE),
            re.compile(r'^\s*Paper\s+\d+.*', re.IGNORECASE),
            re.compile(r'^\s*Question\s+\d+-\d+\s*.*', re.IGNORECASE),
            re.compile(r'^\s*General\s+Instructions.*', re.IGNORECASE),
            re.compile(r'^\s*Use\s+HB\s+pencil\s+throughout.*', re.IGNORECASE),
            re.compile(r'^\s*\(\w\)\s*In\s+the\s+space\s+marked\s+Name.*', re.IGNORECASE),
            re.compile(r'^\s*\(\w\)\s*In\s+the\s+box\s+marked\s+Identification\s+Number.*', re.IGNORECASE),
            re.compile(r'^\s*\(\w\)\s*In\s+the\s+box\s+marked\s+Subject\s+Code.*', re.IGNORECASE),
            re.compile(r'^\s*\(\w\)\s*In\s+the\s+box\s+marked\s+Sex.*', re.IGNORECASE),
            re.compile(r'^\s*\d+\.\s*If\s+you\s+have\s+got\s+a\s+pre-printed\s+answer\s+sheet.*', re.IGNORECASE),
            re.compile(r'^\s*An\s+example\s+is\s+given\s+below.*', re.IGNORECASE),
            re.compile(r'^\s*PRINT\s+IN\s+BLOCK\'?LETTERS.*', re.IGNORECASE),
            re.compile(r'^\s*Surname\s+other\s+Names.*', re.IGNORECASE),
            re.compile(r'^\s*Su\s*$', re.IGNORECASE),
            re.compile(r'^\s*PART\s+\w+\s*$', re.IGNORECASE),
            re.compile(r'^\s*Objective\s*Test\s*$', re.IGNORECASE),
            re.compile(r'^\s*Essay\s*Test\s*$', re.IGNORECASE),
            re.compile(r'^\s*Answer\s+Part\s+I\s+in\s+your\s+Objective\s+Test\s+answer\s+sheet.*', re.IGNORECASE),
            re.compile(r'^\s*\d+\s*$', re.IGNORECASE),
            re.compile(r'^\s*[a-zA-Z]\s*$', re.IGNORECASE),
            re.compile(r'^\s*\(\s*[a-zA-Z]\s*\)\s*$', re.IGNORECASE),
            re.compile(r'^\s*\d+\s+\d+\s*$', re.IGNORECASE),
            re.compile(r'^\s*\d+\s*\d+\s*\d+\s*$', re.IGNORECASE),
        ]

        lines_to_keep = []
        for line in lines:
            is_instructional = False
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if not stripped_line.strip():
                continue

            for pattern in instructional_line_patterns:
                if pattern.search(stripped_line):
                    is_instructional = True
                    break
            if not is_instructional:
                lines_to_keep.append(stripped_line)

        cleaned_text = '\n'.join(lines_to_keep)

        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

        return cleaned_text

    def extract_subject(self, text: str, filename: str = "") -> Optional[str]:
        combined_text = f"{filename} {text}".lower()
        for subject, pattern in self.subject_patterns.items():
            if re.search(pattern, combined_text):
                return subject
        return None

    def extract_options(self, text_after_question: str) -> List[Dict]:
        options = []
        for pattern in self.option_patterns:
            matches = list(pattern.finditer(text_after_question))
            if matches:
                for i, match in enumerate(matches):
                    letter = match.group(1).upper()
                    option_text = match.group(2).strip()
                    if len(option_text) < 1:
                        continue
                    options.append({
                        'letter': letter,
                        'text': option_text
                    })
                break
        return options

    def determine_question_type(self, question_text: str, options: List[Dict]) -> str:
        text_lower = question_text.lower()
        if options:
            return 'multiple_choice'
        elif any(word in text_lower for word in ['calculate', 'find', 'solve', 'compute', 'determine the value of']):
            return 'calculation'
        elif any(word in text_lower for word in ['explain', 'describe', 'discuss', 'define', 'state', 'list', 'outline', 'differentiate', 'account for', 'suggest', 'identify', 'compare']):
            return 'essay'
        elif any(word in text_lower for word in ['true or false', 'correct or incorrect', 'identify the true statement']):
            return 'true_false'
        else:
            return 'short_answer'

    def extract_questions(self, text: str, source: str = "") -> List[Dict]:
        questions = []
        cleaned_text = self._clean_text(text)

        for pattern in self.question_patterns:
            matches = list(pattern.finditer(cleaned_text))
            
            if not matches:
                continue

            for match in matches:
                question_num_str = match.group(1).strip()
                question_content_raw = match.group(2).strip()

                try:
                    question_num = int(question_num_str)
                except ValueError:
                    continue

                if len(question_content_raw) < 10:
                    continue

                question_stem = question_content_raw
                options = []
                options_text = ""

                options_header_match = re.search(r'(?i)\bOptions\b', question_content_raw)
                first_option_marker_match = re.search(r'^\s*([A-Ea-e][\.\)])', question_content_raw, re.MULTILINE)

                split_index = -1
                if options_header_match and first_option_marker_match:
                    if first_option_marker_match.start() > options_header_match.end():
                        split_index = options_header_match.end()
                    else:
                        split_index = first_option_marker_match.start()
                elif first_option_marker_match:
                    split_index = first_option_marker_match.start()
                elif options_header_match:
                    split_index = options_header_match.end()
                
                if split_index != -1:
                    question_stem = question_content_raw[:split_index].strip()
                    options_text = question_content_raw[split_index:].strip()
                    options = self.extract_options(options_text)
                    if options and len(question_stem) < 10 and not re.match(r'(?i)\boptions\b', question_stem):
                        pass
                    elif not options:
                        question_stem = question_content_raw
                        options_text = ""
                else:
                    if len(question_content_raw) > 30:
                        options = self.extract_options(question_content_raw)
                        if options:
                            first_option_letter = options[0]['letter']
                            first_option_pattern_match = re.search(r'(?m)^\s*' + re.escape(first_option_letter) + r'[\.\)]', question_content_raw, re.IGNORECASE)
                            if first_option_pattern_match:
                                question_stem = question_content_raw[:first_option_pattern_match.start()].strip()

                question_stem = re.sub(r'(?i)\bOptions\b', '', question_stem).strip()

                question_type = self.determine_question_type(question_stem, options)
                question_id = hashlib.md5(f"{source}-{question_num}-{question_stem}-{str(options)}".encode()).hexdigest()

                question_data = {
                    'question_number': question_num,
                    'question_text': question_stem,
                    'question_type': question_type,
                    'options': options,
                    'source': source,
                    'question_id': question_id,
                }
                questions.append(question_data)

            if questions:
                logger.info(f"Extracted {len(questions)} questions using pattern: {pattern.pattern[:50]}...")
                break

        if not questions and len(cleaned_text) > 100:
            logger.warning(f"No meaningful questions found after cleaning for source: {source}. Cleaned length: {len(cleaned_text)}")
        return questions